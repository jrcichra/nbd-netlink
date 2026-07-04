//! Exercises `NBD_CMD_DISCONNECT` against a real kernel NBD device that is
//! wedged (an I/O request stuck against a dead connection) — the scenario a
//! force-disconnect escape hatch exists for: releasing a device the kernel
//! would otherwise hold onto forever, at the cost of failing whatever I/O
//! was in flight. Asserts the disconnect call itself returns (rather than
//! hanging, which is exactly the bug the reconfigure() ACK fix caught) and
//! that the device is actually released (sysfs pid returns to 0) instead of
//! silently being a no-op.
//!
//! Requires CAP_NET_ADMIN and the `nbd` kernel module loaded; run as root.

use anyhow::{Context, Result};
use async_trait::async_trait;
use nbd_netlink::{NBDConnect, NBD};
use std::io;
use std::os::unix::net::UnixStream as StdUnixStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::UnixStream;

const SIZE_BYTES: u64 = 1 << 20;
const BLOCK_SIZE: u64 = 4096;
const COOKIE: &str = "nbd-netlink-disconnect-test";
const DEVICE_INDEX: u64 = 1;
const DEAD_CONN_TIMEOUT_SECS: u64 = 30;

struct MemDevice(Arc<Mutex<Vec<u8>>>);

#[async_trait]
impl nbd_async::BlockDeviceSend for MemDevice {
    async fn read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let store = self.0.lock().expect("lock");
        let start = offset as usize;
        buf.copy_from_slice(&store[start..start + buf.len()]);
        Ok(())
    }

    async fn write(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        let mut store = self.0.lock().expect("lock");
        let start = offset as usize;
        store[start..start + buf.len()].copy_from_slice(buf);
        Ok(())
    }
}

/// Read the pid sysfs attribute, treating a missing file as "0" — the
/// kernel's teardown (`nbd_config_put`) removes the `pid` attribute file
/// entirely once the device is fully released, rather than leaving it
/// behind reading "0".
fn read_pid(index: u32) -> Result<String> {
    match std::fs::read_to_string(format!("/sys/block/nbd{index}/pid")) {
        Ok(s) => Ok(s.trim().to_string()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok("0".to_string()),
        Err(e) => Err(e).context("read pid sysfs attr"),
    }
}

fn write_one_block(index: u32, pattern: u8) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(format!("/dev/nbd{index}"))?;
    let buf = vec![pattern; BLOCK_SIZE as usize];
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnect_releases_a_wedged_device() -> Result<()> {
    let mut nbd = NBD::new().context("open NBD netlink socket (need CAP_NET_ADMIN)")?;
    let store = Arc::new(Mutex::new(vec![0u8; SIZE_BYTES as usize]));

    let (kernel_side, our_side) = StdUnixStream::pair().context("socketpair")?;
    let index = tokio::task::block_in_place(|| {
        NBDConnect::new()
            .size_bytes(SIZE_BYTES)
            .block_size(BLOCK_SIZE)
            .index(DEVICE_INDEX)
            .backend_identifier(COOKIE)
            .dead_conn_timeout_secs(DEAD_CONN_TIMEOUT_SECS)
            .connect(&mut nbd, &[kernel_side])
    })
    .context("connect")?;

    assert_ne!(
        read_pid(index)?.parse::<u32>().context("parse pid")?,
        0,
        "pid should be nonzero right after connect"
    );

    our_side.set_nonblocking(true)?;
    let our_side = UnixStream::from_std(our_side)?;
    let serve_handle = tokio::spawn(serve(MemDevice(store.clone()), our_side));

    // Prove real I/O flows before wedging anything.
    let idx_copy = index;
    tokio::task::spawn_blocking(move || write_one_block(idx_copy, 0xAB))
        .await
        .expect("join")
        .context("initial write")?;

    // Kill the serving socket to strand a request against a dead connection
    // — the actual "wedged" state the force-disconnect escape hatch exists
    // for.
    serve_handle.abort();
    let _ = serve_handle.await;

    let idx_copy = index;
    let stuck_write = tokio::task::spawn_blocking(move || write_one_block(idx_copy, 0xCD));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !stuck_write.is_finished(),
        "write completed without a live socket - nothing to disconnect out of"
    );

    // Force-disconnect must itself return promptly, not hang — this is
    // exactly the bug the reconfigure() ACK fix caught, and disconnect has
    // the identical kernel-side shape (no explicit reply on success).
    tokio::task::block_in_place(|| nbd.disconnect(index as u64)).context("disconnect")?;

    // The stuck write must be released one way or another (most likely with
    // an I/O error, since force-disconnect intentionally aborts in-flight
    // requests) rather than hanging forever.
    tokio::time::timeout(Duration::from_secs(10), stuck_write)
        .await
        .context("stuck write never unblocked after disconnect")?
        .expect("join")
        .ok();

    // And the device is actually released, not just superficially ack'd:
    // sysfs pid should return to 0. The kernel's own teardown (workqueue
    // flush, refcount drop to zero) isn't necessarily complete the
    // microsecond disconnect()'s netlink call returns, so poll briefly
    // rather than asserting instantaneously.
    let mut pid = read_pid(index)?;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while pid != "0" && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
        pid = read_pid(index)?;
    }
    assert_eq!(pid, "0", "device pid did not return to 0 after disconnect");

    // The strongest proof it's actually free: a brand new connect() to the
    // same index — the real scenario tetra cares about, since after a
    // force-disconnect the next thing that happens is a fresh attach.
    let (kernel_side2, _our_side2) = StdUnixStream::pair().context("socketpair")?;
    let mut nbd2 = NBD::new().context("reopen NBD netlink socket")?;
    tokio::task::block_in_place(|| {
        NBDConnect::new()
            .size_bytes(SIZE_BYTES)
            .block_size(BLOCK_SIZE)
            .index(DEVICE_INDEX)
            .backend_identifier(COOKIE)
            .connect(&mut nbd2, &[kernel_side2])
    })
    .context("device did not accept a fresh connect() after disconnect")?;

    Ok(())
}

async fn serve(device: MemDevice, sock: UnixStream) {
    if let Err(e) = nbd_async::serve_nbd_send(device, sock).await {
        eprintln!("serve_nbd_send ended: {e}");
    }
}
