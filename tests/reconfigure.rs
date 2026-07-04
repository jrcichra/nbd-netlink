//! Exercises `NBD_CMD_RECONFIGURE` against a real kernel NBD device: connect,
//! kill the serving socket (simulating the userspace process dying), then
//! reconfigure a fresh socket onto the *same* device index using the backend
//! identifier ("cookie") as proof we're reattaching the right device — and
//! confirm I/O that was blocked mid-flight completes instead of erroring.
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
const COOKIE: &str = "nbd-netlink-reconfigure-test";
const DEVICE_INDEX: u64 = 0;
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

fn dev_path(index: u32) -> String {
    format!("/dev/nbd{index}")
}

fn read_backend_cookie(index: u32) -> Result<String> {
    Ok(std::fs::read_to_string(format!("/sys/block/nbd{index}/backend"))
        .context("read backend sysfs attr")?
        .trim()
        .to_string())
}

/// Open the device and read back exactly what was just written, proving the
/// kernel is actually routing I/O to our backing store through this device.
fn roundtrip_io(index: u32, pattern: u8) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(dev_path(index))?;
    let buf = vec![pattern; BLOCK_SIZE as usize];
    f.write_all(&buf)?;
    f.flush()?;
    f.seek(SeekFrom::Start(0))?;
    let mut readback = vec![0u8; BLOCK_SIZE as usize];
    f.read_exact(&mut readback)?;
    anyhow::ensure!(readback == buf, "readback did not match what was written");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reconfigure_preserves_device_and_in_flight_io() -> Result<()> {
    eprintln!("[checkpoint] opening netlink socket");
    let mut nbd = NBD::new().context("open NBD netlink socket (need CAP_NET_ADMIN)")?;
    let store = Arc::new(Mutex::new(vec![0u8; SIZE_BYTES as usize]));

    // --- initial connect ---
    eprintln!("[checkpoint] calling connect()");
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
    .context("initial connect")?;
    eprintln!("[checkpoint] connect() returned index {index}");
    drop(nbd); // netlink socket isn't needed while serving; recreated before reconfigure below

    assert_eq!(
        read_backend_cookie(index)?,
        COOKIE,
        "kernel did not echo back our backend identifier"
    );
    eprintln!("[checkpoint] cookie verified");

    our_side.set_nonblocking(true)?;
    let our_side = UnixStream::from_std(our_side)?;
    let serve_handle = tokio::spawn(serve(MemDevice(store.clone()), our_side));

    // Prove real I/O flows through the device before we touch anything.
    eprintln!("[checkpoint] starting initial roundtrip io");
    let idx_copy = index;
    tokio::task::spawn_blocking(move || roundtrip_io(idx_copy, 0xAB))
        .await
        .expect("join")
        .context("initial roundtrip io")?;
    eprintln!("[checkpoint] initial roundtrip io done");

    // --- simulate the serving process dying ---
    serve_handle.abort();
    let _ = serve_handle.await;
    eprintln!("[checkpoint] serve task aborted");

    // Kick off a write while nothing is listening on the kernel's socket; with
    // no NBD_ATTR_TIMEOUT set the kernel parks this request indefinitely
    // rather than erroring, which is exactly the guarantee we're testing.
    let idx_copy = index;
    let stuck_write = tokio::task::spawn_blocking(move || roundtrip_io(idx_copy, 0xCD));

    // Give it a moment to actually be in-flight against the dead socket
    // before we reattach.
    tokio::time::sleep(Duration::from_millis(300)).await;
    eprintln!("[checkpoint] stuck_write.is_finished() = {}", stuck_write.is_finished());
    assert!(
        !stuck_write.is_finished(),
        "write completed without a live socket — test isn't exercising the stall"
    );

    // --- reconfigure a fresh socket onto the same index ---
    eprintln!("[checkpoint] calling reconfigure()");
    let mut nbd = NBD::new().context("reopen NBD netlink socket")?;
    let (kernel_side, our_side) = StdUnixStream::pair().context("socketpair")?;
    let reconfigured_index = tokio::task::block_in_place(|| {
        NBDConnect::new()
            .size_bytes(SIZE_BYTES)
            .block_size(BLOCK_SIZE)
            .index(index as u64)
            .backend_identifier(COOKIE)
            .dead_conn_timeout_secs(DEAD_CONN_TIMEOUT_SECS)
            .reconfigure(&mut nbd, &[kernel_side])
    })
    .context("reconfigure")?;
    eprintln!("[checkpoint] reconfigure() returned index {reconfigured_index}");
    assert_eq!(reconfigured_index, index, "reconfigure landed on a different index");
    assert_eq!(
        read_backend_cookie(index)?,
        COOKIE,
        "cookie changed across reconfigure"
    );

    our_side.set_nonblocking(true)?;
    let our_side = UnixStream::from_std(our_side)?;
    let serve_handle = tokio::spawn(serve(MemDevice(store.clone()), our_side));

    // The write that was stuck against the dead socket must now complete
    // successfully — proving no data loss and no I/O error surfaced to the
    // caller across the reattach.
    eprintln!("[checkpoint] waiting on stuck_write to unblock");
    tokio::time::timeout(Duration::from_secs(10), stuck_write)
        .await
        .context("stuck write never completed after reconfigure")?
        .expect("join")
        .context("stuck write returned an error")?;
    eprintln!("[checkpoint] stuck_write unblocked and completed");

    // And the device is fully usable afterwards.
    let idx_copy = index;
    tokio::task::spawn_blocking(move || roundtrip_io(idx_copy, 0xEF))
        .await
        .expect("join")
        .context("post-reconfigure roundtrip io")?;
    eprintln!("[checkpoint] post-reconfigure roundtrip io done");

    serve_handle.abort();
    let _ = serve_handle.await;
    Ok(())
}

async fn serve(device: MemDevice, sock: UnixStream) {
    if let Err(e) = nbd_async::serve_nbd_send(device, sock).await {
        // Expected once the test aborts this task by dropping the socket.
        eprintln!("serve_nbd_send ended: {e}");
    }
}
