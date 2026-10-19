//! Thin safe-ish wrappers over device memory, streams, graphs and cuBLAS.

use std::ffi::{c_void, CStr};

use anyhow::{bail, Result};

use crate::ffi;

pub fn check(code: i32, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(ffi::dr_error_string(code)) }.to_string_lossy();
    bail!("{what}: CUDA error {code} ({msg})")
}

/// Status of a kernel launcher: its own argument check, then the launch error.
pub fn launched(code: i32, what: &str) -> Result<()> {
    if code != 0 {
        bail!("{what}: unsupported shape (launcher returned {code})");
    }
    if cfg!(debug_assertions) {
        check(unsafe { ffi::dr_last_error() }, what)?;
    }
    Ok(())
}

pub fn mem_info() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0usize, 0usize);
    check(unsafe { ffi::dr_mem_info(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Ok((free, total))
}

/// A device allocation. Pointers handed to kernels stay valid for its lifetime.
pub struct DevBuf {
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for DevBuf {}

impl DevBuf {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::dr_malloc(&mut ptr, bytes.max(256)) }, &format!("cudaMalloc({bytes})"))?;
        Ok(Self { ptr, bytes })
    }

    pub fn zeros(bytes: usize, st: &Stream) -> Result<Self> {
        let b = Self::alloc(bytes)?;
        b.zero(st)?;
        Ok(b)
    }

    pub fn from_bytes(data: &[u8], st: &Stream) -> Result<Self> {
        let b = Self::alloc(data.len())?;
        b.upload(data, st)?;
        Ok(b)
    }

    pub fn from_slice<T: Copy>(data: &[T], st: &Stream) -> Result<Self> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        Self::from_bytes(bytes, st)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn upload(&self, data: &[u8], st: &Stream) -> Result<()> {
        assert!(data.len() <= self.bytes);
        check(unsafe { ffi::dr_h2d(self.ptr, data.as_ptr() as *const c_void, data.len(), st.raw) }, "h2d")
    }

    pub fn upload_at<T: Copy>(&self, offset_bytes: usize, data: &[T], st: &Stream) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(offset_bytes + n <= self.bytes);
        check(
            unsafe { ffi::dr_h2d((self.ptr as *mut u8).add(offset_bytes) as *mut c_void, data.as_ptr() as *const c_void, n, st.raw) },
            "h2d",
        )
    }

    /// Synchronous copy back to the host (syncs the stream).
    pub fn download<T: Copy + Default>(&self, count: usize, st: &Stream) -> Result<Vec<T>> {
        self.download_at(0, count, st)
    }

    pub fn download_at<T: Copy + Default>(&self, offset_bytes: usize, count: usize, st: &Stream) -> Result<Vec<T>> {
        let mut v = vec![T::default(); count];
        let n = count * std::mem::size_of::<T>();
        assert!(offset_bytes + n <= self.bytes);
        check(
            unsafe { ffi::dr_d2h(v.as_mut_ptr() as *mut c_void, (self.ptr as *const u8).add(offset_bytes) as *const c_void, n, st.raw) },
            "d2h",
        )?;
        st.sync()?;
        Ok(v)
    }

    pub fn zero(&self, st: &Stream) -> Result<()> {
        check(unsafe { ffi::dr_memset(self.ptr, 0, self.bytes, st.raw) }, "memset")
    }

    /// Every byte set to `v` (0xff makes every i32 -1).
    pub fn fill_bytes(&self, v: u8, st: &Stream) -> Result<()> {
        check(unsafe { ffi::dr_memset(self.ptr, v as i32, self.bytes, st.raw) }, "memset")
    }

    pub fn ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Pointer `elems` elements of type T into the buffer.
    pub fn at<T>(&self, elems: usize) -> *mut T {
        assert!((elems * std::mem::size_of::<T>()) <= self.bytes);
        unsafe { (self.ptr as *mut T).add(elems) }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { ffi::dr_free(self.ptr) };
    }
}

pub struct Stream {
    pub raw: ffi::Stream,
}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { ffi::dr_stream_create(&mut raw) }, "cudaStreamCreate")?;
        Ok(Self { raw })
    }

    pub fn sync(&self) -> Result<()> {
        check(unsafe { ffi::dr_stream_sync(self.raw) }, "cudaStreamSynchronize")
    }
}

pub struct Graph {
    exec: ffi::GraphExec,
}

impl Graph {
    /// Capture everything `record` launches on `st` into a replayable graph.
    pub fn capture(st: &Stream, record: impl FnOnce() -> Result<()>) -> Result<Self> {
        check(unsafe { ffi::dr_graph_begin(st.raw) }, "cudaStreamBeginCapture")?;
        let recorded = record();
        let mut exec = std::ptr::null_mut();
        let ended = check(unsafe { ffi::dr_graph_end(st.raw, &mut exec) }, "cudaStreamEndCapture/Instantiate");
        recorded?;
        ended?;
        Ok(Self { exec })
    }

    pub fn launch(&self, st: &Stream) -> Result<()> {
        check(unsafe { ffi::dr_graph_launch(self.exec, st.raw) }, "cudaGraphLaunch")
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { ffi::dr_graph_destroy(self.exec) };
    }
}

pub struct Cublas {
    handle: *mut c_void,
}

impl Cublas {
    pub fn new(st: &Stream) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let r = unsafe { ffi::dr_cublas_create(&mut handle, st.raw) };
        if r != 0 {
            bail!("cublasCreate failed: {r}");
        }
        Ok(Self { handle })
    }

    /// C[m, n] = A[m, k] . W[n, k]^T + beta * C   (bf16 inputs, f32 output, row-major)
    pub fn gemm(&self, a: *const u16, w: *const u16, c: *mut f32, m: usize, n: usize, k: usize, beta: f32) -> Result<()> {
        let r = unsafe { ffi::dr_gemm(self.handle, a, w, c, m as i32, n as i32, k as i32, beta) };
        if r != 0 {
            bail!("cublasGemmEx failed: {r} (m={m} n={n} k={k})");
        }
        Ok(())
    }

    /// Same as `gemm` with f16 inputs.
    pub fn gemm_f16(&self, a: *const u16, w: *const u16, c: *mut f32, m: usize, n: usize, k: usize, beta: f32) -> Result<()> {
        let r = unsafe { ffi::dr_gemm_f16(self.handle, a, w, c, m as i32, n as i32, k as i32, beta) };
        if r != 0 {
            bail!("cublasGemmEx(f16) failed: {r} (m={m} n={n} k={k})");
        }
        Ok(())
    }
}

/// Page-locked host memory for fast, asynchronous copies.
pub struct Pinned<T: Copy> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Copy> Send for Pinned<T> {}

impl<T: Copy> Pinned<T> {
    pub fn new(len: usize) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        check(unsafe { ffi::dr_host_alloc(&mut p, len * std::mem::size_of::<T>()) }, "cudaMallocHost")?;
        Ok(Self { ptr: p as *mut T, len })
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }


    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy> Drop for Pinned<T> {
    fn drop(&mut self) {
        unsafe { ffi::dr_host_free(self.ptr as *mut c_void) };
    }
}

/// Page-locked, write-combined host memory that kernels read directly over PCIe. The CPU writes it once and never
/// reads it back (reads of write-combined memory are uncached and slow).
pub struct HostBuf {
    host: *mut c_void,
    dev: *mut c_void,
    bytes: usize,
}

unsafe impl Send for HostBuf {}

impl HostBuf {
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let (mut host, mut dev) = (std::ptr::null_mut(), std::ptr::null_mut());
        check(unsafe { ffi::dr_host_alloc_mapped(&mut host, &mut dev, data.len().max(256)) }, &format!("cudaHostAlloc({})", data.len()))?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), host as *mut u8, data.len()) };
        Ok(Self { host, dev, bytes: data.len() })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The address kernels use.
    pub fn dev<T>(&self) -> *mut T {
        self.dev as *mut T
    }

    /// The address host-to-device copies use.
    pub fn host<T>(&self) -> *const T {
        self.host as *const T
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        unsafe { ffi::dr_host_free(self.host) };
    }
}
