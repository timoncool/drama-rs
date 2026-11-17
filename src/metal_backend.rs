//! Apple-Silicon backend for `src/ffi.rs` (feature `metal`).
//!
//! How it maps to CUDA concepts:
//! - Unified memory: every "device" allocation is an `MTLBuffer` with shared storage,
//!   so its address is stable and host-visible. CPU reference code reads/writes it
//!   directly; MSL kernels take it via `set_buffer`.
//! - Execution is synchronous in v1: each launch commits and waits, so
//!   `dr_stream_sync` is a no-op. Graphs are recorded as closure lists (`dr_graph_*`)
//!   and replayed eagerly, which preserves `engine.rs` semantics exactly.
//! - Ops already verified on the GPU are marked MSL below; the rest are CPU reference
//!   implementations over the same host-visible buffers (bit-exact with
//!   `kernels/*.cu` by construction, except the sampler RNG, see below).

use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::ffi::{GraphExec, Half, Stream};

// Apple Accelerate (AMX) SGEMM for the prefill/codec GEMMs: one framework call
// replaces the naive triple loop (orders of magnitude faster on big shapes).
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32,
    );
}
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

// ---------------------------------------------------------------------------
// thread-safe Metal handle wrappers (Metal objects are internally synchronized)

#[derive(Clone)]
struct MDevice(metal::Device);
unsafe impl Send for MDevice {}
unsafe impl Sync for MDevice {}

#[derive(Clone)]
struct MQueue(metal::CommandQueue);
unsafe impl Send for MQueue {}
unsafe impl Sync for MQueue {}

#[derive(Clone)]
struct MLib(metal::Library);
unsafe impl Send for MLib {}
unsafe impl Sync for MLib {}

#[derive(Clone)]
struct MPipe(metal::ComputePipelineState);
unsafe impl Send for MPipe {}
unsafe impl Sync for MPipe {}

#[derive(Clone)]
struct MBuf(metal::Buffer);
unsafe impl Send for MBuf {}
unsafe impl Sync for MBuf {}

// ---------------------------------------------------------------------------
// device context

struct Ctx {
    device: MDevice,
    queue: MQueue,
    lib: MLib,
    pipes: Mutex<HashMap<&'static str, MPipe>>,
}

static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> &'static Ctx {
    CTX.get_or_init(|| {
        let device = metal::Device::system_default().expect("metal: no system default device");
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(MSL_SRC, &metal::CompileOptions::new())
            .unwrap_or_else(|e| panic!("metal: MSL compile failed: {e}"));
        Ctx { device: MDevice(device), queue: MQueue(queue), lib: MLib(lib), pipes: Mutex::new(HashMap::new()) }
    })
}

fn pipe(name: &'static str) -> MPipe {
    let c = ctx();
    if let Some(p) = c.pipes.lock().unwrap().get(name) {
        return p.clone();
    }
    let f = c.lib.0.get_function(name, None).unwrap_or_else(|e| panic!("metal: function {name}: {e}"));
    let p = MPipe(c.device.0.new_compute_pipeline_state_with_function(&f).unwrap_or_else(|e| {
        panic!("metal: pipeline {name}: {e}")
    }));
    c.pipes.lock().unwrap().insert(name, p.clone());
    p
}

// ---------------------------------------------------------------------------
// buffer registry: raw pointer -> (buffer, base). `DevBuf::at` creates interior
// pointers, so lookup finds the containing buffer and its offset.

static BUFS: Mutex<BTreeMap<usize, MBuf>> = Mutex::new(BTreeMap::new());

fn track(buf: &metal::Buffer) -> *mut c_void {
    let p = buf.contents() as *mut c_void;
    BUFS.lock().unwrap().insert(p as usize, MBuf(buf.clone()));
    p
}

fn lookup(ptr: *const c_void) -> Option<(MBuf, u64)> {
    if ptr.is_null() {
        return None;
    }
    let m = BUFS.lock().unwrap();
    let p = ptr as usize;
    let (&base, buf) = m.range(..=p).next_back()?;
    let off = p - base;
    if off > buf.0.length() as usize {
        return None;
    }
    Some((buf.clone(), off as u64))
}

fn require(ptr: *const c_void) -> (MBuf, u64) {
    lookup(ptr).expect("metal: pointer is not a tracked device buffer")
}

// ---------------------------------------------------------------------------
// pinned host memory

static HOST: Mutex<BTreeMap<usize, std::alloc::Layout>> = Mutex::new(BTreeMap::new());

// ---------------------------------------------------------------------------
// streams (synchronous execution; the id only routes graph capture)

static NEXT_STREAM: AtomicUsize = AtomicUsize::new(1);
static QUEUES: Mutex<BTreeMap<usize, MQueue>> = Mutex::new(BTreeMap::new());

// ---------------------------------------------------------------------------
// graph capture: record closures, replay eagerly (thread-local).
//
// Execution is synchronous, so capture only needs to live until the matching
// `dr_graph_end` on the SAME thread (this matches engine.rs: Engine::new records
// the graph and step() replays it on one thread; the codec never uses graphs).
// Cross-thread record/replay is rejected in dr_graph_end/dr_graph_launch.

type Op = Box<dyn Fn()>;

struct Capture {
    stream: usize,
    thread: std::thread::ThreadId,
    ops: Vec<Op>,
}

struct F32Cache { map: std::collections::HashMap<(usize, usize, usize), MBuf>, bytes: usize }
static F32CACHE: std::sync::OnceLock<Mutex<F32Cache>> = std::sync::OnceLock::new();
const F32CACHE_CAP: usize = 512 << 20;
fn f32cache() -> &'static Mutex<F32Cache> {
    F32CACHE.get_or_init(|| Mutex::new(F32Cache { map: std::collections::HashMap::new(), bytes: 0 }))
}

std::thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Capture>> = const { std::cell::RefCell::new(None) };
    // H1: pending (queue, command buffer, encoder) triples. dispatches encode
    // back-to-back; flush_gpu() ends/commits/waits them all. Buffers from the
    // same serial queue execute in commit order, so GPU->GPU chains need no
    // intermediate sync; CPU reads of GPU data must flush first.
    static PENDING: std::cell::RefCell<Vec<(usize, metal::CommandBuffer, metal::ComputeCommandEncoder, &'static str)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct GraphData {
    ops: Vec<Op>,
}

/// Run `f` now, or record it when `st` is being captured on this thread.
fn launch(st: Stream, f: impl Fn() + 'static) -> i32 {
    let key = st as usize;
    let mut f = Some(f);
    let captured = CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        match c.as_mut() {
            Some(cap) if cap.stream == key => {
                cap.ops.push(Box::new(f.take().expect("launch: closure taken twice")));
                true
            }
            _ => false,
        }
    });
    if captured {
        return 0;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f.expect("launch: closure missing"))) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

// ---------------------------------------------------------------------------
// small helpers shared by the CPU reference ops

/// A raw device pointer that may cross thread boundaries (`Send` instead of
/// the raw pointer, which is not `Send` on this toolchain). Sound when threads
/// touch disjoint memory, or shared memory read-only — the case in every
/// `par_for!` site below (each thread owns its row range; weights are read-only).
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
impl<T: Copy> SendPtr<T> {
    #[inline]
    unsafe fn read(self, i: usize) -> T {
        *self.0.add(i)
    }
    #[inline]
    unsafe fn write(self, i: usize, v: T) {
        *self.0.add(i) = v;
    }
    #[inline]
    fn is_null(self) -> bool {
        self.0.is_null()
    }
    #[inline]
    fn ptr(self) -> *mut T {
        self.0
    }
    /// Signed offset (for history reads before a buffer's nominal start).
    #[inline]
    fn at(self, i: isize) -> *mut T {
        unsafe { self.0.offset(i) }
    }
}

/// Parallel row-range loop over `0..n` with std threads. Expands at the call
/// site, so the worker closure only needs `Send` (raw device pointers qualify);
/// all other captures must be `Copy` as each thread gets its own copy.
/// Callers must ensure ranges write disjoint memory.
/// Persistent worker pool (H2): `par_for!` used to spawn N OS threads per
/// op (profiled as a top hotspot); workers now persist and take jobs from a
/// queue. Jobs are `FnOnce + Send`; the main thread blocks until its batch
/// completes. Re-entrant calls from workers run serially (TLS guard).
type Job = Box<dyn FnOnce() + Send + 'static>;
struct Pool {
    tx: std::sync::mpsc::Sender<Job>,
    pending: std::sync::Mutex<usize>,
    done: std::sync::Condvar,
    threads: usize,
}
std::thread_local! {
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        for _ in 0..threads {
            let rx = rx.clone();
            std::thread::spawn(move || {
                IN_POOL.with(|f| f.set(true));
                loop {
                    let job = rx.lock().unwrap().recv().unwrap();
                    job();
                }
            });
        }
        Pool { tx, pending: std::sync::Mutex::new(0), done: std::sync::Condvar::new(), threads }
    })
}
fn pool_run<F: FnOnce() + Send>(f: F) {
    let p = pool();
    *p.pending.lock().unwrap() += 1;
    // SAFETY: lifetime-erased like std::thread::scope — the caller always
    // joins via pool_wait() (enforced by WaitGuard on panic paths) before its
    // stack frame can exit, so borrowed captures stay alive.
    let f: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send>, Job>(Box::new(f)) };
    p.tx.send(f).unwrap();
}
/// Joins pending pool jobs on drop (panic safety for par_for!).
struct WaitGuard(bool);
impl Drop for WaitGuard {
    fn drop(&mut self) {
        if !self.0 {
            pool_wait();
        }
    }
}
fn pool_wait() {
    let p = pool();
    let mut g = p.pending.lock().unwrap();
    while *g > 0 {
        g = p.done.wait(g).unwrap();
    }
}
fn pool_job_done() {
    let p = pool();
    let mut g = p.pending.lock().unwrap();
    *g -= 1;
    if *g == 0 {
        p.done.notify_all();
    }
}
macro_rules! par_for {
    ($n:expr, |$lo:ident, $hi:ident| $body:block) => {{
        let n__ = $n;
        if n__ > 0 {
            let reenter__ = IN_POOL.with(|f| f.get());
            let threads__ = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(n__).max(1);
            if threads__ <= 1 || reenter__ {
                let ($lo, $hi) = (0usize, n__);
                $body
            } else {
                let chunk__ = (n__ + threads__ - 1) / threads__;
                let mut guard__ = WaitGuard(false);
                for t__ in 0..threads__ {
                    let ($lo, $hi) = (t__ * chunk__, ((t__ + 1) * chunk__).min(n__));
                    pool_run(move || {
                        $body
                        pool_job_done();
                    });
                }
                pool_wait();
                guard__.0 = true;
            }
        }
    }};
}

#[inline]
fn f32s(p: *const f32, n: usize) -> &'static [f32] {
    unsafe { std::slice::from_raw_parts(p, n) }
}

#[inline]
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
fn f16_to_f32(h: Half) -> f32 {
    half_to_f32(h)
}

fn half_to_f32(h: u16) -> f32 {
    // IEEE 754 binary16 -> f32.
    let s = (h & 0x8000) as u32;
    let e = (h & 0x7c00) as u32;
    let m = (h & 0x03ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 16
        } else {
            // subnormal: renormalize
            let mut m = m;
            let mut e = 127 - 15 + 1;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            (s << 16) | (e << 23) | (m << 13)
        }
    } else if e == 0x7c00 {
        (s << 16) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 16) | (((e >> 10) + (127 - 15)) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Round-to-nearest-even f32 -> bf16 bits (matches `__float2bfloat16`; NaN -> 0x7fc0).
fn f32_to_bf16_rn(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn rms_inv(x: &[f32], eps: f32) -> f32 {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    (ss / x.len() as f32 + eps).sqrt().recip()
}

// ---------------------------------------------------------------------------
// MSL kernels (MSL = Metal Shading Language)

const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline ushort f32_to_bf16_rn(float v) {
    uint b = as_type<uint>(v);
    if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) return (ushort)0x7fc0;
    return (ushort)(((b + 0x7fffu + ((b >> 16) & 1u)) >> 16));
}
inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }

struct NormP { uint rows; uint d; float eps; };
kernel void rmsnorm(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device float* y [[buffer(2)]],
                    device ushort* yb [[buffer(3)]],
                    constant NormP& p [[buffer(4)]],
                    uint3 tpg [[threads_per_threadgroup]],
                    uint3 tid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.x;
    if (r >= p.rows) return;
    uint d = p.d;
    float acc = 0.0f;
    for (uint i = tid.x; i < d; i += tpg.x) { float v = x[r*d+i]; acc += v*v; }
    threadgroup float part[256];
    part[tid.x] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(part[0] / (float)d + p.eps);
    bool has_w = (w != nullptr);
    for (uint i = tid.x; i < d; i += tpg.x) {
        float v = x[r*d+i] * inv * (has_w ? w[i] : 1.0f);
        if (y != nullptr) y[r*d+i] = v;
        if (yb != nullptr) yb[r*d+i] = f32_to_bf16_rn(v);
    }
}

struct GemvP { uint n; uint k; };
struct GemvQ8P { uint n; uint k; };

struct MoeP { uint h; uint mi; uint topk; uint k; };


inline float dot_q4_elem(device const uchar* Q, device const ushort* SM, uint row, uint K, uint j, float x) {
    uint qb = Q[row * (K / 2) + j / 2];
    uint q = (qb >> ((j & 1u) * 4u)) & 15u;
    uint g = j / 64;
    ushort2 smh = ushort2(SM[row * (K / 64) * 2 + g * 2], SM[row * (K / 64) * 2 + g * 2 + 1]);
    float2 sm = (float2)as_type<half2>(smh);
    return ((float)q * sm.x + sm.y) * x;
}

inline float dot_q8_elem(device const uchar* W, device const ushort* S, uint row, uint K, uint j, float x) {
    uint u = W[row * K + j];
    float wv = (float)(int)u - 256.0f * (float)(u >> 7);
    float s = (float)as_type<half>(S[row * (K / 128) + j / 128]);
    return wv * s * x;
}

// MoE decode gate: one threadgroup per act row ((topk+1)*I rows).
// Row kk*I+j holds silu(G . x) * (U . x) for routed expert idx[kk] (q4),
// the last block (kk == topk) is the shared int8 expert.
kernel void moe_gate_up(device const uchar* GQ [[buffer(0)]],
                    device const ushort* GS [[buffer(1)]],
                    device const uchar* SGQ [[buffer(2)]],
                    device const ushort* SGS [[buffer(3)]],
                    device const float* X [[buffer(4)]],
                    device const int* IDX [[buffer(5)]],
                    device float* ACT [[buffer(6)]],
                    constant MoeP& p [[buffer(7)]],
                    uint3 tpg [[threads_per_threadgroup]],
                    uint3 tid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]]) {
    uint slot = gid.x;
    uint mi = p.mi, k = p.k;
    if (slot >= (p.topk + 1) * mi) return;
    uint kk = slot / mi, j = slot % mi;
    float g = 0.0f, u = 0.0f;
    if (kk < p.topk) {
        uint e = (uint)IDX[kk];
        for (uint t = tid.x; t < k; t += tpg.x) {
            g += dot_q4_elem(GQ, GS, e * 2 * mi + j, k, t, X[t]);
            u += dot_q4_elem(GQ, GS, e * 2 * mi + mi + j, k, t, X[t]);
        }
    } else {
        for (uint t = tid.x; t < k; t += tpg.x) {
            g += dot_q8_elem(SGQ, SGS, j, k, t, X[t]);
            u += dot_q8_elem(SGQ, SGS, mi + j, k, t, X[t]);
        }
    }
    threadgroup float part[256];
    part[tid.x] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    g = part[0];
    part[tid.x] = u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    u = part[0];
    if (tid.x == 0) ACT[slot] = (g / (1.0f + exp(-g))) * u;
}

// MoE decode down: one threadgroup per output row y[n] =
// res[n] + sigmoid(sgate) * shared[n] + sum_k w[k] * D[n] . act[k].
kernel void moe_down(device const uchar* DQ [[buffer(0)]],
                 device const ushort* DS [[buffer(1)]],
                 device const uchar* SDQ [[buffer(2)]],
                 device const ushort* SDS [[buffer(3)]],
                 device const float* ACT [[buffer(4)]],
                 device const int* IDX [[buffer(5)]],
                 device const float* WTS [[buffer(6)]],
                 device const float* SGRT [[buffer(7)]],
                 device const float* RES [[buffer(8)]],
                 device float* Y [[buffer(9)]],
                 constant MoeP& p [[buffer(10)]],
                 uint3 tpg [[threads_per_threadgroup]],
                 uint3 tid [[thread_position_in_threadgroup]],
                 uint3 gid [[threadgroup_position_in_grid]]) {
    uint n = gid.x;
    uint h = p.h, mi = p.mi;
    if (n >= h) return;
    float acc = 0.0f;
    uint total = p.topk * mi;
    for (uint t = tid.x; t < total; t += tpg.x) {
        uint kk = t / mi, j = t % mi;
        uint e = (uint)IDX[kk];
        uint qb = DQ[(e * h + n) * (mi / 2) + j / 2];
        uint q = (qb >> ((j & 1u) * 4u)) & 15u;
        uint g = j / 64;
        ushort2 smh = ushort2(DS[(e * h + n) * (mi / 64) * 2 + g * 2], DS[(e * h + n) * (mi / 64) * 2 + g * 2 + 1]);
        float2 sm = (float2)as_type<half2>(smh);
        acc += WTS[kk] * ((float)q * sm.x + sm.y) * ACT[kk * mi + j];
    }
    threadgroup float part[256];
    part[tid.x] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid.x == 0) {
        float sh = 0.0f;
        for (uint j = 0; j < mi; ++j) sh += dot_q8_elem(SDQ, SDS, n, mi, j, ACT[p.topk * mi + j]);
        float sg = 1.0f / (1.0f + exp(-SGRT[0]));
        Y[n] = RES[n] + sg * sh + part[0];
    }
}
struct MoePFP { uint total; uint k; uint mi; uint h; };

inline float bf16_to_f32_msl(ushort b) {
    return as_type<float>((uint)b << 16);
}

inline ushort f32_to_bf16_msl(float v) {
    uint b = as_type<uint>(v);
    if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0) return 0x7fc0u;
    return (ushort)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}


struct AttnP { uint h; uint hkv; uint hd; uint nkeys; float scale; };

struct GdnP { uint n; uint nk; uint nv; uint ps; uint z_off; uint b_off; uint a_off; float eps; };

// Gated delta rule: one threadgroup per head, 128 threads (one per v).
// Fully serial per thread over k and t; no barriers needed (disjoint columns).
kernel void gdn(device const float* CONV [[buffer(0)]],
           device const float* PROJ [[buffer(1)]],
           device const float* ALOG [[buffer(2)]],
           device const float* DTB [[buffer(3)]],
           device const float* NW [[buffer(4)]],
           device float* STATE [[buffer(5)]],
           device float* OUT [[buffer(6)]],
           constant GdnP& p [[buffer(7)]],
           uint3 tpg [[threads_per_threadgroup]],
           uint3 tid [[thread_position_in_threadgroup]],
           uint3 gid [[threadgroup_position_in_grid]]) {
    uint hh = gid.x;
    if (hh >= p.nv) return;
    uint v = tid.x;
    if (v >= 128) return;
    const uint DK = 128;
    uint kh = hh / (p.nv / p.nk);
    uint cs = 2 * p.nk * DK + p.nv * DK;
    float qscale = rsqrt((float)DK);
    float a = -exp(ALOG[hh]);
    float dtb = DTB[hh];
    for (uint t = 0; t < p.n; ++t) {
        device const float* cr = CONV + t * cs;
        device const float* pr = PROJ + t * p.ps;
        float qsum = 0.0f, ksum = 0.0f;
        for (uint i = 0; i < DK; ++i) {
            float qv = cr[kh * DK + i];
            float kv = cr[p.nk * DK + kh * DK + i];
            qsum += qv * qv;
            ksum += kv * kv;
        }
        float qi = qscale / sqrt(qsum + 1e-6f);
        float ki = 1.0f / sqrt(ksum + 1e-6f);
        float vv = cr[2 * p.nk * DK + hh * DK + v];
        float gv = pr[p.a_off + hh] + dtb;
        float decay = exp(a * (gv > 20.0f ? gv : log(1.0f + exp(gv))));
        float beta = 1.0f / (1.0f + exp(-pr[p.b_off + hh]));
        float kvm = 0.0f;
        for (uint k = 0; k < DK; ++k) {
            uint idx = hh * DK * DK + k * DK + v;
            STATE[idx] *= decay;
            kvm += STATE[idx] * (cr[p.nk * DK + kh * DK + k] * ki);
        }
        float delta = (vv - kvm) * beta;
        float oacc = 0.0f;
        for (uint k = 0; k < DK; ++k) {
            uint idx = hh * DK * DK + k * DK + v;
            float ks = cr[p.nk * DK + kh * DK + k] * ki;
            float qs = cr[kh * DK + k] * qi;
            STATE[idx] += ks * delta;
            oacc += STATE[idx] * qs;
        }
        threadgroup float tacc[128];
        tacc[v] = oacc * oacc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // rms over v (parallel halves), then per-v output.
        threadgroup float invm[1];
        if (v < 64) tacc[v] += tacc[v + 64];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (v < 32) tacc[v] += tacc[v + 32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // NOTE: 32-way tail reduced serially by thread 0 (short).
        if (v == 0) {
            float ss = tacc[0];
            for (uint i = 1; i < 32; ++i) ss += tacc[i];
            invm[0] = rsqrt(ss / (float)DK + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float zr = pr[p.z_off + hh * DK + v];
        OUT[t * p.nv * DK + hh * DK + v] = oacc * invm[0] * NW[v] * (zr / (1.0f + exp(-zr)));
    }
}

// Single-row decode attention: one threadgroup per head. Phases: parallel
// max-score, serial l (thread 0), parallel outputs (one d per thread).
// Scores are recomputed per phase (no length cap) rather than staged.
kernel void attn_decode(device const float* Q [[buffer(0)]],
                    device const ushort* KC [[buffer(1)]],
                    device const ushort* VC [[buffer(2)]],
                    device const float* GATE [[buffer(3)]],
                    device float* OUT [[buffer(4)]],
                    constant AttnP& p [[buffer(5)]],
                    uint3 tpg [[threads_per_threadgroup]],
                    uint3 tid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]]) {
    uint hh = gid.x;
    if (hh >= p.h) return;
    uint hkv = p.hkv, hd = p.hd, nkeys = p.nkeys;
    uint kh = hh / (p.h / hkv);
    float lmax = -INFINITY;
    for (uint j = tid.x; j < nkeys; j += tpg.x) {
        float s = 0.0f;
        for (uint d = 0; d < hd; ++d)
            s += Q[hh * hd + d] * bf16_to_f32_msl(KC[(j * hkv + kh) * hd + d]);
        lmax = max(lmax, s * p.scale);
    }
    threadgroup float part[256];
    part[tid.x] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] = max(part[tid.x], part[tid.x + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = part[0];
    threadgroup float lmem[1];
    if (tid.x == 0) {
        float l = 0.0f;
        for (uint j = 0; j < nkeys; ++j) {
            float s = 0.0f;
            for (uint d = 0; d < hd; ++d)
                s += Q[hh * hd + d] * bf16_to_f32_msl(KC[(j * hkv + kh) * hd + d]);
            l += exp(s * p.scale - m);
        }
        lmem[0] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float l = lmem[0];
    for (uint d = tid.x; d < hd; d += tpg.x) {
        float acc = 0.0f;
        for (uint j = 0; j < nkeys; ++j) {
            float s = 0.0f;
            for (uint dd = 0; dd < hd; ++dd)
                s += Q[hh * hd + dd] * bf16_to_f32_msl(KC[(j * hkv + kh) * hd + dd]);
            acc += exp(s * p.scale - m) * bf16_to_f32_msl(VC[(j * hkv + kh) * hd + d]);
        }
        float gv = GATE[hh * hd + d];
        OUT[hh * hd + d] = acc / l * (1.0f / (1.0f + exp(-gv)));
    }
}


// MoE prefill gate: one threadgroup per slot; 256 threads x 2 cols each.
// act[s*mi+j] = bf16(silu(G_e . x) * (U_e . x)), e = slot_expert[s].
kernel void moe_pf_gate_up(device const uchar* GQ [[buffer(0)]],
                       device const ushort* GS [[buffer(1)]],
                       device const ushort* XP [[buffer(2)]],
                       device const int* SE [[buffer(3)]],
                       device ushort* ACT [[buffer(4)]],
                       constant MoePFP& p [[buffer(5)]],
                       uint3 tpg [[threads_per_threadgroup]],
                       uint3 tid [[thread_position_in_threadgroup]],
                       uint3 gid [[threadgroup_position_in_grid]]) {
    uint s = gid.x;
    if (s >= p.total) return;
    uint mi = p.mi, k = p.k;
    uint e = (uint)SE[s];
    for (uint jj = 0; jj < 2; ++jj) {
        uint j = tid.x * 2 + jj;
        if (j >= mi) return;
        float g = 0.0f, u = 0.0f;
        for (uint t = 0; t < k; ++t) {
            float x = bf16_to_f32_msl(XP[s * k + t]);
            g += dot_q4_elem(GQ, GS, e * 2 * mi + j, k, t, x);
            u += dot_q4_elem(GQ, GS, e * 2 * mi + mi + j, k, t, x);
        }
        ACT[s * mi + j] = f32_to_bf16_msl((g / (1.0f + exp(-g))) * u);
    }
}

// MoE prefill down: one threadgroup per slot; 256 threads x 8 rows each.
// y[s*h+n] = D_e[n] . act[s].
kernel void moe_pf_down(device const uchar* DQ [[buffer(0)]],
                    device const ushort* DS [[buffer(1)]],
                    device const ushort* ACT [[buffer(2)]],
                    device const int* SE [[buffer(3)]],
                    device float* Y [[buffer(4)]],
                    constant MoePFP& p [[buffer(5)]],
                    uint3 tpg [[threads_per_threadgroup]],
                    uint3 tid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]]) {
    uint s = gid.x;
    if (s >= p.total) return;
    uint mi = p.mi, h = p.h;
    uint e = (uint)SE[s];
    for (uint nn = 0; nn < 8; ++nn) {
        uint n = tid.x * 8 + nn;
        if (n >= h) return;
        float acc = 0.0f;
        for (uint j = 0; j < mi; ++j) {
            float a = bf16_to_f32_msl(ACT[s * mi + j]);
            acc += dot_q4_elem(DQ, DS, e * h + n, mi, j, a);
        }
        Y[s * h + n] = acc;
    }
}
kernel void gemv_q8(device const uchar* W [[buffer(0)]],
                    device const ushort* S [[buffer(1)]],
                    device const float* x [[buffer(2)]],
                    device const float* res [[buffer(3)]],
                    device float* y [[buffer(4)]],
                    constant GemvQ8P& p [[buffer(5)]],
                    uint3 tpg [[threads_per_threadgroup]],
                    uint3 tid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.x;
    if (r >= p.n) return;
    uint k = p.k;
    uint ng = k / 128;
    float acc = 0.0f;
    for (uint j = tid.x; j < k; j += tpg.x) {
        // W is int8; read as unsigned and fix the sign explicitly so this does
        // not depend on the implementation-defined signedness of `char`.
        uint u = W[r * k + j];
        float wv = (float)(int)u - 256.0f * (float)(u >> 7);
        float s = (float)as_type<half>(S[r * ng + j / 128]);
        acc += wv * s * x[j];
    }
    threadgroup float part[256];
    part[tid.x] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid.x == 0) y[r] = part[0] + (res != nullptr ? res[r] : 0.0f);
}
kernel void gemv_bf16(device const ushort* W [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant GemvP& p [[buffer(3)]],
                      uint3 tpg [[threads_per_threadgroup]],
                      uint3 tid [[thread_position_in_threadgroup]],
                      uint3 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.x;
    if (r >= p.n) return;
    float acc = 0.0f;
    for (uint j = tid.x; j < p.k; j += tpg.x) acc += bf16_to_f32(W[r*p.k+j]) * x[j];
    threadgroup float part[256];
    part[tid.x] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg.x >> 1; s > 0; s >>= 1) {
        if (tid.x < s) part[tid.x] += part[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid.x == 0) y[r] = part[0];
}

struct DqP { uint n; uint k; };
kernel void dequant_q8(device const char* W [[buffer(0)]],
                       device const ushort* S [[buffer(1)]],
                       device ushort* o [[buffer(2)]],
                       constant DqP& p [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
    uint total = p.n * p.k;
    if (tid >= total) return;
    uint k = tid % p.k;
    float s = (float)as_type<half>(S[(tid / p.k) * (p.k / 128) + k / 128]);
    o[tid] = f32_to_bf16_rn((float)W[tid] * s);
}

kernel void f32_to_bf16(device const float* x [[buffer(0)]],
                        device ushort* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid < n) y[tid] = f32_to_bf16_rn(x[tid]);
}

kernel void swiglu_rows(device const float* h [[buffer(0)]],
                        device ushort* y [[buffer(1)]],
                        constant uint2& dims [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    uint I = dims.y;
    uint total = dims.x * I;
    if (tid >= total) return;
    uint j = tid % I;
    uint r = tid / I;
    float g = h[r * 2 * I + j];
    float u = h[r * 2 * I + I + j];
    y[tid] = f32_to_bf16_rn(g / (1.0f + exp(-g)) * u);
}
"#;

#[repr(C)]
struct NormP {
    rows: u32,
    d: u32,
    eps: f32,
}

#[repr(C)]
struct GemvP {
    n: u32,
    k: u32,
}

#[repr(C)]
struct GemvQ8P {
    n: u32,
    k: u32,
}

#[repr(C)]
struct MoeP {
    h: u32,
    mi: u32,
    topk: u32,
    k: u32,
}

#[repr(C)]
struct MoePFP {
    total: u32,
    k: u32,
    mi: u32,
    h: u32,
}

#[repr(C)]
struct AttnP {
    h: u32,
    hkv: u32,
    hd: u32,
    nkeys: u32,
    scale: f32,
}

#[repr(C)]
struct GdnP {
    n: u32,
    nk: u32,
    nv: u32,
    ps: u32,
    z_off: u32,
    b_off: u32,
    a_off: u32,
    eps: f32,
}

#[repr(C)]
struct DqP {
    n: u32,
    k: u32,
}

fn pb<T>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Dispatch an MSL kernel. `bufs` are (pointer, extra_offset); null pointers become
/// nil buffers. `params` go to the next buffer index as inline constants.
fn pending_enc(q: usize) -> bool {
    PENDING.with(|p| {
        if p.borrow().iter().any(|(qq, _, _, _)| *qq == q) {
            return true;
        }
        let c = ctx();
        let queue = &c.queue.0;
        let cb = queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder().to_owned();
        p.borrow_mut().push((q, cb, enc, ""));
        true
    })
}

/// End, commit and wait for every pending command buffer (all queues).
/// Call before any CPU read of GPU-produced data and at graph end.
/// End, commit and wait for every pending command buffer (all queues).
/// Call before any CPU read of GPU-produced data and at graph end.
/// Panics with the Metal error on command-buffer failure: silent GPU errors
/// used to poison downstream CPU reads (NaN -> topk panic far away).
/// End, commit and wait for every pending command buffer (all queues).
/// Call before any CPU read of GPU-produced data and at graph end.
/// Returns 0 on success, 1 if any command buffer failed (silent GPU errors
/// used to poison downstream CPU reads); failures are logged with the status.
/// (metal 0.33 exposes no NSError details on CommandBuffer, only the status.)
fn flush_gpu() -> i32 {
    use metal::MTLCommandBufferStatus;
    let mut rc = 0;
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        for (_, cb, enc, _) in p.drain(..) {
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if cb.status() == MTLCommandBufferStatus::Error {
                eprintln!("metal: command buffer failed (status=Error)");
                rc = 1;
            }
        }
    });
    rc
}

/// flush_gpu for launch() closures: a failed buffer must abort the op, and
/// launch converts the panic into return code 1 (established error path).
fn flush_ok() {
    if flush_gpu() != 0 {
        panic!("metal: command buffer failed");
    }
}

fn dispatch(name: &'static str, bufs: &[Option<(MBuf, u64)>], params: &[u8], grid: (u64, u64, u64), tg: (u64, u64, u64), threadgroups: bool) {
    let p = pipe(name);
    pending_enc(0);
    PENDING.with(|pp| {
        let ppp = pp.borrow();
        let (_, _, enc, _) = ppp.iter().find(|(qq, _, _, _)| *qq == 0).expect("dispatch: no pending buffer");
        enc.set_compute_pipeline_state(&p.0);
        for (i, b) in bufs.iter().enumerate() {
            match b {
                Some((buf, off)) => enc.set_buffer(i as u64, Some(&buf.0), *off),
                None => enc.set_buffer(i as u64, None, 0),
            }
        }
        enc.set_bytes(bufs.len() as u64, params.len() as u64, params.as_ptr() as *const c_void);
        let g = metal::MTLSize { width: grid.0, height: grid.1, depth: grid.2 };
        let t = metal::MTLSize { width: tg.0, height: tg.1, depth: tg.2 };
        if threadgroups {
            enc.dispatch_thread_groups(g, t);
        } else {
            enc.dispatch_threads(g, t);
        }
    });
}

fn opt_buf(ptr: *const c_void, extra: u64) -> Option<(MBuf, u64)> {
    if ptr.is_null() {
        return None;
    }
    let (b, off) = require(ptr);
    Some((b, off + extra))
}

fn ceil256(n: u64) -> u64 {
    (n + 255) / 256 * 256
}

// Scratch buffers are thread-local: parallel callers (tests; future engine
// threads) must never clobber each other's convert/norm staging. A single
// thread reuses its buffer monotonically (execution is synchronous, so there
// is no aliasing hazard within a thread).
/// TLS scratch slot: unregisters its buffer from BUFS on thread exit
/// (otherwise the entry outlives the thread; shows up mostly in tests —
/// pool workers persist, but test threads come and go). `static`s are never
/// dropped, so BUFS is always alive here; ignore a poisoned lock.
struct TrackedScratch(Option<MBuf>);
impl Drop for TrackedScratch {
    fn drop(&mut self) {
        if let Some(b) = self.0.take() {
            if let Ok(mut bufs) = BUFS.lock() {
                bufs.remove(&(b.0.contents() as usize));
            }
        }
    }
}
std::thread_local! {
    static SCRATCH: std::cell::RefCell<TrackedScratch> = const { std::cell::RefCell::new(TrackedScratch(None)) };
    static SCRATCH_W: std::cell::RefCell<TrackedScratch> = const { std::cell::RefCell::new(TrackedScratch(None)) };
}

fn scratch_f32(n_elems: usize) -> (MBuf, u64) {
    scratch_here(&SCRATCH, n_elems)
}

fn scratch_f32_w(n_elems: usize) -> (MBuf, u64) {
    scratch_here(&SCRATCH_W, n_elems)
}

fn scratch_here(slot: &'static std::thread::LocalKey<std::cell::RefCell<TrackedScratch>>, n_elems: usize) -> (MBuf, u64) {
    slot.with(|s| {
        let mut s = s.borrow_mut();
        let need = (n_elems.max(1) as u64) * 4;
        let ok = s.0.as_ref().map(|b| b.0.length() >= need).unwrap_or(false);
        if !ok {
            // Unregister the replaced buffer: pending command buffers hold their
            // own refs via set_buffer, so removal is safe and stops BUFS growth.
            if let Some(old) = s.0.take() {
                BUFS.lock().unwrap().remove(&(old.0.contents() as usize));
            }
            let b = ctx().device.0.new_buffer(need, metal::MTLResourceOptions::StorageModeShared);
            track(&b);
            s.0 = Some(MBuf(b));
        }
        (s.0.as_ref().unwrap().clone(), 0)
    })
}

// ---------------------------------------------------------------------------
// MSL-backed ops

/// MSL: rmsnorm.
///
/// y[r] = x[r] * inv * w (f32 and/or bf16). `w` is required by every caller.
pub unsafe fn dr_rmsnorm(x: *const f32, w: *const f32, y: *mut f32, yb: *mut Half, rows: i32, d: i32, eps: f32, st: Stream) -> i32 {
    launch(st, move || {
        let xb = require(x as *const c_void);
        let wb = require(w as *const c_void);
        let prm = NormP { rows: rows as u32, d: d as u32, eps };
        dispatch(
            "rmsnorm",
            &[Some(xb), Some(wb), opt_buf(y as *const c_void, 0), opt_buf(yb as *const c_void, 0)],
            pb(&prm),
            (rows as u64, 1, 1),
            (256, 1, 1),
            true,
        );
    })
}

/// MSL: y[n] = W[n] . norm(x) (+ fused RMSNorm when `nw` is set).
pub unsafe fn dr_gemv_bf16(w: *const Half, x: *const f32, y: *mut f32, n: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    launch(st, move || {
        let wb = require(w as *const c_void);
        let yb = require(y as *const c_void);
        let xv: (MBuf, u64) = if nw.is_null() {
            require(x as *const c_void)
        } else {
            let (sb, soff) = scratch_f32(k as usize);
            let nwb = require(nw as *const c_void);
            let xb = require(x as *const c_void);
            let prm = NormP { rows: 1, d: k as u32, eps };
            dispatch("rmsnorm", &[Some(xb), Some(nwb), Some((sb.clone(), soff)), None], pb(&prm), (1, 1, 1), (256, 1, 1), true);
            (sb, soff)
        };
        let prm = GemvP { n: n as u32, k: k as u32 };
        dispatch("gemv_bf16", &[Some(wb), Some(xv), Some(yb)], pb(&prm), (n as u64, 1, 1), (256, 1, 1), true);
    })
}

/// MSL: dequantise int8 group-128 weights to bf16.
pub unsafe fn dr_dequant_q8(w: *const i8, s: *const Half, out: *mut Half, n: i32, k: i32, st: Stream) -> i32 {
    launch(st, move || {
        let prm = DqP { n: n as u32, k: k as u32 };
        let total = n as u64 * k as u64;
        dispatch(
            "dequant_q8",
            &[opt_buf(w as *const c_void, 0), opt_buf(s as *const c_void, 0), opt_buf(out as *const c_void, 0)],
            pb(&prm),
            (ceil256(total), 1, 1),
            (256, 1, 1),
            false,
        );
    })
}

/// MSL: f32 -> bf16 (round-to-nearest-even).
pub unsafe fn dr_f32_to_bf16(x: *const f32, y: *mut Half, n: usize, st: Stream) -> i32 {
    launch(st, move || {
        let nu = n as u32;
        dispatch(
            "f32_to_bf16",
            &[opt_buf(x as *const c_void, 0), opt_buf(y as *const c_void, 0)],
            pb(&nu),
            (ceil256(n as u64), 1, 1),
            (256, 1, 1),
            false,
        );
    })
}

/// MSL: y[r, j] = silu(h[r, j]) * h[r, I + j] in bf16.
pub unsafe fn dr_swiglu_rows(h: *const f32, y: *mut Half, rows: i32, i: i32, st: Stream) -> i32 {
    launch(st, move || {
        let dims = [rows as u32, i as u32];
        let total = rows as u64 * i as u64;
        dispatch(
            "swiglu_rows",
            &[opt_buf(h as *const c_void, 0), opt_buf(y as *const c_void, 0)],
            pb(&dims),
            (ceil256(total), 1, 1),
            (256, 1, 1),
            false,
        );
    })
}

// ---------------------------------------------------------------------------
// runtime: memory, streams, graphs, misc (all CPU-side; synchronous)

static METAL_UNAVAILABLE: &[u8] = b"metal backend error (op not yet ported or dispatch failed; see docs/MLX-PORT.md)\0";

pub unsafe fn dr_error_string(_e: i32) -> *const c_char {
    METAL_UNAVAILABLE.as_ptr() as *const c_char
}

pub unsafe fn dr_last_error() -> i32 {
    0
}

pub unsafe fn dr_malloc(p: *mut *mut c_void, n: usize) -> i32 {
    let n = n.max(16);
    let buf = ctx().device.0.new_buffer(n as u64, metal::MTLResourceOptions::StorageModeShared);
    if buf.contents().is_null() {
        return 1;
    }
    *p = track(&buf);
    0
}

pub unsafe fn dr_free(p: *mut c_void) -> i32 {
    if p.is_null() {
        return 0;
    }
    BUFS.lock().unwrap().remove(&(p as usize));
    // Drop f32-cache entries keyed by the freed buffer: the allocator may
    // hand the same address to different weights of the same shape, and a
    // stale entry would feed GEMM the old matrix. Also unregister the cached
    // twin buffers so BUFS never points at dropped memory.
    // (Scoped locks: never hold F32CACHE and BUFS at once — gemm takes them
    // in the opposite order.)
    let twins: Vec<usize> = {
        let mut fc = f32cache().lock().unwrap();
        let stale: Vec<_> = fc.map.iter().filter(|(k, _)| k.0 == p as usize).map(|(k, b)| (*k, b.0.contents() as usize)).collect();
        let mut twins = Vec::with_capacity(stale.len());
        for (k, twin) in stale {
            if let Some(b) = fc.map.remove(&k) {
                fc.bytes -= (b.0.length() as usize).min(fc.bytes);
            }
            twins.push(twin);
        }
        twins
    };
    if !twins.is_empty() {
        let mut bufs = BUFS.lock().unwrap();
        for t in twins {
            bufs.remove(&t);
        }
    }
    0
}

pub unsafe fn dr_host_alloc(p: *mut *mut c_void, n: usize) -> i32 {
    let layout = match std::alloc::Layout::from_size_align(n.max(1), 16) {
        Ok(l) => l,
        Err(_) => return 1,
    };
    let ptr = std::alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        return 1;
    }
    HOST.lock().unwrap().insert(ptr as usize, layout);
    *p = ptr as *mut c_void;
    0
}

pub unsafe fn dr_host_free(p: *mut c_void) -> i32 {
    if !p.is_null() {
        if let Some(layout) = HOST.lock().unwrap().remove(&(p as usize)) {
            std::alloc::dealloc(p as *mut u8, layout);
        }
    }
    0
}

unsafe extern "C" {
    fn posix_madvise(addr: *const c_void, len: usize, advice: i32) -> i32;
    fn mlock(addr: *const c_void, len: usize) -> i32;
}

/// Drop clean file pages backing a just-uploaded weight tensor (macOS page
/// size is 16 KiB on Apple Silicon). Safe: pages are clean (re-faultable)
/// and never read again after upload.
fn release_file_pages(p: *const u8, n: usize) {
    const SZ: usize = 16384;
    const DONTNEED: i32 = 4;
    let a = p as usize & !(SZ - 1);
    let b = (p as usize + n + SZ - 1) & !(SZ - 1);
    unsafe {
        posix_madvise(a as *const c_void, b - a, DONTNEED);
    }
}

fn raw_copy(dst: *mut c_void, src: *const c_void, n: usize) {
    if n == 0 {
        return;
    }
    unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, n) };
}

pub unsafe fn dr_h2d(d: *mut c_void, h: *const c_void, n: usize, st: Stream) -> i32 {
    launch(st, move || raw_copy(d, h, n));
    // Weight uploads are read-once file pages (safetensors mmaps). Keeping
    // 21 GB of them in page cache next to 20 GB of Metal buffers forces the
    // OS into swap thrash on a 32 GB machine (observed: 150+ GB swapped,
    // every cold weight touch stalling 100-800 ms). Release them right away.
    // Runtime staging uploads are KBs and never hit this path (threshold).
    if n > 1 << 20 {
        release_file_pages(h as *const u8, n);
        // Pin the just-written destination pages: 20 GB of weights must
        // never be swapped (a single swapped-out layer stalls prefill by
        // ~1 s on first touch). Cheap here: pages are already resident.
        unsafe {
            mlock(d, n);
        }
    }
    0
}

pub unsafe fn dr_d2h(h: *mut c_void, d: *const c_void, n: usize, st: Stream) -> i32 {
    // CPU reads GPU-produced data: drain the pending batch first.
    launch(st, move || {
        flush_ok();
        raw_copy(h, d, n)
    })
}

pub unsafe fn dr_d2d(d: *mut c_void, s: *const c_void, n: usize, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        raw_copy(d, s, n)
    })
}

pub unsafe fn dr_memset(d: *mut c_void, v: i32, n: usize, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        if n > 0 {
            std::slice::from_raw_parts_mut(d as *mut u8, n).fill(v as u8);
        }
    })
}

pub unsafe fn dr_stream_create(s: *mut Stream) -> i32 {
    let id = NEXT_STREAM.fetch_add(1, Ordering::Relaxed);
    QUEUES.lock().unwrap().insert(id, MQueue(ctx().device.0.new_command_queue()));
    *s = id as Stream;
    0
}

pub unsafe fn dr_stream_sync(_s: Stream) -> i32 {
    // H1: with batched dispatches, sync must drain the pending queue.
    // Runs outside launch(), so the code is returned, not panicked.
    flush_gpu()
}

pub unsafe fn dr_mem_info(free_b: *mut usize, total_b: *mut usize) -> i32 {
    let total = ctx().device.0.recommended_max_working_set_size();
    let used = ctx().device.0.current_allocated_size();
    *total_b = total as usize;
    *free_b = total.saturating_sub(used) as usize;
    0
}

pub unsafe fn dr_graph_begin(st: Stream) -> i32 {
    CAPTURE.with(|c| {
        *c.borrow_mut() = Some(Capture { stream: st as usize, thread: std::thread::current().id(), ops: Vec::new() });
    });
    0
}

pub unsafe fn dr_graph_end(st: Stream, exec: *mut GraphExec) -> i32 {
    CAPTURE.with(|c| {
        let mut c = c.borrow_mut();
        match c.take() {
            Some(cap) if cap.stream == st as usize && cap.thread == std::thread::current().id() => {
                let g = Box::new(GraphData { ops: cap.ops });
                *exec = Box::into_raw(g) as GraphExec;
                0
            }
            _ => 1,
        }
    })
}

pub unsafe fn dr_graph_launch(exec: GraphExec, _st: Stream) -> i32 {
    let g = &*(exec as *const GraphData);
    for op in &g.ops {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op())).is_err() {
            return 1;
        }
    }
    flush_gpu();
    0
}

pub unsafe fn dr_graph_destroy(exec: GraphExec) -> i32 {
    if !exec.is_null() {
        drop(Box::from_raw(exec as *mut GraphData));
    }
    0
}

pub unsafe fn dr_cublas_create(h: *mut *mut c_void, _st: Stream) -> i32 {
    // No handle needed: GEMMs run on the CPU in v1 (see dr_gemm).
    *h = std::ptr::null_mut();
    0
}

/// CPU: C[m, n] = A[m, k] . W[n, k]^T + beta * C (bf16 inputs, f32 output).
///
/// Prefill only; decode uses the GEMV kernels. Slow on big shapes (a blocked or
/// MPS matmul is future work) but exact and fine for validation.
pub unsafe fn dr_gemm(_h: *mut c_void, a: *const Half, w: *const Half, c: *mut f32, m: i32, n: i32, k: i32, beta: f32) -> i32 {
    if m <= 0 || n <= 0 || k <= 0 {
        return 0;
    }
    let (m, n, k) = (m as usize, n as usize, k as usize);
    // bf16 inputs -> f32 scratch (bandwidth-bound, threaded), then one AMX SGEMM.
    // C[m, n] = A[m, k] . W[n, k]^T + beta * C (row-major).
    // H1: CPU reads GPU-produced A/W/C here — drain the pending batch first.
    // (Without this, dequant/gdn/etc. queued just before are silently skipped.)
    flush_gpu();
    let (abuf, _) = scratch_f32(m * k);
    let (wbuf, _) = scratch_f32_w(n * k);
    let (a, w, c) = (SendPtr(a as *mut Half), SendPtr(w as *mut Half), SendPtr(c));
    let (af, wf) = (SendPtr(abuf.0.contents() as *mut f32), SendPtr(wbuf.0.contents() as *mut f32));
    par_for!(m * k, |lo, hi| {
        for i in lo..hi {
            unsafe { af.write(i, bf16_to_f32(a.read(i))) };
        }
    });
    par_for!(n * k, |lo, hi| {
        for i in lo..hi {
            unsafe { wf.write(i, bf16_to_f32(w.read(i))) };
        }
    });
    unsafe {
        cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                af.ptr() as *const f32,
                k as i32,
                wf.ptr() as *const f32,
                k as i32,
                beta,
                c.ptr(),
                n as i32,
            )
        };
    0
}

/// CPU: C[m, n] = A[m, k] . W[n, k]^T + beta * C (f16 inputs, f32 output).
pub unsafe fn dr_gemm_f16(_h: *mut c_void, a: *const Half, w: *const Half, c: *mut f32, m: i32, n: i32, k: i32, beta: f32) -> i32 {
    if m <= 0 || n <= 0 || k <= 0 {
        return 0;
    }
    let (m, n, k) = (m as usize, n as usize, k as usize);
    flush_gpu();
    let (abuf, _) = scratch_f32(m * k);
    let (wbuf, _) = scratch_f32_w(n * k);
    let (a, w, c) = (SendPtr(a as *mut Half), SendPtr(w as *mut Half), SendPtr(c));
    let (af, wf) = (SendPtr(abuf.0.contents() as *mut f32), SendPtr(wbuf.0.contents() as *mut f32));
    par_for!(m * k, |lo, hi| {
        for i in lo..hi {
            unsafe { af.write(i, f16_to_f32(a.read(i))) };
        }
    });
    // Codec weights are immutable: cache their f32 image keyed by (ptr, n, k)
    // so repeated chunks skip the bandwidth-bound convert. Bounded (512 MB);
    // bypass when full. (Backbone dr_gemm is NOT cached: its W is volatile
    // dequant scratch, and its f32 image would cost GBs.)
    let wf: SendPtr<f32> = {
        let key = (w.ptr() as usize, n, k);
        let hit = f32cache().lock().unwrap().map.get(&key).cloned();
        if let Some(b) = hit {
            SendPtr(b.0.contents() as *mut f32)
        } else {
            par_for!(n * k, |lo, hi| {
                for i in lo..hi {
                    unsafe { wf.write(i, f16_to_f32(w.read(i))) };
                }
            });
            let mut fc = f32cache().lock().unwrap();
            if fc.bytes + n * k * 4 <= F32CACHE_CAP {
                let need = (n * k * 4) as u64;
                let nb = ctx().device.0.new_buffer(need, metal::MTLResourceOptions::StorageModeShared);
                track(&nb);
                let cb = MBuf(nb);
                unsafe { std::ptr::copy_nonoverlapping(wf.ptr() as *const u8, cb.0.contents() as *mut u8, n * k * 4) };
                fc.bytes += n * k * 4;
                fc.map.insert(key, cb.clone());
                SendPtr(cb.0.contents() as *mut f32)
            } else {
                wf
            }
        }
    };
    unsafe {
        cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                af.ptr() as *const f32,
                k as i32,
                wf.ptr() as *const f32,
                k as i32,
                beta,
                c.ptr(),
                n as i32,
            )
        };
    0
}

pub unsafe fn dr_attn_part_floats() -> i32 {
    // Must match the CUDA value: callers size the `part` buffer with it.
    16 * 258
}

pub unsafe fn dr_sample_init() -> i32 {
    0
}

// ---------------------------------------------------------------------------
// CPU reference: MoE routing helpers

/// CPU: softmax over E logits, top-k (descending, lower-index tie-break),
/// renormalised weights; shared-expert gate logit at column E.
pub unsafe fn dr_moe_topk(logits: *const f32, rows: i32, e: i32, topk: i32, stride: i32, idx: *mut i32, wts: *mut f32, sgate: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, e, topk, stride) = (rows as usize, e as usize, topk as usize, stride as usize);
        for r in 0..rows {
            let l = &f32s(logits, (r + 1) * stride)[r * stride..];
            let m = l[..e].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
            let ex: Vec<f32> = order.iter().map(|&i| (l[i] - m).exp()).collect();
            let z: f32 = ex.iter().sum();
            let zt: f32 = ex[..topk].iter().sum();
            for k in 0..topk {
                *idx.add(r * topk + k) = order[k] as i32;
                *wts.add(r * topk + k) = ex[k] / zt;
            }
            let _ = z;
            *sgate.add(r) = l[e];
        }
    })
}

/// CPU: counts[e] += #{i : idx[i] == e}.
pub unsafe fn dr_moe_count(idx: *const i32, n: i32, counts: *mut i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        for i in 0..n as usize {
            let e = *idx.add(i) as usize;
            *counts.add(e) += 1;
        }
    })
}

/// CPU: scatter bf16 rows of x into expert-sorted order.
pub unsafe fn dr_moe_permute(idx: *const i32, rows: i32, topk: i32, offsets: *const i32, fill: *mut i32, slot_of: *mut i32, x: *const Half, xp: *mut Half, h: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, topk, h) = (rows as usize, topk as usize, h as usize);
        for i in 0..rows * topk {
            let e = *idx.add(i) as usize;
            let slot = (*offsets.add(e) + *fill.add(e)) as usize;
            *fill.add(e) += 1;
            *slot_of.add(i) = slot as i32;
            let r = i / topk;
            std::ptr::copy_nonoverlapping(x.add(r * h), xp.add(slot * h), h);
        }
    })
}

/// CPU: x[r] += sigmoid(sgate[r]) * shared[r] + sum_k w[r,k] * Y[slot_of[r,k]].
pub unsafe fn dr_moe_combine(x: *mut f32, y: *const f32, slot_of: *const i32, wts: *const f32, sgate: *const f32, shared: *const f32, rows: i32, topk: i32, h: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, topk, h) = (rows as usize, topk as usize, h as usize);
        for r in 0..rows {
            let sg = sigmoid(*sgate.add(r));
            for cc in 0..h {
                let mut acc = sg * *shared.add(r * h + cc);
                for k in 0..topk {
                    let slot = *slot_of.add(r * topk + k) as usize;
                    acc += *wts.add(r * topk + k) * *y.add(slot * h + cc);
                }
                *x.add(r * h + cc) += acc;
            }
        }
    })
}

// ---------------------------------------------------------------------------
// CPU reference: sampling. Distribution matches kernels/drama.cu (top-k / top-p with
// the sglang-omni rule); the Philox stream below is distribution-equivalent to
// curand but NOT bit-identical, so sampled tokens may differ from CUDA runs.
// (Validation compares logits, never samples.)

#[repr(C)]
#[derive(Clone, Copy)]
struct SampleParams {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    pad: i32,
    seed: u64,
}

fn philox4x32(seed: u64, stream: u64, step: u64) -> [u32; 4] {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;
    let mut k0 = seed as u32;
    let mut k1 = (seed >> 32) as u32;
    let mut c = [step as u32, (step >> 32) as u32, stream as u32, 0u32];
    for _ in 0..10 {
        let p0 = c[0] as u64 * M0 as u64;
        let p1 = c[2] as u64 * M1 as u64;
        c = [(p1 >> 32) as u32 ^ c[1] ^ k0, p1 as u32, (p0 >> 32) as u32 ^ c[3] ^ k1, p0 as u32];
        k0 = k0.wrapping_add(W0);
        k1 = k1.wrapping_add(W1);
    }
    c
}

/// Uniform in (0, 1], like curand_uniform.
fn philox_uniform(seed: u64, stream: u64, step: u64) -> f32 {
    let x = philox4x32(seed, stream, step)[0];
    1.0 - x as f32 / 4294967296.0
}

/// Sample a logit position: full descending sort (index tie-break), sglang-omni
/// top-k/top-p rule over the full distribution.
fn sample_logits(logits: &[f32], temp: f32, top_k: i32, top_p: f32, u: f32) -> usize {
    let n = logits.len();
    let inv_t = 1.0 / temp.max(1e-5);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    let k0 = logits[order[0]];
    let mut cum = 0.0f32;
    let mut cums = vec![0.0f32; n];
    for (i, &idx) in order.iter().enumerate() {
        cum += ((logits[idx] - k0) * inv_t).exp();
        cums[i] = cum;
    }
    let z = cum;
    let mut kept = cums.iter().filter(|&&c| c <= top_p * z).count();
    kept = kept.clamp(1, top_k.max(1) as usize);
    let target = u * cums[kept - 1];
    let rank = (0..kept).find(|&i| cums[i] >= target).unwrap_or(kept - 1);
    order[rank]
}

pub unsafe fn dr_sample_main(logits: *const f32, n: i32, prm: *const c_void, ras: *const i32, ras_len: i32, step: *const u64, eos: i32, sem0: i32, cb: i32, frame: *mut i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let p = *(prm as *const SampleParams);
        let l = f32s(logits, n as usize);
        let s = *step;
        let u1 = philox_uniform(p.seed, 0, s);
        let u2 = philox_uniform(p.seed, 1, s);
        let r1 = sample_logits(l, p.temperature, p.top_k, p.top_p, u1);
        let r2 = sample_logits(l, 1.0, 30, 0.95, u2);
        let t1 = if r1 == 0 { eos } else { sem0 + r1 as i32 - 1 };
        let t2 = if r2 == 0 { eos } else { sem0 + r2 as i32 - 1 };
        let mut repeat = false;
        for i in 0..ras_len as usize {
            repeat |= *ras.add(i) == t1;
        }
        let main = if repeat && r1 != 0 { t2 } else { t1 };
        *frame.add(0) = main;
        *frame.add(1) = (main - sem0).clamp(0, cb - 1);
    })
}

pub unsafe fn dr_sample_head(logits: *const f32, n: i32, prm: *const c_void, stream: i32, step: *const u64, frame: *mut i32, col: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let p = *(prm as *const SampleParams);
        let l = f32s(logits, n as usize);
        let u = philox_uniform(p.seed, stream as u64, *step);
        let r = sample_logits(l, p.temperature, p.top_k, p.top_p, u);
        *frame.add(col as usize) = r as i32;
    })
}

pub unsafe fn dr_frame_end(next: *const i32, in_frame: *mut i32, ras: *mut i32, ras_len: i32, pos: *mut i32, step: *mut u64, hist: *mut i32, hist_len: *mut i32, hist_cap: i32, width: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (ras_len, width) = (ras_len as usize, width as usize);
        for i in 0..width {
            *in_frame.add(i) = *next.add(i);
        }
        for i in 0..ras_len.saturating_sub(1) {
            *ras.add(i) = *ras.add(i + 1);
        }
        if ras_len > 0 {
            *ras.add(ras_len - 1) = *next.add(0);
        }
        *pos.add(0) += 1;
        *step.add(0) += 1;
        let h = *hist_len.add(0);
        if h < hist_cap {
            for i in 0..width {
                *hist.add(h as usize * width + i) = *next.add(i);
            }
            *hist_len.add(0) = h + 1;
        }
    })
}

// ---------------------------------------------------------------------------
// (every dr_* above is implemented; nothing routes to unported() anymore)

// ---------------------------------------------------------------------------
// CPU reference: quantised GEMV (decode)

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// RMS-normalise (with optional fused weight); plain copy when `nw` is None.
fn norm_fused(x: &[f32], nw: Option<&[f32]>, eps: f32) -> Vec<f32> {
    match nw {
        None => x.to_vec(),
        Some(w) => {
            let inv = rms_inv(x, eps);
            x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
        }
    }
}

/// int4 asymmetric row dot, group-64 (f16 scale + f16 min per group).
fn dot_q4_row(q: &[u8], sm: &[u16], k: usize, row: usize, x: &[f32]) -> f32 {
    let qb = &q[row * (k / 2)..(row + 1) * (k / 2)];
    let smb = &sm[row * (k / 64) * 2..];
    let mut acc = 0.0f32;
    for (j, &xv) in x.iter().enumerate().take(k) {
        let g = j / 64;
        let qq = (qb[j / 2] >> ((j & 1) * 4)) & 15;
        let w = qq as f32 * f16_to_f32(smb[g * 2]) + f16_to_f32(smb[g * 2 + 1]);
        acc += w * xv;
    }
    acc
}

/// bf16 row dot.
fn dot_bf16_row(w: &[u16], k: usize, row: usize, x: &[f32]) -> f32 {
    let wr = &w[row * k..(row + 1) * k];
    wr.iter().zip(x.iter()).map(|(a, b)| bf16_to_f32(*a) * b).sum()
}

/// CPU: y[n] = W[n] . norm(x) (+ res[n]); fused RMSNorm when `nw` is set.
/// MSL: y[n] = W[n] . norm(x) (+ res[n]); fused RMSNorm when `nw` is set.
pub unsafe fn dr_gemv_q8(w: *const i8, s: *const Half, x: *const f32, res: *const f32, y: *mut f32, n: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    launch(st, move || {
        let (n, k) = (n as usize, k as usize);
        let xv: (MBuf, u64) = if nw.is_null() {
            require(x as *const c_void)
        } else {
            let (sb, soff) = scratch_f32(k);
            let nwb = require(nw as *const c_void);
            let xb = require(x as *const c_void);
            let prm = NormP { rows: 1, d: k as u32, eps };
            dispatch("rmsnorm", &[Some(xb), Some(nwb), Some((sb.clone(), soff)), None], pb(&prm), (1, 1, 1), (256, 1, 1), true);
            (sb, soff)
        };
        let prm = GemvQ8P { n: n as u32, k: k as u32 };
        dispatch(
            "gemv_q8",
            &[opt_buf(w as *const c_void, 0), opt_buf(s as *const c_void, 0), Some(xv), opt_buf(res as *const c_void, 0), opt_buf(y as *const c_void, 0)],
            pb(&prm),
            (n as u64, 1, 1),
            (256, 1, 1),
            true,
        );
    })
}

/// CPU: y[j] = silu(W[j] . norm(x)) * (W[I + j] . norm(x)).
pub unsafe fn dr_gemv_q8_swiglu(w: *const i8, s: *const Half, x: *const f32, y: *mut f32, i: i32, k: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (i, k) = (i as usize, k as usize);
        let xn = norm_fused(f32s(x, k), if nw.is_null() { None } else { Some(f32s(nw, k)) }, eps);
        let (w, s, y) = (SendPtr(w as *mut i8), SendPtr(s as *mut Half), SendPtr(y));
        let xn: &[f32] = &xn;
        par_for!(i, |lo, hi| {
            for j in lo..hi {
                let g = unsafe { dot_q8_ptr(w, s, k, j, &xn) };
                let u = unsafe { dot_q8_ptr(w, s, k, i + j, &xn) };
                unsafe { y.write(j, silu(g) * u); }
            }
        });
    })
}

/// CPU: router logits (fused norm) + softmax/top-k; warms nothing, resets counter.
pub unsafe fn dr_router_topk(w: *const Half, x: *const f32, nw: *const f32, eps: f32, logits: *mut f32, e: i32, topk: i32, k: i32, idx: *mut i32, wts: *mut f32, sgate: *mut f32, counter: *mut u32, _gq: *const c_char, _gs: *const c_char, _dq: *const c_char, _ds: *const c_char, _gqb: usize, _gsb: usize, _dqb: usize, _dsb: usize, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (e, topk, k) = (e as usize, topk as usize, k as usize);
        let wv = std::slice::from_raw_parts(w, (e + 1) * k);
        let xn = norm_fused(f32s(x, k), if nw.is_null() { None } else { Some(f32s(nw, k)) }, eps);
        for row in 0..=e {
            *logits.add(row) = dot_bf16_row(wv, k, row, &xn);
        }
        let l = std::slice::from_raw_parts(logits, e + 1);
        let m = l[..e].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let ex: Vec<f32> = order.iter().map(|&i| (l[i] - m).exp()).collect();
        let zt: f32 = ex[..topk].iter().sum();
        for kk in 0..topk {
            *idx.add(kk) = order[kk] as i32;
            *wts.add(kk) = ex[kk] / zt;
        }
        *sgate.add(0) = l[e];
        *counter.add(0) = 0;
    })
}

/// int4 asymmetric row dot with raw pointers: expert tensors are larger than any
/// bound visible at the call site (indexed by global expert id), so the engine's
/// guarantee is used directly instead of a pre-sized slice.
unsafe fn dot_q4_ptr(q: SendPtr<u8>, sm: SendPtr<u16>, k: usize, row: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (j, &xv) in x.iter().enumerate().take(k) {
        let g = j / 64;
        let qq = ((q.read(row * (k / 2) + j / 2) >> ((j & 1) * 4)) & 15) as f32;
        acc += (qq * f16_to_f32(sm.read(row * (k / 64) * 2 + g * 2)) + f16_to_f32(sm.read(row * (k / 64) * 2 + g * 2 + 1))) * xv;
    }
    acc
}

/// int8 row dot with raw pointers (same reason as `dot_q4_ptr`).
unsafe fn dot_q8_ptr(q: SendPtr<i8>, s: SendPtr<u16>, k: usize, row: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (j, &xv) in x.iter().enumerate().take(k) {
        acc += (q.read(row * k + j) as f32) * f16_to_f32(s.read(row * (k / 128) + j / 128)) * xv;
    }
    acc
}

/// CPU: decode MoE — routed q4 experts + shared q8 expert.
/// Requires H == 2048, I == 512, topk == 8 (same limits as CUDA).
#[allow(clippy::too_many_arguments)]
pub unsafe fn dr_moe_decode(gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, sgu_q: *const i8, sgu_s: *const Half, sd_q: *const i8, sd_s: *const Half, x: *const f32, idx: *const i32, wts: *const f32, sgate: *const f32, res: *const f32, act: *mut f32, y: *mut f32, h: i32, i: i32, topk: i32, nw: *const f32, eps: f32, _pf: *const c_char, _pfb: usize, st: Stream) -> i32 {
    if h != 2048 || i != 512 || topk != 8 {
        return 1;
    }
    launch(st, move || {
        let (h, mi, topk) = (h as usize, i as usize, topk as usize);
        let k = h; // K = H = 2048 for gate_up rows
        // NOTE: gu/d weight tensors hold ALL experts ([E, ...]); rows are addressed
        // by global expert id inside the kernels.
        let xv: (MBuf, u64) = if nw.is_null() {
            require(x as *const c_void)
        } else {
            let (sb, soff) = scratch_f32(k);
            let nwb = require(nw as *const c_void);
            let xb = require(x as *const c_void);
            let prm = NormP { rows: 1, d: k as u32, eps };
            dispatch("rmsnorm", &[Some(xb), Some(nwb), Some((sb.clone(), soff)), None], pb(&prm), (1, 1, 1), (256, 1, 1), true);
            (sb, soff)
        };
        let prm = MoeP { h: h as u32, mi: mi as u32, topk: topk as u32, k: k as u32 };
        let nrows = (topk + 1) * mi;
        dispatch(
            "moe_gate_up",
            &[
                opt_buf(gu_q as *const c_void, 0),
                opt_buf(gu_sm as *const c_void, 0),
                opt_buf(sgu_q as *const c_void, 0),
                opt_buf(sgu_s as *const c_void, 0),
                Some(xv),
                opt_buf(idx as *const c_void, 0),
                opt_buf(act as *const c_void, 0),
            ],
            pb(&prm),
            (nrows as u64, 1, 1),
            (256, 1, 1),
            true,
        );
        dispatch(
            "moe_down",
            &[
                opt_buf(d_q as *const c_void, 0),
                opt_buf(d_sm as *const c_void, 0),
                opt_buf(sd_q as *const c_void, 0),
                opt_buf(sd_s as *const c_void, 0),
                opt_buf(act as *const c_void, 0),
                opt_buf(idx as *const c_void, 0),
                opt_buf(wts as *const c_void, 0),
                opt_buf(sgate as *const c_void, 0),
                opt_buf(res as *const c_void, 0),
                opt_buf(y as *const c_void, 0),
            ],
            pb(&prm),
            (h as u64, 1, 1),
            (256, 1, 1),
            true,
        );
    })
}

/// CPU: prefill MoE over expert-sorted rows. Requires H == 2048, I == 512.
pub unsafe fn dr_moe_prefill(gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, xp: *const Half, offsets: *const i32, counts: *const i32, act: *mut Half, y: *mut f32, e: i32, h: i32, i: i32, st: Stream) -> i32 {
    if h != 2048 || i != 512 {
        return 1;
    }
    launch(st, move || {
        let (e, h, mi) = (e as usize, h as usize, i as usize);
        let k = h; // K = 2048
        // slot -> expert map (uniform-cost slots; computed once on CPU).
        let mut se = Vec::with_capacity(1024);
        for ex in 0..e {
            let cnt = unsafe { *counts.add(ex) } as usize;
            se.extend(std::iter::repeat(ex as i32).take(cnt));
        }
        let total = se.len();
        if total == 0 {
            return;
        }

        let seb = scratch_f32(total);
        // NOTE: reuses the f32 scratch as raw bytes for the i32 slot map.
        unsafe { std::ptr::copy_nonoverlapping(se.as_ptr(), seb.0.0.contents() as *mut i32, se.len()) };
        let prm = MoePFP { total: total as u32, k: k as u32, mi: mi as u32, h: h as u32 };
        dispatch(
            "moe_pf_gate_up",
            &[
                opt_buf(gu_q as *const c_void, 0),
                opt_buf(gu_sm as *const c_void, 0),
                opt_buf(xp as *const c_void, 0),
                Some(seb.clone()),
                opt_buf(act as *const c_void, 0),
            ],
            pb(&prm),
            (total as u64, 1, 1),
            (256, 1, 1),
            true,
        );
        
        dispatch(
            "moe_pf_down",
            &[
                opt_buf(d_q as *const c_void, 0),
                opt_buf(d_sm as *const c_void, 0),
                opt_buf(act as *const c_void, 0),
                Some(seb),
                opt_buf(y as *const c_void, 0),
            ],
            pb(&prm),
            (total as u64, 1, 1),
            (256, 1, 1),
            true,
        );

    })
}

// ---------------------------------------------------------------------------
// CPU reference: embeddings

/// CPU: backbone input rows (prefill gather or decode candidate lookup + codes).
pub unsafe fn dr_embed(rows_emb: *const Half, cand_table: *const Half, frames: *const i32, fstride: i32, cb_emb: *const Half, x: *mut f32, rows: i32, d: i32, sem0: i32, sem1: i32, eos: i32, cb: i32, nb: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, d, fstride) = (rows as usize, d as usize, fstride as usize);
        for r in 0..rows {
            // Decode passes fstride = 0 for its single row; size the row view by use.
            let flen = fstride.max(nb as usize + 2);
            let f = std::slice::from_raw_parts(frames.add(r * fstride), flen);
            let main = f[0];
            let is_sem = (sem0..=sem1).contains(&main);
            let xr = std::slice::from_raw_parts_mut(x.add(r * d), d);
            if !rows_emb.is_null() {
                let er = std::slice::from_raw_parts(rows_emb.add(r * d), d);
                for (o, &v) in xr.iter_mut().zip(er.iter()) {
                    *o = bf16_to_f32(v);
                }
            } else {
                let ci = if main == eos { 0 } else { clampi(main - sem0, 0, cb - 1) + 1 };
                let er = std::slice::from_raw_parts(cand_table.add(ci as usize * d), d);
                for (o, &v) in xr.iter_mut().zip(er.iter()) {
                    *o = bf16_to_f32(v);
                }
            }
            if is_sem {
                let scale = (1.0 + nb as f32).sqrt().recip();
                for c in 1..=nb as usize {
                    let code = clampi(f[1 + c], 0, cb - 1) as usize;
                    let cbr = std::slice::from_raw_parts(cb_emb.add((c * cb as usize + code) * d), d);
                    for (o, &v) in xr.iter_mut().zip(cbr.iter()) {
                        *o += bf16_to_f32(v);
                    }
                }
                for o in xr.iter_mut() {
                    *o *= scale;
                }
            }
        }
    })
}

/// CPU: tower layer input — [rms(e) | rms(h)] with label/codebook gathers.
pub unsafe fn dr_tower_in(h: *const f32, frames: *const i32, fstride: i32, code_col: i32, in_frames: *const i32, in_stride: i32, tok_emb: *const Half, cb_emb: *const Half, enorm: *const f32, hnorm: *const f32, pre_norm: *const f32, y: *mut f32, yb: *mut Half, rows: i32, d: i32, sem0: i32, sem1: i32, cb: i32, ncb: i32, nb: i32, eps: f32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, d) = (rows as usize, d as usize);
        let (fstride, code_col) = (fstride as usize, code_col as usize);
        for r in 0..rows {
            // Decode passes fstride = 0 for its single row; size the row view by use.
            let flen = fstride.max(code_col + 1);
            let f = std::slice::from_raw_parts(frames.add(r * fstride), flen);
            let lab_sem = (sem0..=sem1).contains(&f[0]);
            let code = clampi(f[code_col], 0, cb - 1) as usize;
            let mut ev = vec![0.0f32; d];
            if lab_sem {
                let ter = std::slice::from_raw_parts(tok_emb.add(code * d), d);
                for (o, &v) in ev.iter_mut().zip(ter.iter()) {
                    *o = bf16_to_f32(v);
                }
            }
            let mut hv: Vec<f32> = f32s(h, (r + 1) * d)[r * d..].to_vec();
            if !in_frames.is_null() {
                let ilen = (in_stride as usize).max(ncb as usize + 1);
                let fi = std::slice::from_raw_parts(in_frames.add(r * in_stride as usize), ilen);
                if (sem0..=sem1).contains(&fi[0]) {
                    for c in (1 + nb as usize)..ncb as usize {
                        let cc = clampi(fi[1 + c], 0, cb - 1) as usize;
                        let cbr = std::slice::from_raw_parts(cb_emb.add((c * cb as usize + cc) * d), d);
                        for (o, &v) in ev.iter_mut().zip(cbr.iter()) {
                            *o += bf16_to_f32(v);
                        }
                    }
                }
            }
            if !pre_norm.is_null() {
                let inv = rms_inv(&hv, eps);
                let pn = f32s(pre_norm, d);
                for (o, &w) in hv.iter_mut().zip(pn.iter()) {
                    *o = *o * inv * w;
                }
            }
            let ie = rms_inv(&ev, eps);
            let ih = rms_inv(&hv, eps);
            let en = f32s(enorm, d);
            let hn = f32s(hnorm, d);
            for i in 0..d {
                let a = ev[i] * ie * en[i];
                let b = hv[i] * ih * hn[i];
                if !y.is_null() {
                    *y.add(r * 2 * d + i) = a;
                    *y.add(r * 2 * d + d + i) = b;
                }
                if !yb.is_null() {
                    *yb.add(r * 2 * d + i) = f32_to_bf16_rn(a);
                    *yb.add(r * 2 * d + d + i) = f32_to_bf16_rn(b);
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// CPU reference: full attention

/// CPU: per-(row, head) RMSNorm, rotary embedding, KV-cache write.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dr_qk_norm_rope(qkv: *const f32, rs: i32, rows: i32, h: i32, hkv: i32, hd: i32, qn: *const f32, kn: *const f32, inv_freq: *const f32, rot: i32, interleaved: i32, pos_base: *const i32, pos_off: i32, q_out: *mut f32, kc: *mut Half, vc: *mut Half, eps: f32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rs, rows, h, hkv, hd, rot) = (rs as usize, rows as usize, h as usize, hkv as usize, hd as usize, rot as usize);
        for r in 0..rows {
            let pos = (*pos_base.add(0) + pos_off) as usize + r;
            let row = f32s(qkv, (r + 1) * rs);
            let row = &row[r * rs..];
            for hh in 0..h + hkv {
                let is_q = hh < h;
                let src_off = if is_q { hh * hd } else { 2 * h * hd + (hh - h) * hd };
                let src = &row[src_off..src_off + hd];
                let nw = f32s(if is_q { qn } else { kn }, hd);
                let inv = rms_inv(src, eps);
                let mut b = vec![0.0f32; hd];
                for t in 0..hd {
                    b[t] = src[t] * inv * nw[t];
                }
                // Rotary on the first `rot` dims.
                let mut o = b.clone();
                if interleaved != 0 {
                    for i in 0..rot / 2 {
                        let f = *inv_freq.add(i);
                        let (sn, cs) = (pos as f32 * f).sin_cos();
                        o[2 * i] = b[2 * i] * cs - b[2 * i + 1] * sn;
                        o[2 * i + 1] = b[2 * i + 1] * cs + b[2 * i] * sn;
                    }
                } else {
                    let half = rot / 2;
                    for i in 0..half {
                        let f = *inv_freq.add(i);
                        let (sn, cs) = (pos as f32 * f).sin_cos();
                        o[i] = b[i] * cs - b[i + half] * sn;
                        o[i + half] = b[i + half] * cs + b[i] * sn;
                    }
                }
                if is_q {
                    let dst = std::slice::from_raw_parts_mut(q_out.add(r * h * hd + hh * hd), hd);
                    dst.copy_from_slice(&o);
                } else {
                    let kh = hh - h;
                    let at = (pos * hkv + kh) * hd;
                    for t in 0..hd {
                        *kc.add(at + t) = f32_to_bf16_rn(o[t]);
                        *vc.add(at + t) = f32_to_bf16_rn(row[2 * h * hd + hkv * hd + kh * hd + t]);
                    }
                }
            }
        }
    })
}

/// Shared softmax attention over cached keys/values: out = softmax(qK/sqrt(hd))V / L,
/// multiplied by sigmoid(gate). `nrows` query rows starting at `pos0`.
unsafe fn attn_rows(q: &[f32], kc: &[u16], vc: &[u16], gate: &[f32], gate_stride: usize, nrows: usize, h: usize, hkv: usize, hd: usize, pos0: usize, out: SendPtr<f32>) {
    let scale = (hd as f32).sqrt().recip();
    let g = h / hkv;
    par_for!(nrows, |lo, hi| {
        for r in lo..hi {
            let nkeys = pos0 + r + 1;
            for hh in 0..h {
                let kh = hh / g;
                let qr = &q[r * h * hd + hh * hd..(r + 1) * h * hd];
                let qr = &qr[..hd];
                // Scores then online softmax.
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut acc = vec![0.0f32; hd];
                for j in 0..nkeys {
                    let base = (j * hkv + kh) * hd;
                    let mut s = 0.0f32;
                    for d in 0..hd {
                        s += qr[d] * bf16_to_f32(kc[base + d]);
                    }
                    s *= scale;
                    let mn = m.max(s);
                    let corr = (m - mn).exp();
                    let p = (s - mn).exp();
                    l = l * corr + p;
                    for d in 0..hd {
                        acc[d] = acc[d] * corr + p * bf16_to_f32(vc[base + d]);
                    }
                    m = mn;
                }
                for d in 0..hd {
                    let gv = gate[r * gate_stride + hh * hd + d];
                    out.write(r * h * hd + hh * hd + d, acc[d] / l * sigmoid(gv));
                }
            }
        }
    });
}

/// CPU: prefill attention over keys 0..pos+r.
pub unsafe fn dr_attn_prefill(q: *const f32, kc: *const Half, vc: *const Half, gate: *const f32, gate_stride: i32, rows: i32, h: i32, hkv: i32, hd: i32, pos_base: *const i32, pos_off: i32, o: *mut f32, st: Stream) -> i32 {
    if hd != 256 {
        return 1;
    }
    launch(st, move || {
        flush_ok();
        let (rows, h, hkv, hd) = (rows as usize, h as usize, hkv as usize, hd as usize);
        let pos0 = (*pos_base.add(0) + pos_off) as usize;
        // KV caches are append-only and sized max_len rows; the prompt fits by construction.
        let max_keys = pos0 + rows;
        let kcv = std::slice::from_raw_parts(kc, max_keys * hkv * hd);
        let vcv = std::slice::from_raw_parts(vc, max_keys * hkv * hd);
        let qv = f32s(q, rows * h * hd);
        let gv = f32s(gate, rows * gate_stride as usize);
        attn_rows(qv, kcv, vcv, gv, gate_stride as usize, rows, h, hkv, hd, pos0, SendPtr(o));
    })
}

/// CPU: single-row decode attention (the `part` buffer is unused on this path).
pub unsafe fn dr_attn_decode(q: *const f32, kc: *const Half, vc: *const Half, gate: *const f32, h: i32, hkv: i32, hd: i32, pos_base: *const i32, pos_off: i32, _part: *mut f32, o: *mut f32, st: Stream) -> i32 {
    if hd != 256 {
        return 1;
    }
    // Bisect hatch: DRAMA_CPU_ATTN=1 restores the pre-H4 CPU path.
    if std::env::var("DRAMA_CPU_ATTN").is_ok() {
        return launch(st, move || {
            flush_ok();
            let (h, hkv, hd) = (h as usize, hkv as usize, hd as usize);
            let pos0 = (*pos_base.add(0) + pos_off) as usize;
            let max_keys = pos0 + 1;
            let kcv = std::slice::from_raw_parts(kc, max_keys * hkv * hd);
            let vcv = std::slice::from_raw_parts(vc, max_keys * hkv * hd);
            let qv = f32s(q, h * hd);
            let gv = f32s(gate, h * hd);
            attn_rows(qv, kcv, vcv, gv, h * hd, 1, h, hkv, hd, pos0, SendPtr(o));
        });
    }
    launch(st, move || {
        let (h, hkv, hd) = (h as usize, hkv as usize, hd as usize);
        let pos0 = (*pos_base.add(0) + pos_off) as usize;
        let nkeys = pos0 + 1;
        let prm = AttnP { h: h as u32, hkv: hkv as u32, hd: hd as u32, nkeys: nkeys as u32, scale: (hd as f32).sqrt().recip() };
        dispatch(
            "attn_decode",
            &[
                opt_buf(q as *const c_void, 0),
                opt_buf(kc as *const c_void, 0),
                opt_buf(vc as *const c_void, 0),
                opt_buf(gate as *const c_void, 0),
                opt_buf(o as *const c_void, 0),
            ],
            pb(&prm),
            (h as u64, 1, 1),
            (256, 1, 1),
            true,
        );
    })
}

// ---------------------------------------------------------------------------
// CPU reference: linear attention (depthwise conv + gated DeltaNet)

/// CPU: causal depthwise conv (kernel 4) + SiLU; state keeps 3 inputs/channel.
pub unsafe fn dr_conv(input: *const f32, in_stride: i32, w: *const f32, state: *mut f32, out: *mut f32, n: i32, c: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (n, c, ins) = (n as usize, c as usize, in_stride as usize);
        let (input, w, state, out) = (SendPtr(input as *mut f32), SendPtr(w as *mut f32), SendPtr(state), SendPtr(out));
        par_for!(c, |lo, hi| {
            for ch in lo..hi {
                let mut h = unsafe { [state.read(ch * 3), state.read(ch * 3 + 1), state.read(ch * 3 + 2)] };
                let wv = unsafe { [w.read(ch * 4), w.read(ch * 4 + 1), w.read(ch * 4 + 2), w.read(ch * 4 + 3)] };
                for t in 0..n {
                    let xt = unsafe { input.read(t * ins + ch) };
                    unsafe { out.write(t * c + ch, silu(wv[0] * h[0] + wv[1] * h[1] + wv[2] * h[2] + wv[3] * xt)); }
                    h[0] = h[1];
                    h[1] = h[2];
                    h[2] = xt;
                }
                unsafe {
                    state.write(ch * 3, h[0]);
                    state.write(ch * 3 + 1, h[1]);
                    state.write(ch * 3 + 2, h[2]);
                }
            }
        });
    })
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (x.exp() + 1.0).ln() }
}

/// CPU: gated DeltaNet recurrence with gated RMSNorm output. dk == dv == 128 only.
#[allow(clippy::too_many_arguments)]
unsafe fn dr_gdn_cpu(conv: *const f32, proj: *const f32, ps: i32, z_off: i32, b_off: i32, a_off: i32, a_log: *const f32, dt_bias: *const f32, norm_w: *const f32, state: *mut f32, o: *mut f32, n: i32, nk: i32, nv: i32, dk: i32, dv: i32, eps: f32, st: Stream) -> i32 {
    if dk != 128 || dv != 128 {
        return 1;
    }
    launch(st, move || {
        flush_ok();
        const DK: usize = 128;
        let (n, nk, nv) = (n as usize, nk as usize, nv as usize);
        let (ps, z_off, b_off, a_off) = (ps as usize, z_off as usize, b_off as usize, a_off as usize);
        let cs = 2 * nk * DK + nv * DK;
        let qscale = (DK as f32).sqrt().recip();
        let (conv, proj, a_log, dt_bias, norm_w, state, o) =
            (SendPtr(conv as *mut f32), SendPtr(proj as *mut f32), SendPtr(a_log as *mut f32), SendPtr(dt_bias as *mut f32), SendPtr(norm_w as *mut f32), SendPtr(state), SendPtr(o));
        par_for!(nv, |lo, hi| {
            for hh in lo..hi {
            let kh = hh / (nv / nk);
            let mut s = vec![0.0f32; DK * DK];
            for k in 0..DK {
                for v in 0..DK {
                    s[k * DK + v] = unsafe { state.read(hh * DK * DK + k * DK + v) };
                }
            }
            let a = -unsafe { a_log.read(hh) }.exp();
            let dtb = unsafe { dt_bias.read(hh) };
            let convp = conv.ptr() as *const f32;
            let projp = proj.ptr() as *const f32;
            for t in 0..n {
                let cr = f32s(convp, (t + 1) * cs);
                let cr = &cr[t * cs..];
                let pr = f32s(projp, (t + 1) * ps);
                let pr = &pr[t * ps..];
                // q/k normalisation WITHOUT the head-dim average (matches k_gdn:
                // rsqrt(sum) * qscale, not an RMSNorm).
                let qv: Vec<f32> = (0..DK).map(|i| cr[kh * DK + i]).collect();
                let kv: Vec<f32> = (0..DK).map(|i| cr[nk * DK + kh * DK + i]).collect();
                let qi = qscale / (qv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let ki = 1.0 / (kv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let mut qs = vec![0.0f32; DK];
                let mut ks = vec![0.0f32; DK];
                for i in 0..DK {
                    qs[i] = qv[i] * qi;
                    ks[i] = kv[i] * ki;
                }
                let vv: Vec<f32> = (0..DK).map(|v| cr[2 * nk * DK + hh * DK + v]).collect();
                let decay = (a * softplus(pr[a_off + hh] + dtb)).exp();
                let beta = sigmoid(pr[b_off + hh]);
                let mut outv = vec![0.0f32; DK];
                for v in 0..DK {
                    let mut kvm = 0.0f32;
                    for k in 0..DK {
                        s[k * DK + v] *= decay;
                        kvm += s[k * DK + v] * ks[k];
                    }
                    let delta = (vv[v] - kvm) * beta;
                    let mut oacc = 0.0f32;
                    for k in 0..DK {
                        s[k * DK + v] += ks[k] * delta;
                        oacc += s[k * DK + v] * qs[k];
                    }
                    outv[v] = oacc;
                }
                let zr: Vec<f32> = (0..DK).map(|v| pr[z_off + hh * DK + v]).collect();
                let inv = rms_inv(&outv, eps);
                for v in 0..DK {
                    unsafe { o.write(t * nv * DK + hh * DK + v, outv[v] * inv * norm_w.read(v) * silu(zr[v])); }
                }
            }
            for k in 0..DK {
                for v in 0..DK {
                    unsafe { state.write(hh * DK * DK + k * DK + v, s[k * DK + v]); }
                }
            }
            }
        });
    })
}

pub unsafe fn dr_gdn(conv: *const f32, proj: *const f32, ps: i32, z_off: i32, b_off: i32, a_off: i32, a_log: *const f32, dt_bias: *const f32, norm_w: *const f32, state: *mut f32, o: *mut f32, n: i32, nk: i32, nv: i32, dk: i32, dv: i32, eps: f32, st: Stream) -> i32 {
    if dk != 128 || dv != 128 {
        return 1;
    }
    // Bisect hatch: DRAMA_CPU_GDN=1 restores the pre-H5 CPU path.
    if std::env::var("DRAMA_CPU_GDN").is_ok() {
        return unsafe { dr_gdn_cpu(conv, proj, ps, z_off, b_off, a_off, a_log, dt_bias, norm_w, state, o, n, nk, nv, dk, dv, eps, st) };
    }
    launch(st, move || {
        let prm = GdnP {
            n: n as u32, nk: nk as u32, nv: nv as u32, ps: ps as u32,
            z_off: z_off as u32, b_off: b_off as u32, a_off: a_off as u32, eps,
        };
        dispatch(
            "gdn",
            &[
                opt_buf(conv as *const c_void, 0),
                opt_buf(proj as *const c_void, 0),
                opt_buf(a_log as *const c_void, 0),
                opt_buf(dt_bias as *const c_void, 0),
                opt_buf(norm_w as *const c_void, 0),
                opt_buf(state as *const c_void, 0),
                opt_buf(o as *const c_void, 0),
            ],
            pb(&prm),
            (nv as u64, 1, 1),
            (128, 1, 1),
            true,
        );
        if std::env::var("DRAMA_DEBUG_GDN").is_ok() {
            flush_ok();
            let oo = unsafe { std::slice::from_raw_parts(o as *const f32, (n as usize) * (nv as usize) * 128) };
            let bad = oo.iter().filter(|x| !x.is_finite()).count();
            let mx = oo.iter().cloned().fold(0f32, |a, x| if x.is_finite() { a.max(x.abs()) } else { a });
            eprintln!("GDN-OUT n={} nv={} bad={} maxabs={:.3}", n, nv, bad, mx);
        }
    })
}

// ---------------------------------------------------------------------------
// CPU reference: codec (kernels/codec.cu). Activations are time-major [T, C].
// `x` may point into a buffer with readable history rows before it (causal
// convs), so negative offsets are valid and handled with raw arithmetic.

fn snake(x: f32, a: f32) -> f32 {
    let s = (a * x).sin();
    x + s * s / (a + 1e-9)
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, |err| <= 1.5e-7.
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    const P: f32 = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

fn gelu_erf(v: f32) -> f32 {
    0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752))
}

/// Round-to-nearest-even f32 -> f16 bits (matches `__float2half`:
/// finite overflow -> +inf, never NaN). Delegated to the `half` crate
/// (the hand-rolled version misclassified finite 70000.0 as NaN).
/// Round-to-nearest-even f32 -> f16 bits; also used by tests.
pub fn f32_to_f16_rn(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// CPU: out[t] = sum_c table[row(c, codes[t, c])].
pub unsafe fn dr_rvq_sum(codes: *const i32, frames: i32, ncb: i32, table: *const f32, out: *mut f32, d: i32, sem_size: i32, res_size: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (frames, ncb, d) = (frames as usize, ncb as usize, d as usize);
        let (sem, res) = (sem_size as usize, res_size as usize);
        let out = SendPtr(out);
        let codes = unsafe { std::slice::from_raw_parts(codes, frames * ncb) };
        let table = unsafe { std::slice::from_raw_parts(table, (sem + (ncb - 1) * res) * d) };
        par_for!(frames, |lo, hi| {
            for t in lo..hi {
                let orow = unsafe { std::slice::from_raw_parts_mut(out.ptr().add(t * d), d) };
                orow.fill(0.0);
                for c in 0..ncb {
                    let size = if c == 0 { sem } else { res };
                    let code = codes[t * ncb + c].clamp(0, size as i32 - 1) as usize;
                    let row = if c == 0 { code } else { sem + (c - 1) * res + code };
                    let tr = &table[row * d..(row + 1) * d];
                    for (o, &v) in orow.iter_mut().zip(tr.iter()) {
                        *o += v;
                    }
                }
            }
        });
    })
}

/// CPU: causal im2col with optional snake -> f16.
pub unsafe fn dr_im2col(x: *const f32, n: i32, c: i32, k: i32, d: i32, s: i32, alpha: *const f32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (n, c, k, dil, strd) = (n as usize, c as usize, k as usize, d as isize, s as isize);
        let has_a = !alpha.is_null();
        let (x, alpha, out) = (SendPtr(x as *mut f32), SendPtr(alpha as *mut f32), SendPtr(out));
        par_for!(n * k * c, |lo, hi| {
            for i in lo..hi {
                let cc = i % c;
                let tj = i / c;
                let j = (tj % k) as isize;
                let t = (tj / k) as isize;
                let src = t * strd + (strd - 1) - ((k as isize - 1 - j) * dil);
                let mut v = unsafe { *x.at(src * c as isize + cc as isize) };
                if has_a {
                    v = snake(v, unsafe { alpha.read(cc) });
                }
                unsafe { out.write(i, f32_to_f16_rn(v)); }
            }
        });
    })
}

/// CPU: one residual-VQ stage per row (argmax cosine, lower-index tie-break).
pub unsafe fn dr_vq_encode(r: *mut f32, rows: i32, d: i32, in_w: *const f32, in_b: *const f32, codebook_n: *const f32, n: i32, table: *const f32, codes: *mut i32, code_stride: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, d) = (rows as usize, d as usize);
        let (r, in_w, in_b, codebook_n, table, codes) = (SendPtr(r), SendPtr(in_w as *mut f32), SendPtr(in_b as *mut f32), SendPtr(codebook_n as *mut f32), SendPtr(table as *mut f32), SendPtr(codes));
        par_for!(rows, |lo, hi| {
            for row in lo..hi {
                let rr = unsafe { std::slice::from_raw_parts_mut(r.ptr().add(row * d), d) };
                let mut z = [0.0f32; 8];
                for o in 0..8 {
                    let wr = unsafe { std::slice::from_raw_parts(in_w.ptr().add(o * d) as *const f32, d) };
                    z[o] = wr.iter().zip(rr.iter()).map(|(a, b)| a * b).sum::<f32>() + unsafe { in_b.read(o) };
                }
                let mut best = 0i32;
                let mut bv = f32::NEG_INFINITY;
                for e in 0..n as usize {
                    let er = unsafe { std::slice::from_raw_parts(codebook_n.ptr().add(e * 8) as *const f32, 8) };
                    let dot: f32 = z.iter().zip(er.iter()).map(|(a, b)| a * b).sum();
                    if dot > bv {
                        bv = dot;
                        best = e as i32;
                    }
                }
                let tr = unsafe { std::slice::from_raw_parts(table.ptr().add(best as usize * d) as *const f32, d) };
                for (a, &b) in rr.iter_mut().zip(tr.iter()) {
                    *a -= b;
                }
                unsafe { codes.write(row * code_stride as usize, best) };
            }
        });
    })
}

/// CPU: transposed-conv input gather ([snake(x[t]) | snake(x[t-1])]) -> f16.
pub unsafe fn dr_tconv_gather(x: *const f32, n: i32, c: i32, two: i32, alpha: *const f32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (n, c) = (n as usize, c as usize);
        let w = if two != 0 { 2 * c } else { c };
        let has_a = !alpha.is_null();
        let (x, alpha, out) = (SendPtr(x as *mut f32), SendPtr(alpha as *mut f32), SendPtr(out));
        par_for!(n * w, |lo, hi| {
            for i in lo..hi {
                let t = (i / w) as isize;
                let cc = i % w;
                let ch = cc % c;
                let src = if cc < c { t } else { t - 1 };
                let mut v = unsafe { *x.at(src * c as isize + ch as isize) };
                if has_a {
                    v = snake(v, unsafe { alpha.read(ch) });
                }
                unsafe { out.write(i, f32_to_f16_rn(v)); }
            }
        });
    })
}

/// CPU: biased/scaled activation (+residual) -> f32 and/or f16.
pub unsafe fn dr_epilogue(y: *const f32, n: usize, c: i32, bias: *const f32, scale: *const f32, res: *const f32, act: i32, out: *mut f32, out16: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let c = c as usize;
        let (y, bias, scale, res, out, out16) =
            (SendPtr(y as *mut f32), SendPtr(bias as *mut f32), SendPtr(scale as *mut f32), SendPtr(res as *mut f32), SendPtr(out), SendPtr(out16));
        par_for!(n, |lo, hi| {
            for i in lo..hi {
                let cc = i % c;
                let mut v = unsafe { y.read(i) } + if bias.is_null() { 0.0 } else { unsafe { bias.read(cc) } };
                if act == 1 {
                    v = v.tanh();
                } else if act == 2 {
                    v = gelu_erf(v);
                }
                if !scale.is_null() {
                    v *= unsafe { scale.read(cc) };
                }
                if !res.is_null() {
                    v += unsafe { res.read(i) };
                }
                if !out.is_null() {
                    unsafe { out.write(i, v) };
                }
                if !out16.is_null() {
                    unsafe { out16.write(i, f32_to_f16_rn(v)) };
                }
            }
        });
    })
}

/// CPU: depthwise causal conv + bias.
pub unsafe fn dr_dwconv(x: *const f32, t: i32, c: i32, k: i32, w: *const f32, b: *const f32, out: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (t, c, k) = (t as usize, c as usize, k as usize);
        let (x, w, b, out) = (SendPtr(x as *mut f32), SendPtr(w as *mut f32), SendPtr(b as *mut f32), SendPtr(out));
        par_for!(t, |lo, hi| {
            for tt in lo..hi {
                for cc in 0..c {
                    let mut acc = unsafe { b.read(cc) };
                    for j in 0..k {
                        acc += unsafe { w.read(cc * k + j) * *x.at((tt as isize - (k - 1 - j) as isize) * c as isize + cc as isize) };
                    }
                    unsafe { out.write(tt * c + cc, acc) };
                }
            }
        });
    })
}

/// CPU: affine LayerNorm per row -> f16.
pub unsafe fn dr_layernorm(x: *const f32, rows: i32, c: i32, w: *const f32, b: *const f32, eps: f32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, c) = (rows as usize, c as usize);
        let (x, w, b, out) = (SendPtr(x as *mut f32), SendPtr(w as *mut f32), SendPtr(b as *mut f32), SendPtr(out));
        par_for!(rows, |lo, hi| {
            for r in lo..hi {
                let xr = unsafe { std::slice::from_raw_parts(x.ptr() as *const f32, (r + 1) * c) };
                let xr = &xr[r * c..];
                let mean = xr.iter().sum::<f32>() / c as f32;
                let inv = (xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32 + eps).sqrt().recip();
                for i in 0..c {
                    unsafe { out.write(r * c + i, f32_to_f16_rn((xr[i] - mean) * inv * w.read(i) + b.read(i))) };
                }
            }
        });
    })
}

/// CPU: plain-weight RMSNorm per row -> f32 and/or f16.
pub unsafe fn dr_rmsnorm_plain(x: *const f32, rows: i32, c: i32, w: *const f32, eps: f32, out: *mut f32, out16: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, c) = (rows as usize, c as usize);
        let (x, w, out, out16) = (SendPtr(x as *mut f32), SendPtr(w as *mut f32), SendPtr(out), SendPtr(out16));
        par_for!(rows, |lo, hi| {
            for r in lo..hi {
                let xr = unsafe { std::slice::from_raw_parts(x.ptr() as *const f32, (r + 1) * c) };
                let xr = &xr[r * c..];
                let inv = rms_inv(xr, eps);
                for i in 0..c {
                    let v = xr[i] * inv * unsafe { w.read(i) };
                    if !out.is_null() {
                        unsafe { out.write(r * c + i, v) };
                    }
                    if !out16.is_null() {
                        unsafe { out16.write(r * c + i, f32_to_f16_rn(v)) };
                    }
                }
            }
        });
    })
}

/// CPU: codec qkv split + interleaved rope (cos/sin rounded via bf16, like the
/// reference table) + ring KV-cache write.
pub unsafe fn dr_codec_rope(qkv: *const f32, n: i32, h: i32, inv_freq: *const f32, t0: i32, ring: i32, q: *mut f32, kc: *mut f32, vc: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (n, h) = (n as usize, h as usize);
        let w = h * 64;
        let (qkv, inv_freq, q, kc, vc) = (SendPtr(qkv as *mut f32), SendPtr(inv_freq as *mut f32), SendPtr(q), SendPtr(kc), SendPtr(vc));
        par_for!(n, |lo, hi| {
            for r in lo..hi {
                let t = t0 as usize + r;
                let slot = (t % ring as usize) * w;
                for i in 0..w / 2 {
                    let head = i / 32;
                    let p = i % 32;
                    let (sn, cs) = unsafe { (t as f32 * inv_freq.read(p)).sin_cos() };
                    let c = bf16_to_f32(f32_to_bf16_rn(cs));
                    let s = bf16_to_f32(f32_to_bf16_rn(sn));
                    let base = r * 3 * w;
                    let e = head * 64 + 2 * p;
                    let q0 = unsafe { qkv.read(base + e) };
                    let q1 = unsafe { qkv.read(base + e + 1) };
                    let k0 = unsafe { qkv.read(base + w + e) };
                    let k1 = unsafe { qkv.read(base + w + e + 1) };
                    unsafe {
                        q.write(r * w + e, q0 * c - q1 * s);
                        q.write(r * w + e + 1, q1 * c + q0 * s);
                        kc.write(slot + e, k0 * c - k1 * s);
                        kc.write(slot + e + 1, k1 * c + k0 * s);
                        vc.write(slot + e, qkv.read(base + 2 * w + e));
                        vc.write(slot + e + 1, qkv.read(base + 2 * w + e + 1));
                    }
                }
            }
        });
    })
}

/// CPU: causal windowed attention over the ring caches, head_dim 64 -> f16.
pub unsafe fn dr_codec_attn(q: *const f32, kc: *const f32, vc: *const f32, n: i32, h: i32, window: i32, t0: i32, ring: i32, out: *mut Half, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (n, h) = (n as usize, h as usize);
        let w = h * 64;
        let (q, kc, vc, out) = (SendPtr(q as *mut f32), SendPtr(kc as *mut f32), SendPtr(vc as *mut f32), SendPtr(out));
        par_for!(n, |lo, hi| {
            for r in lo..hi {
                let t = t0 as usize + r;
                let start = t.saturating_sub(window as usize - 1);
                for hh in 0..h {
                    let mut m = f32::NEG_INFINITY;
                    let mut l = 0.0f32;
                    let mut acc = [0.0f32; 64];
                    for j in start..=t {
                        let at = (j % ring as usize) * w + hh * 64;
                        let mut s = 0.0f32;
                        for dd in 0..64 {
                            s += unsafe { q.read(r * w + hh * 64 + dd) * kc.read(at + dd) };
                        }
                        s *= 0.125;
                        let mn = m.max(s);
                        let corr = (m - mn).exp();
                        let p = (s - mn).exp();
                        l = l * corr + p;
                        for dd in 0..64 {
                            acc[dd] = unsafe { acc[dd] * corr + p * vc.read(at + dd) };
                        }
                        m = mn;
                    }
                    for dd in 0..64 {
                        unsafe { out.write(r * w + hh * 64 + dd, f32_to_f16_rn(acc[dd] / l)) };
                    }
                }
            }
        });
    })
}

/// CPU: y[t, j] = silu(h[t, j]) * h[t, I + j] -> f16.
pub unsafe fn dr_swiglu16(h: *const f32, y: *mut Half, rows: i32, i: i32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (rows, ii) = (rows as usize, i as usize);
        let (h, y) = (SendPtr(h as *mut f32), SendPtr(y));
        par_for!(rows, |lo, hi| {
            for r in lo..hi {
                for j in 0..ii {
                    let g = unsafe { h.read(r * 2 * ii + j) };
                    let u = unsafe { h.read(r * 2 * ii + ii + j) };
                    unsafe { y.write(r * ii + j, f32_to_f16_rn(silu(g) * u)) };
                }
            }
        });
    })
}

/// CPU: final layer — tanh over the snake-activated causal conv.
pub unsafe fn dr_conv_out(x: *const f32, n: i32, c: i32, k: i32, alpha: *const f32, w: *const f32, b: *const f32, out: *mut f32, st: Stream) -> i32 {
    launch(st, move || {
        flush_ok();
        let (n, c, k) = (n as usize, c as usize, k as usize);
        let (x, alpha, w, b, out) = (SendPtr(x as *mut f32), SendPtr(alpha as *mut f32), SendPtr(w as *mut f32), SendPtr(b as *mut f32), SendPtr(out));
        par_for!(n, |lo, hi| {
            for t in lo..hi {
                let mut acc = 0.0f32;
                for j in 0..k {
                    for cc in 0..c {
                        acc += unsafe { w.read(j * c + cc) * snake(*x.at((t as isize - (k - 1 - j) as isize) * c as isize + cc as isize), alpha.read(cc)) };
                    }
                }
                unsafe { out.write(t, (acc + b.read(0)).tanh()) };
            }
        });
    })
}


