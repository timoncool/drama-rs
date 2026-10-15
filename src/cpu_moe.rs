//! Routed experts computed on the CPU (as KTransformers and Fiddler do for one token): when an expert is not in the
//! VRAM cache, moving its 8 KB input and output costs far less than moving its 1.7 MB of weights over PCIe. A pool of
//! worker threads waits for jobs that CUDA host functions hand over inside the decode graph.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What the GPU leaves for the CPU at one layer (in page-locked memory), and what the CPU gives back.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Io {
    pub x: [f32; HIDDEN],
    pub idx: [i32; TOPK],
    pub wts: [f32; TOPK],
    pub fill: [i32; TOPK],
    pub out: [f32; HIDDEN],
}

pub const HIDDEN: usize = 2048;
pub const INTER: usize = 512;
pub const TOPK: usize = 8;

/// One layer's experts in RAM (host addresses) and its post-attention norm.
#[derive(Clone, Copy)]
pub struct LayerExperts {
    pub gu_q: *const u8,
    pub gu_sm: *const u16,
    pub d_q: *const u8,
    pub d_sm: *const u16,
    pub norm: *const f32,
}

unsafe impl Send for LayerExperts {}
unsafe impl Sync for LayerExperts {}

struct Job {
    layer: LayerExperts,
    io: *mut Io,
    eps: f32,
    /// The missed experts: (expert, weight).
    experts: [(usize, f32); TOPK],
    n: usize,
    xn: Quant<HIDDEN>,
    act: [[f32; INTER]; TOPK],
}

/// Activations quantized for the int4 x int8 dot (as llama.cpp's Q8_0): per 32 inputs one scale; inside a block the
/// even inputs come first, then the odd ones, the order the low and high nibbles of the weights unpack in. `gsum`
/// holds the plain sums per 64 inputs for the groups' min term.
#[derive(Clone, Copy)]
struct Quant<const K: usize> {
    q: [[i8; 32]; 64],
    d: [f32; 64],
    gsum: [f32; 32],
}

impl<const K: usize> Quant<K> {
    const fn zero() -> Self {
        Self { q: [[0; 32]; 64], d: [0.0; 64], gsum: [0.0; 32] }
    }

    fn set(&mut self, x: &[f32]) {
        for b in 0..K / 32 {
            let xb = &x[b * 32..b * 32 + 32];
            let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            self.d[b] = d;
            for i in 0..16 {
                self.q[b][i] = (xb[2 * i] * inv).round() as i8;
                self.q[b][16 + i] = (xb[2 * i + 1] * inv).round() as i8;
            }
        }
        for g in 0..K / 64 {
            self.gsum[g] = x[g * 64..g * 64 + 64].iter().sum();
        }
    }
}

struct Shared {
    job: UnsafeCell<Job>,
    generation: AtomicU64,
    phase_a: AtomicUsize,
    phase_b: AtomicUsize,
    finished: AtomicU64,
    stop: AtomicBool,
    threads: usize,
}

unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct CpuMoe {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    pub layers: Vec<LayerExperts>,
}

/// Spinning for a new job this long keeps the next layer's hand-over fast; after it a worker sleeps.
const SPIN: Duration = Duration::from_micros(200);

impl CpuMoe {
    pub fn new(layers: Vec<LayerExperts>, eps: f32) -> Self {
        let threads = std::env::var("DRAMA_CPU_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (std::thread::available_parallelism().map_or(8, |n| n.get()) / 2).max(1));
        let shared = Arc::new(Shared {
            job: UnsafeCell::new(Job {
                layer: layers[0],
                io: std::ptr::null_mut(),
                eps,
                experts: [(0, 0.0); TOPK],
                n: 0,
                xn: Quant::zero(),
                act: [[0.0; INTER]; TOPK],
            }),
            generation: AtomicU64::new(0),
            phase_a: AtomicUsize::new(0),
            phase_b: AtomicUsize::new(0),
            finished: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            threads,
        });
        let workers = (0..threads)
            .map(|t| {
                let s = shared.clone();
                std::thread::spawn(move || worker(&s, t))
            })
            .collect();
        eprintln!("CPU experts: {threads} threads, {}", if avx2() { "AVX2" } else { "scalar" });
        Self { shared, workers, layers }
    }

    /// Hand over layer `layer`'s missed experts from `io` (called on the CUDA host-function thread).
    ///
    /// # Safety
    /// `io` stays valid until `wait` returns; only one job runs at a time.
    pub unsafe fn submit(&self, layer: usize, io: *mut Io) {
        let s = &*self.shared;
        let job = &mut *s.job.get();
        let io_ref = &*io;
        job.layer = self.layers[layer];
        job.io = io;
        job.n = 0;
        for k in 0..TOPK {
            if io_ref.fill[k] != -1 {
                job.experts[job.n] = (io_ref.idx[k] as usize, io_ref.wts[k]);
                job.n += 1;
            }
        }
        let norm = std::slice::from_raw_parts(job.layer.norm, HIDDEN);
        let ss: f32 = io_ref.x.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / HIDDEN as f32 + job.eps).sqrt();
        let mut xn = [0.0f32; HIDDEN];
        for i in 0..HIDDEN {
            xn[i] = io_ref.x[i] * inv * norm[i];
        }
        job.xn.set(&xn);
        s.phase_a.store(0, Ordering::Relaxed);
        s.phase_b.store(0, Ordering::Relaxed);
        // a job with nothing to compute is finished before any worker can see it
        let g = s.generation.load(Ordering::Relaxed) + 1;
        if job.n == 0 {
            (*io).out.fill(0.0);
            s.finished.store(g, Ordering::Release);
        }
        s.generation.store(g, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark();
        }
    }

    /// Block until the job handed over last is done; its output is in `io.out`.
    pub fn wait(&self) {
        let s = &*self.shared;
        let g = s.generation.load(Ordering::Acquire);
        while s.finished.load(Ordering::Acquire) < g {
            std::hint::spin_loop();
        }
    }
}

impl Drop for CpuMoe {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(s: &Shared, t: usize) {
    let mut seen = 0u64;
    loop {
        let t0 = Instant::now();
        let g = loop {
            if s.stop.load(Ordering::Acquire) {
                return;
            }
            let g = s.generation.load(Ordering::Acquire);
            if g != seen {
                break g;
            }
            if t0.elapsed() < SPIN {
                std::hint::spin_loop();
            } else {
                std::thread::park_timeout(Duration::from_millis(5));
            }
        };
        seen = g;
        // a job with nothing to compute is finished by `submit` itself, and its fields may already be the next job's
        if s.finished.load(Ordering::Acquire) >= g {
            continue;
        }
        // workers write disjoint parts of the job: act rows in phase A, out rows in phase B
        let jp = s.job.get();
        let job = unsafe { &*jp };
        let n = job.n;
        // phase A: act[k][j] = silu(G_j . xn) * (U_j . xn), rows split across threads
        let rows = n * INTER;
        let per = rows.div_ceil(s.threads);
        for r in (t * per)..((t + 1) * per).min(rows) {
            let (k, j) = (r / INTER, r % INTER);
            let e = job.experts[k].0;
            let (gq, gsm) = expert(job.layer.gu_q, job.layer.gu_sm, e, 2 * INTER, HIDDEN);
            let g = dot_row::<HIDDEN>(gq, gsm, j, &job.xn);
            let u = dot_row::<HIDDEN>(gq, gsm, INTER + j, &job.xn);
            unsafe { (*jp).act[k][j] = g / (1.0 + (-g).exp()) * u };
        }
        if s.phase_a.fetch_add(1, Ordering::AcqRel) + 1 < s.threads {
            while s.phase_a.load(Ordering::Acquire) < s.threads {
                std::hint::spin_loop();
            }
        }
        // phase B: out[o] = sum_k w_k * (D_o . act_k), output rows split across threads
        let mut acts = [Quant::<INTER>::zero(); TOPK];
        for k in 0..n {
            acts[k].set(&job.act[k]);
        }
        let per = HIDDEN.div_ceil(s.threads);
        let out = unsafe { std::ptr::addr_of_mut!((*job.io).out) as *mut f32 };
        for o in (t * per)..((t + 1) * per).min(HIDDEN) {
            let mut acc = 0.0f32;
            for k in 0..n {
                let (e, w) = job.experts[k];
                let (dq, dsm) = expert(job.layer.d_q, job.layer.d_sm, e, HIDDEN, INTER);
                acc += w * dot_row::<INTER>(dq, dsm, o, &acts[k]);
            }
            unsafe { *out.add(o) = acc };
        }
        if s.phase_b.fetch_add(1, Ordering::AcqRel) + 1 == s.threads {
            s.finished.store(g, Ordering::Release);
        }
    }
}

/// Expert `e` of a bank with `n` rows of `k` inputs: (q, scale/min pairs).
fn expert(q: *const u8, sm: *const u16, e: usize, n: usize, k: usize) -> (*const u8, *const u16) {
    unsafe { (q.add(e * n * k / 2), sm.add(e * n * (k / 64) * 2)) }
}

fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Row `row` of an int4 matrix (low nibble = even input, group 64 with f16 scale and min) dotted with x.
fn dot_row<const K: usize>(q: *const u8, sm: *const u16, row: usize, x: &Quant<K>) -> f32 {
    let qr = unsafe { std::slice::from_raw_parts(q.add(row * K / 2), K / 2) };
    let sr = unsafe { std::slice::from_raw_parts(sm.add(row * (K / 64) * 2), (K / 64) * 2) };
    #[cfg(target_arch = "x86_64")]
    {
        if avx2() {
            return unsafe { dot_avx2::<K>(qr, sr, x) };
        }
    }
    dot_scalar::<K>(qr, sr, x)
}

fn dot_scalar<const K: usize>(qr: &[u8], sr: &[u16], x: &Quant<K>) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..K / 64 {
        let scale = half::f16::from_bits(sr[2 * g]).to_f32();
        let min = half::f16::from_bits(sr[2 * g + 1]).to_f32();
        let mut qx = 0.0f32;
        for b in 2 * g..2 * g + 2 {
            let qb = &qr[b * 16..b * 16 + 16];
            let mut isum = 0i32;
            for i in 0..16 {
                isum += (qb[i] & 15) as i32 * x.q[b][i] as i32 + (qb[i] >> 4) as i32 * x.q[b][16 + i] as i32;
            }
            qx += isum as f32 * x.d[b];
        }
        acc += scale * qx + min * x.gsum[g];
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const K: usize>(qr: &[u8], sr: &[u16], x: &Quant<K>) -> f32 {
    use std::arch::x86_64::*;
    let low = _mm_set1_epi8(0x0f);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut min_acc = 0.0f32;
    for g in 0..K / 64 {
        let scale = half::f16::from_bits(sr[2 * g]).to_f32();
        min_acc += half::f16::from_bits(sr[2 * g + 1]).to_f32() * x.gsum[g];
        for b in 2 * g..2 * g + 2 {
            let v = _mm_loadu_si128(qr.as_ptr().add(b * 16) as *const __m128i);
            let lo = _mm_and_si128(v, low);
            let hi = _mm_and_si128(_mm_srli_epi16(v, 4), low);
            let w = _mm256_set_m128i(hi, lo);
            let xq = _mm256_loadu_si256(x.q[b].as_ptr() as *const __m256i);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(w, xq), ones);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(scale * x.d[b]), acc);
        }
    }
    let h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    _mm_cvtss_f32(h) + min_acc
}
