//! GPU tests for the Metal kernels (feature `metal`): each MSL kernel is checked
//! against an independent CPU computation, including null-output variants that
//! the engine actually uses.
#![cfg(feature = "metal")]

use drama::ffi;
use std::ffi::c_void;

fn st() -> ffi::Stream {
    let mut s = std::ptr::null_mut();
    assert_eq!(unsafe { ffi::dr_stream_create(&mut s) }, 0);
    s
}

fn alloc(nbytes: usize, s: ffi::Stream, data: Option<&[u8]>) -> *mut c_void {
    let mut p = std::ptr::null_mut();
    assert_eq!(unsafe { ffi::dr_malloc(&mut p, nbytes.max(16)) }, 0);
    if let Some(d) = data {
        assert_eq!(unsafe { ffi::dr_h2d(p, d.as_ptr() as *const c_void, d.len(), s) }, 0);
    }
    p
}

fn down<T: Copy + Default>(p: *mut c_void, n: usize, s: ffi::Stream) -> Vec<T> {
    let mut v = vec![T::default(); n];
    assert_eq!(
        unsafe { ffi::dr_d2h(v.as_mut_ptr() as *mut c_void, p as *const c_void, n * std::mem::size_of::<T>(), s) },
        0
    );
    v
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)).to_vec() }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self) -> f32 {
        (self.next() % 2000) as f32 / 1000.0 - 1.0
    }
}

/// Independent round-to-nearest-even f32 -> bf16 (the spec both sides must meet).
fn bf16_rn(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[test]
fn rmsnorm_matches_cpu() {
    let s = st();
    let (rows, d) = (5usize, 300usize);
    let mut rng = Rng(0x1234);
    let x: Vec<f32> = (0..rows * d).map(|_| rng.f32() * 3.0).collect();
    let w: Vec<f32> = (0..d).map(|_| 0.5 + rng.f32().abs()).collect();
    let dp = alloc(rows * d * 4, s, Some(&bytes_of(&x)));
    let wp = alloc(d * 4, s, Some(&bytes_of(&w)));
    let yp = alloc(rows * d * 4, s, None);
    let bp = alloc(rows * d * 2, s, None);
    assert_eq!(unsafe { ffi::dr_rmsnorm(dp as *const f32, wp as *const f32, yp as *mut f32, bp as *mut u16, rows as i32, d as i32, 1e-5, s) }, 0);
    let y: Vec<f32> = down(yp, rows * d, s);
    let yb: Vec<u16> = down(bp, rows * d, s);
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let inv = (xr.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-5).sqrt().recip();
        for i in 0..d {
            let want = xr[i] * inv * w[i];
            assert!((y[r * d + i] - want).abs() < 1e-5, "row {r} col {i}");
            assert_eq!(yb[r * d + i], bf16_rn(want), "row {r} col {i}");
        }
    }
}

#[test]
fn rmsnorm_null_outputs() {
    let s = st();
    // Engine calls with y == null (prefill norms) and with yb == null (final norm).
    let (rows, d) = (2usize, 64usize);
    let x = vec![0.5f32; rows * d];
    let w = vec![2.0f32; d];
    let dp = alloc(rows * d * 4, s, Some(&bytes_of(&x)));
    let wp = alloc(d * 4, s, Some(&bytes_of(&w)));
    let bp = alloc(rows * d * 2, s, None);
    assert_eq!(
        unsafe { ffi::dr_rmsnorm(dp as *const f32, wp as *const f32, std::ptr::null_mut(), bp as *mut u16, rows as i32, d as i32, 1e-6, s) },
        0
    );
    let yb: Vec<u16> = down(bp, rows * d, s);
    let want = bf16_rn(0.5 * (0.25f32 + 1e-6).sqrt().recip() * 2.0);
    assert!(yb.iter().all(|&v| v == want));
    let yp = alloc(rows * d * 4, s, None);
    assert_eq!(
        unsafe { ffi::dr_rmsnorm(dp as *const f32, wp as *const f32, yp as *mut f32, std::ptr::null_mut(), rows as i32, d as i32, 1e-6, s) },
        0
    );
    let y: Vec<f32> = down(yp, rows * d, s);
    assert!(y.iter().all(|&v| (v - bf16_f32(want)).abs() < 1e-5));
}

#[test]
fn gemv_bf16_matches_cpu() {
    let s = st();
    let (n, k) = (130usize, 300usize);
    let mut rng = Rng(0xabcd);
    let w: Vec<u16> = (0..n * k).map(|_| bf16_rn(rng.f32())).collect();
    let x: Vec<f32> = (0..k).map(|_| rng.f32()).collect();
    let wp = alloc(n * k * 2, s, Some(&bytes_of(&w)));
    let xp = alloc(k * 4, s, Some(&bytes_of(&x)));
    let yp = alloc(n * 4, s, None);
    assert_eq!(
        unsafe { ffi::dr_gemv_bf16(wp as *const u16, xp as *const f32, yp as *mut f32, n as i32, k as i32, std::ptr::null(), 0.0, std::ptr::null(), 0, s) },
        0
    );
    let y: Vec<f32> = down(yp, n, s);
    for r in 0..n {
        let want: f32 = (0..k).map(|j| bf16_f32(w[r * k + j]) * x[j]).sum();
        assert!((y[r] - want).abs() < 1e-3, "row {r}: {} vs {want}", y[r]);
    }
}

#[test]
fn gemv_bf16_fused_norm() {
    let s = st();
    let (n, k) = (17usize, 256usize);
    let mut rng = Rng(42);
    let w: Vec<u16> = (0..n * k).map(|_| bf16_rn(rng.f32())).collect();
    let x: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0).collect();
    let nw: Vec<f32> = (0..k).map(|_| 0.5 + rng.f32().abs()).collect();
    let wp = alloc(n * k * 2, s, Some(&bytes_of(&w)));
    let xp = alloc(k * 4, s, Some(&bytes_of(&x)));
    let np = alloc(k * 4, s, Some(&bytes_of(&nw)));
    let yp = alloc(n * 4, s, None);
    assert_eq!(
        unsafe { ffi::dr_gemv_bf16(wp as *const u16, xp as *const f32, yp as *mut f32, n as i32, k as i32, np as *const f32, 1e-5, std::ptr::null(), 0, s) },
        0
    );
    let y: Vec<f32> = down(yp, n, s);
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / k as f32 + 1e-5).sqrt().recip();
    let xn: Vec<f32> = x.iter().zip(nw.iter()).map(|(a, b)| a * inv * b).collect();
    for r in 0..n {
        let want: f32 = (0..k).map(|j| bf16_f32(w[r * k + j]) * xn[j]).sum();
        assert!((y[r] - want).abs() < 1e-3, "row {r}");
    }
}

#[test]
fn dequant_q8_matches_cpu() {
    let s = st();
    let (n, k) = (3usize, 256usize);
    let mut rng = Rng(7);
    let q: Vec<i8> = (0..n * k).map(|_| (rng.next() % 256) as u8 as i8).collect();
    // Scales with exact f16 values (small integers).
    let sc: Vec<u16> = (0..n * (k / 128)).map(|i| f32_to_f16(0.25 * (1 + (i % 5)) as f32)).collect();
    let qp = alloc(n * k, s, Some(&bytes_of(&q)));
    let sp = alloc(sc.len() * 2, s, Some(&bytes_of(&sc)));
    let op = alloc(n * k * 2, s, None);
    assert_eq!(unsafe { ffi::dr_dequant_q8(qp as *const i8, sp as *const u16, op as *mut u16, n as i32, k as i32, s) }, 0);
    let o: Vec<u16> = down(op, n * k, s);
    for r in 0..n * k {
        let row = r / k;
        let want = bf16_rn(q[r] as f32 * f16_to_f32(sc[row * (k / 128) + (r % k) / 128]));
        assert_eq!(o[r], want, "elem {r}");
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = (b >> 13) & 0x3ff;
    if e <= 0 {
        return s as u16; // flush subnormals (test only uses normal values)
    }
    if e >= 31 {
        return (s | 0x7c00) as u16;
    }
    (s | ((e as u32) << 10) | m) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let s = (h & 0x8000) as u32;
    let e = (h & 0x7c00) as u32;
    let m = (h & 0x03ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 16
        } else {
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

#[test]
fn f32_to_bf16_edge_cases() {
    let s = st();
    let x = vec![0.0f32, -0.0, 1.0, -1.0, 3.14159, 1e-8, 1e8, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.49999, 123.456];
    let xp = alloc(x.len() * 4, s, Some(&bytes_of(&x)));
    let yp = alloc(x.len() * 2, s, None);
    assert_eq!(unsafe { ffi::dr_f32_to_bf16(xp as *const f32, yp as *mut u16, x.len(), s) }, 0);
    let y: Vec<u16> = down(yp, x.len(), s);
    for (i, &v) in x.iter().enumerate() {
        assert_eq!(y[i], bf16_rn(v), "elem {i} ({v})");
    }
}

#[test]
fn swiglu_matches_cpu() {
    let s = st();
    let (rows, ii) = (3usize, 70usize);
    let mut rng = Rng(99);
    let h: Vec<f32> = (0..rows * 2 * ii).map(|_| rng.f32() * 2.0).collect();
    let hp = alloc(h.len() * 4, s, Some(&bytes_of(&h)));
    let yp = alloc(rows * ii * 2, s, None);
    assert_eq!(unsafe { ffi::dr_swiglu_rows(hp as *const f32, yp as *mut u16, rows as i32, ii as i32, s) }, 0);
    let y: Vec<u16> = down(yp, rows * ii, s);
    for r in 0..rows {
        for j in 0..ii {
            let g = h[r * 2 * ii + j];
            let u = h[r * 2 * ii + ii + j];
            assert_eq!(y[r * ii + j], bf16_rn(g / (1.0 + (-g).exp()) * u), "({r}, {j})");
        }
    }
}

#[test]
fn graph_capture_replays() {
    let s = st();
    // Record rmsnorm + gemv into a graph, replay twice with changed inputs.
    let (n, k) = (4usize, 64usize);
    let mut rng = Rng(5);
    let w: Vec<u16> = (0..n * k).map(|_| bf16_rn(rng.f32())).collect();
    let wp = alloc(n * k * 2, s, Some(&bytes_of(&w)));
    let xp = alloc(k * 4, s, None);
    let yp = alloc(n * 4, s, None);
    let mut exec = std::ptr::null_mut();
    assert_eq!(unsafe { ffi::dr_graph_begin(s) }, 0);
    assert_eq!(
        unsafe { ffi::dr_gemv_bf16(wp as *const u16, xp as *const f32, yp as *mut f32, n as i32, k as i32, std::ptr::null(), 0.0, std::ptr::null(), 0, s) },
        0
    );
    assert_eq!(unsafe { ffi::dr_graph_end(s, &mut exec) }, 0);
    assert!(!exec.is_null());
    for round in 0..2 {
        let x: Vec<f32> = (0..k).map(|_| rng.f32() + round as f32).collect();
        assert_eq!(unsafe { ffi::dr_h2d(xp, x.as_ptr() as *const c_void, x.len() * 4, s) }, 0);
        assert_eq!(unsafe { ffi::dr_graph_launch(exec, s) }, 0);
        let y: Vec<f32> = down(yp, n, s);
        for r in 0..n {
            let want: f32 = (0..k).map(|j| bf16_f32(w[r * k + j]) * x[j]).sum();
            assert!((y[r] - want).abs() < 1e-3, "round {round} row {r}");
        }
    }
    assert_eq!(unsafe { ffi::dr_graph_destroy(exec) }, 0);
}

fn f16_bits_test(v: f32) -> u16 {
    // round-to-nearest f16 for the test oracle (finite values only)
    let b = v.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7fffff;
    if e <= 0 {
        return s as u16;
    }
    if e >= 31 {
        return (s | 0x7c00) as u16;
    }
    let m = m + 0xfff + ((m >> 13) & 1);
    if m >= 0x800000 {
        return (s | ((e as u32 + 1) << 10)) as u16;
    }
    (s | ((e as u32) << 10) | (m >> 13)) as u16
}

fn f16f_test(h: u16) -> f32 {
    let s = (h & 0x8000) as u32;
    let e = (h & 0x7c00) as u32;
    let m = (h & 0x03ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 16
        } else {
            (s << 16) | 0x38000000
        }
    } else if e == 0x7c00 {
        (s << 16) | 0x7f800000
    } else {
        (s << 16) | (((e >> 10) + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn gemv_q8_case(n: usize, k: usize, norm: bool, with_res: bool) {
    let s = st();
    let mut rng = Rng(0xbeef + n as u64 * 100003 + k as u64);
    // weights incl. negatives (catches char-signedness bugs), scales positive
    let q: Vec<i8> = (0..n * k).map(|_| (rng.next() % 256) as u8 as i8).collect();
    let sc: Vec<u16> = (0..n * (k / 128)).map(|i| f16_bits_test(0.02 * (1.0 + (i % 5) as f32 * 0.1))).collect();
    let x: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0).collect();
    let nw: Vec<f32> = (0..k).map(|_| 0.5 + rng.f32().abs()).collect();
    let res: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
    let qp = alloc(n * k, s, Some(&bytes_of(&q)));
    let sp = alloc(sc.len() * 2, s, Some(&bytes_of(&sc)));
    let xp = alloc(k * 4, s, Some(&bytes_of(&x)));
    let np = alloc(k * 4, s, Some(&bytes_of(&nw)));
    let rp = alloc(n * 4, s, Some(&bytes_of(&res)));
    let yp = alloc(n * 4, s, None);
    assert_eq!(
        unsafe {
            ffi::dr_gemv_q8(
                qp as *const i8, sp as *const u16, xp as *const f32,
                if with_res { rp as *const f32 } else { std::ptr::null() },
                yp as *mut f32, n as i32, k as i32,
                if norm { np as *const f32 } else { std::ptr::null() }, 1e-5,
                std::ptr::null(), 0, s,
            )
        },
        0
    );
    let y: Vec<f32> = down(yp, n, s);
    let inv = if norm {
        // matches stage_x_norm: norm over raw x, then fuse the weight
        (x.iter().map(|v| v * v).sum::<f32>() / k as f32 + 1e-5).sqrt().recip()
    } else {
        1.0
    };
    let xn: Vec<f32> = if norm {
        x.iter().zip(nw.iter()).map(|(a, b)| a * inv * b).collect()
    } else {
        x.clone()
    };
    for r in 0..n {
        let mut want = 0.0f32;
        for j in 0..k {
            want += q[r * k + j] as f32 * f16f_test(sc[r * (k / 128) + j / 128]) * xn[j];
        }
        if with_res {
            want += res[r];
        }
        assert!((y[r] - want).abs() < 1e-2 * want.abs().max(1.0), "row {r} n={n} k={k}: {} vs {want}", y[r]);
    }
}

#[test]
fn gemv_q8_matches_cpu() {
    gemv_q8_case(130, 2048, false, false);
    gemv_q8_case(65, 2048, true, true);
    gemv_q8_case(17, 512, true, false);
    gemv_q8_case(9, 4096, false, true);
}

fn dot_q4_test(q: &[u8], sm: &[u16], k: usize, row: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for j in 0..k {
        let qq = ((q[row * (k / 2) + j / 2] >> ((j & 1) * 4)) & 15) as f32;
        let g = j / 64;
        acc += (qq * f16_test(sm[row * (k / 64) * 2 + g * 2]) + f16_test(sm[row * (k / 64) * 2 + g * 2 + 1])) * x[j];
    }
    acc
}

fn dot_q8_test(q: &[i8], s: &[u16], k: usize, row: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for j in 0..k {
        acc += q[row * k + j] as f32 * f16_test(s[row * (k / 128) + j / 128]) * x[j];
    }
    acc
}

fn f16_test(h: u16) -> f32 {
    let s = (h & 0x8000) as u32;
    let e = (h & 0x7c00) as u32;
    let m = (h & 0x03ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 16
        } else {
            (s << 16) | 0x38000000
        }
    } else if e == 0x7c00 {
        (s << 16) | 0x7f800000
    } else {
        (s << 16) | (((e >> 10) + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[test]
fn moe_decode_matches_cpu() {
    let s = st();
    // real engine dims, 4 experts (ids 0..3)
    const H: usize = 2048;
    const MI: usize = 512;
    const TOPK: usize = 8;
    const E: usize = 4;
    let mut rng = Rng(0x1234abcd);
    let mut rbytes = |n: usize| -> Vec<u8> { (0..n).map(|_| (rng.next() % 256) as u8).collect() };
    // gate_up q4 [E, 2*MI, K/2], K=H=2048
    let gq = rbytes(E * 2 * MI * 1024);
    let gs: Vec<u16> = (0..E * 2 * MI * 32 * 2).map(|i| f16_bits_test(0.03 * (1.0 + (i % 5) as f32 * 0.1))).collect();
    let dq = rbytes(E * H * 256);
    let ds: Vec<u16> = (0..E * H * 8 * 2).map(|i| f16_bits_test(0.03 * (1.0 + (i % 5) as f32 * 0.1))).collect();
    // shared int8: [2*MI, K] and [H, MI]
    let sgq: Vec<i8> = (0..2 * MI * H).map(|_| (rng.next() % 256) as u8 as i8).collect();
    let sgs: Vec<u16> = (0..2 * MI * 16).map(|i| f16_bits_test(0.02 * (1.0 + (i % 5) as f32 * 0.1))).collect();
    let sdq: Vec<i8> = (0..H * MI).map(|_| (rng.next() % 256) as u8 as i8).collect();
    let sds: Vec<u16> = (0..H * 4).map(|i| f16_bits_test(0.02 * (1.0 + (i % 5) as f32 * 0.1))).collect();
    let x: Vec<f32> = (0..H).map(|_| rng.f32()).collect();
    let nw: Vec<f32> = (0..H).map(|_| 0.5 + rng.f32().abs()).collect();
    let idx: Vec<i32> = vec![3, 0, 2, 1, 0, 2, 1, 3];
    let wts: Vec<f32> = vec![0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1];
    let sgate = vec![0.3f32];
    let res: Vec<f32> = (0..H).map(|_| rng.f32() * 0.5).collect();
    let up = |d: &[u8]| alloc(d.len(), s, Some(d));
    let gqp = up(&gq);
    let gsp = up(&bytes_of(&gs));
    let dqp = up(&dq);
    let dsp = up(&bytes_of(&ds));
    let sgqp = up(&bytes_of(&sgq));
    let sgsp = up(&bytes_of(&sgs));
    let sdqp = up(&bytes_of(&sdq));
    let sdsp = up(&bytes_of(&sds));
    let xp = up(&bytes_of(&x));
    let nwp = up(&bytes_of(&nw));
    let idxp = up(&bytes_of(&idx));
    let wtsp = up(&bytes_of(&wts));
    let sgp = up(&bytes_of(&sgate));
    let resp = up(&bytes_of(&res));
    let actp = up(&vec![0u8; (TOPK + 1) * MI * 4]);
    let yp = up(&vec![0u8; H * 4]);
    assert_eq!(
        unsafe {
            ffi::dr_moe_decode(
                gqp as *const u8, gsp as *const u16, dqp as *const u8, dsp as *const u16,
                sgqp as *const i8, sgsp as *const u16, sdqp as *const i8, sdsp as *const u16,
                xp as *const f32, idxp as *const i32, wtsp as *const f32, sgp as *const f32,
                resp as *const f32, actp as *mut f32, yp as *mut f32,
                H as i32, MI as i32, TOPK as i32, nwp as *const f32, 1e-5, std::ptr::null(), 0, s,
            )
        },
        0
    );
    let y: Vec<f32> = down(yp, H, s);
    // CPU oracle
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / H as f32 + 1e-5).sqrt().recip();
    let xn: Vec<f32> = x.iter().zip(nw.iter()).map(|(a, b)| a * inv * b).collect();
    let q4 = |q: &[u8], sm: &[u16], k: usize, row: usize| dot_q4_test(q, sm, k, row, &xn);
    let q8 = |q: &[i8], sm: &[u16], k: usize, row: usize| dot_q8_test(q, sm, k, row, &xn);
    let mut act = vec![0.0f32; (TOPK + 1) * MI];
    for kk in 0..TOPK {
        let e = idx[kk] as usize;
        for j in 0..MI {
            let g = q4(&gq, &gs, H, e * 2 * MI + j);
            let u = q4(&gq, &gs, H, e * 2 * MI + MI + j);
            act[kk * MI + j] = g / (1.0 + (-g).exp()) * u;
        }
    }
    for j in 0..MI {
        let g = q8(&sgq, &sgs, H, j);
        let u = q8(&sgq, &sgs, H, MI + j);
        act[TOPK * MI + j] = g / (1.0 + (-g).exp()) * u;
    }
    let sg = 1.0 / (1.0 + (-sgate[0]).exp());
    let mut want = vec![0.0f32; H];
    for n in 0..H {
        let mut acc = 0.0f32;
        for kk in 0..TOPK {
            let e = idx[kk] as usize;
            let mut qx = 0.0f32;
            for j in 0..MI {
                let qb = dq[(e * H + n) * (MI / 2) + j / 2];
                let qq = ((qb >> ((j & 1) * 4)) & 15) as f32;
                let g = j / 64;
                qx += (qq * f16_test(ds[(e * H + n) * (MI / 64) * 2 + g * 2]) + f16_test(ds[(e * H + n) * (MI / 64) * 2 + g * 2 + 1])) * act[kk * MI + j];
            }
            acc += wts[kk] * qx;
        }
        let mut sh = 0.0f32;
        for j in 0..MI {
            sh += sdq[n * MI + j] as f32 * f16_test(sds[n * (MI / 128) + j / 128]) * act[TOPK * MI + j];
        }
        want[n] = res[n] + sg * sh + acc;
    }
    for n in 0..H {
        assert!((y[n] - want[n]).abs() < 1e-1 * want[n].abs().max(1.0), "row {n}: {} vs {}", y[n], want[n]);
    }
}

fn bf16_test(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return 0x7fc0;
    }
    (((b + 0x7fff + ((b >> 16) & 1)) >> 16)) as u16
}

#[test]
fn gemm_matches_cpu() {
    let s = st();
    let mut rng = Rng(0x6e6d);
    for (m, n, k) in [(2usize, 3, 4), (17, 65, 129), (64, 128, 256)] {
        let a: Vec<u16> = (0..m * k).map(|_| bf16_test(rng.f32())).collect();
        let w: Vec<u16> = (0..n * k).map(|_| bf16_test(rng.f32())).collect();
        let ap = alloc(a.len() * 2, s, Some(&bytes_of(&a)));
        let wp = alloc(w.len() * 2, s, Some(&bytes_of(&w)));
        let cp = alloc(m * n * 4, s, None);
        assert_eq!(unsafe { ffi::dr_gemm(std::ptr::null_mut(), ap as *const u16, wp as *const u16, cp as *mut f32, m as i32, n as i32, k as i32, 0.0) }, 0);
        let c: Vec<f32> = down(cp, m * n, s);
        for r in 0..m {
            for cc in 0..n {
                let mut wantb = 0.0f32;
                for j in 0..k {
                    wantb += bf16_test_f32(a[r * k + j]) * bf16_test_f32(w[cc * k + j]);
                }
                assert!((c[r * n + cc] - wantb).abs() < 1e-3 * wantb.abs().max(1.0), "m={m} n={n} k={k} ({r},{cc}): {} vs {wantb}", c[r * n + cc]);
            }
        }
    }
}

fn bf16_test_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[test]
fn moe_prefill_matches_cpu() {
    let s = st();
    const H: usize = 2048;
    let e: usize = std::env::var("MOE_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    const MI: usize = 512;
    
    let mut rng = Rng(0x9f17);
    let rbytes = |n: usize| -> Vec<u8> { let mut r = Rng(0xbeef); (0..n).map(|_| (r.next() % 256) as u8).collect() };
    let gq = rbytes(e * 2 * MI * 1024);
    let gs: Vec<u16> = (0..e * 2 * MI * 32 * 2).map(|i| f16_bits_test(0.03 * (1.0 + (i % 5) as f32 * 0.1))).collect();
    let dq = rbytes(e * H * 256);
    let ds: Vec<u16> = (0..e * H * 8 * 2).map(|i| f16_bits_test(0.03 * (1.0 + (i % 5) as f32 * 0.1))).collect();
    let total: usize = std::env::var("MOE_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let se0: Vec<usize> = (0..total).map(|s| s * e / total.max(1)).collect();
    let counts: Vec<i32> = { let mut c = vec![0i32; e]; for &sx in &se0 { c[sx] += 1; } c };
    let xp: Vec<u16> = (0..total * H).map(|_| bf16_test(rng.f32())).collect();
    let offs: Vec<i32> = { let mut o = vec![0i32]; for &c in &counts { o.push(o.last().unwrap() + c); } o };
    let up = |d: &[u8]| alloc(d.len(), s, Some(d));
    let gqp = up(&gq);
    let gsp = up(&bytes_of(&gs));
    let dqp = up(&dq);
    let dsp = up(&bytes_of(&ds));
    let xpp = up(&bytes_of(&xp));
    let offp = up(&bytes_of(&offs));
    let cntp = up(&bytes_of(&counts));
    let actp = up(&vec![0u8; total * MI * 2]);
    let yp = up(&vec![0u8; total * H * 4]);
    assert_eq!(
        unsafe {
            ffi::dr_moe_prefill(
                gqp as *const u8, gsp as *const u16, dqp as *const u8, dsp as *const u16,
                xpp as *const u16, offp as *const i32, cntp as *const i32,
                actp as *mut u16, yp as *mut f32, e as i32, H as i32, MI as i32, s,
            )
        },
        0
    );
    let y: Vec<f32> = down(yp, total * H, s);
    let actb: Vec<u8> = down(actp, total * MI * 2, s);
    let act: Vec<u16> = actb.chunks(2).map(|c| u16::from_ne_bytes([c[0], c[1]])).collect();
    // CPU oracle
    let se: Vec<usize> = se0;
    let bf = |h: u16| f32::from_bits((h as u32) << 16);
    let mut want_a = vec![0u16; total * MI];
    let mut want_y = vec![0f32; total * H];
    for sl in 0..total {
        let ex = se[sl];
        let xr: Vec<f32> = (0..H).map(|t| bf(xp[sl * H + t])).collect();
        for j in 0..MI {
            let g = dot_q4_test(&gq, &gs, H, ex * 2 * MI + j, &xr);
            let u = dot_q4_test(&gq, &gs, H, ex * 2 * MI + MI + j, &xr);
            let v = g / (1.0 + (-g).exp()) * u;
            want_a[sl * MI + j] = bf16_test(v);
        }
        let ar: Vec<f32> = (0..MI).map(|j| bf(want_a[sl * MI + j])).collect();
        for n in 0..H {
            want_y[sl * H + n] = dot_q4_test(&dq, &ds, MI, ex * H + n, &ar);
        }
    }
    for sl in 0..total {
        for j in 0..MI {
            let got = bf(act[sl * MI + j]);
            let want = bf(want_a[sl * MI + j]);
            assert!((got - want).abs() < 0.02 * want.abs().max(1.0), "act ({sl},{j}): {got} vs {want}");
        }
        for n in (0..H).step_by(7) {
            assert!((y[sl * H + n] - want_y[sl * H + n]).abs() < 0.02 * want_y[sl * H + n].abs().max(1.0), "y ({sl},{n})");
        }
    }
}

#[test]
fn sgemm_determinism() {
    let s = st();
    let mut rng = Rng(0xde7);
    let (m, n, k) = (16usize, 1024, 3072);
    let a: Vec<u16> = (0..m * k).map(|_| bf16_test(rng.f32())).collect();
    let w: Vec<u16> = (0..n * k).map(|_| bf16_test(rng.f32())).collect();
    let ap = alloc(a.len() * 2, s, Some(&bytes_of(&a)));
    let wp = alloc(w.len() * 2, s, Some(&bytes_of(&w)));
    let cp = alloc(m * n * 4, s, None);
    let mut first: Vec<f32> = Vec::new();
    for it in 0..500 {
        assert_eq!(unsafe { ffi::dr_gemm_f16(std::ptr::null_mut(), ap as *const u16, wp as *const u16, cp as *mut f32, m as i32, n as i32, k as i32, 0.0) }, 0);
        let c: Vec<f32> = down(cp, m * n, s);
        assert!(c.iter().all(|v| v.is_finite()), "iter {it}: non-finite");
        if it == 0 {
            first = c;
        } else {
            assert!(c == first, "iter {it}: nondeterministic sgemm!");
        }
    }
}

#[test]
fn attn_decode_matches_cpu() {
    let s = st();
    const H: usize = 2;
    const HKV: usize = 1;
    const HD: usize = 256;
    const NK: usize = 17;
    let mut rng = Rng(0xa77e);
    let q: Vec<f32> = (0..H * HD).map(|_| rng.f32()).collect();
    let kc: Vec<u16> = (0..NK * HKV * HD).map(|_| bf16_test(rng.f32())).collect();
    let vc: Vec<u16> = (0..NK * HKV * HD).map(|_| bf16_test(rng.f32())).collect();
    let gate: Vec<f32> = (0..H * HD).map(|_| rng.f32()).collect();
    let pos = vec![(NK - 1) as i32];
    let up = |d: &[u8]| alloc(d.len(), s, Some(d));
    let qp = up(&bytes_of(&q));
    let kcp = up(&bytes_of(&kc));
    let vcp = up(&bytes_of(&vc));
    let gp = up(&bytes_of(&gate));
    let pp = up(&bytes_of(&pos));
    let op = up(&vec![0u8; H * HD * 4]);
    let partp = up(&vec![0u8; 64]);
    assert_eq!(
        unsafe {
            ffi::dr_attn_decode(
                qp as *const f32, kcp as *const u16, vcp as *const u16, gp as *const f32,
                H as i32, HKV as i32, HD as i32, pp as *const i32, 0, partp as *mut f32, op as *mut f32, s,
            )
        },
        0
    );
    let o: Vec<f32> = down(op, H * HD, s);
    // CPU oracle (mirrors attn_rows, decode row).
    let scale = (HD as f32).sqrt().recip();
    let bf = |h: u16| bf16_test_f32(h);
    let mut want = vec![0.0f32; H * HD];
    for hh in 0..H {
        let kh = hh / (H / HKV);
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = vec![0.0f32; HD];
        for j in 0..NK {
            let mut sc = 0.0f32;
            for d in 0..HD {
                sc += q[hh * HD + d] * bf(kc[(j * HKV + kh) * HD + d]);
            }
            sc *= scale;
            let mn = m.max(sc);
            let corr = (m - mn).exp();
            let p = (sc - mn).exp();
            l = l * corr + p;
            for d in 0..HD {
                acc[d] = acc[d] * corr + p * bf(vc[(j * HKV + kh) * HD + d]);
            }
            m = mn;
        }
        for d in 0..HD {
            let gv = gate[hh * HD + d];
            want[hh * HD + d] = acc[d] / l * (1.0 / (1.0 + (-gv).exp()));
        }
    }
    for i in 0..H * HD {
        assert!((o[i] - want[i]).abs() < 0.03 * want[i].abs().max(1.0), "out[{i}]: {} vs {}", o[i], want[i]);
    }
}

fn silu_cpu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn gdn_matches_cpu() {
    let s = st();
    const DK: usize = 128;
    let nv: usize = std::env::var("GDN_NV").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let nk: usize = if nv == 32 { 16 } else { 2 };
    let nn: usize = std::env::var("GDN_N").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let cs = 2 * nk * DK + nv * DK;
    let ps = nv * DK + 512;
    let (z_off, b_off, a_off) = (0usize, nv * DK, nv * DK + 64);
    let mut rng = Rng(0x6d6e);
    let conv: Vec<f32> = (0..nn * cs).map(|_| rng.f32() * 0.5).collect();
    let proj: Vec<f32> = (0..nn * ps).map(|_| rng.f32() * 0.5).collect();
    let alog: Vec<f32> = (0..nv).map(|_| rng.f32().abs() * 2.0).collect(); // realistic: decay << 1
    let dtb: Vec<f32> = (0..nv).map(|_| rng.f32() * 0.2).collect();
    let nw: Vec<f32> = (0..DK).map(|_| 0.8 + rng.f32().abs() * 0.4).collect();
    let state: Vec<f32> = (0..nv * DK * DK).map(|_| rng.f32() * 0.1).collect();
    let up = |d: &[u8]| alloc(d.len(), s, Some(d));
    let cvp = up(&bytes_of(&conv));
    let prp = up(&bytes_of(&proj));
    let alp = up(&bytes_of(&alog));
    let dtp = up(&bytes_of(&dtb));
    let nwp = up(&bytes_of(&nw));
    let stp = up(&bytes_of(&state));
    let op = up(&vec![0u8; nn * nv * DK * 4]);
    assert_eq!(
        unsafe {
            ffi::dr_gdn(
                cvp as *const f32, prp as *const f32, ps as i32, z_off as i32, b_off as i32, a_off as i32,
                alp as *const f32, dtp as *const f32, nwp as *const f32, stp as *mut f32, op as *mut f32,
                nn as i32, nk as i32, nv as i32, DK as i32, DK as i32, 1e-5, s,
            )
        },
        0
    );
    let o: Vec<f32> = down(op, nn * nv * DK, s);
    // CPU oracle (mirrors the old dr_gdn math).
    let qscale = (DK as f32).sqrt().recip();
    let mut stt_oracle = state.clone();
    let mut want = vec![0.0f32; nn * nv * DK];
    for hh in 0..nv {
        let kh = hh / (nv / nk);
        let a = -alog[hh].exp(); // A = -exp(A_log), decay = exp(A * softplus(.)) < 1
        for tt in 0..nn {
            let cr = &conv[tt * cs..(tt + 1) * cs];
            let pr = &proj[tt * ps..(tt + 1) * ps];
            let qv: Vec<f32> = (0..DK).map(|i| cr[kh * DK + i]).collect();
            let kv: Vec<f32> = (0..DK).map(|i| cr[nk * DK + kh * DK + i]).collect();
            let qi = qscale / (qv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            let ki = 1.0 / (kv.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            let qs: Vec<f32> = qv.iter().map(|x| x * qi).collect();
            let ks: Vec<f32> = kv.iter().map(|x| x * ki).collect();
            let vv: Vec<f32> = (0..DK).map(|v| cr[2 * nk * DK + hh * DK + v]).collect();
            let decay = (a * (pr[a_off + hh] + dtb[hh]).exp().ln_1p()).exp();
            let beta = 1.0 / (1.0 + (-pr[b_off + hh]).exp());
            let mut outv = vec![0.0f32; DK];
            for v in 0..DK {
                let mut kvm = 0.0f32;
                for k in 0..DK {
                    stt_oracle[hh * DK * DK + k * DK + v] *= decay;
                    kvm += stt_oracle[hh * DK * DK + k * DK + v] * ks[k];
                }
                let delta = (vv[v] - kvm) * beta;
                let mut oacc = 0.0f32;
                for k in 0..DK {
                    stt_oracle[hh * DK * DK + k * DK + v] += ks[k] * delta;
                    oacc += stt_oracle[hh * DK * DK + k * DK + v] * qs[k];
                }
                outv[v] = oacc;
            }
            let inv = 1.0 / ((outv.iter().map(|x| x * x).sum::<f32>() / DK as f32 + 1e-5).sqrt());
            for v in 0..DK {
                let zr = pr[z_off + hh * DK + v];
                want[tt * nv * DK + hh * DK + v] = outv[v] * inv * nw[v] * silu_cpu(zr);
            }
        }
    }
    for i in 0..nn * nv * DK {
        assert!((o[i] - want[i]).abs() < 0.05 * want[i].abs().max(1.0), "out[{i}]: {} vs {}", o[i], want[i]);
    }
    let stt: Vec<f32> = down(stp, nv * DK * DK, s);
    let stt_want = |i: usize| stt_oracle[i];
    for i in 0..nv * DK * DK {
        assert!((stt[i] - stt_want(i)).abs() < 0.05 * stt_want(i).abs().max(1.0), "state[{i}]: {} vs {}", stt[i], stt_want(i));
    }
}

#[test]
fn f32_to_f16_edge_cases() {
    // 65504 = f16 max, 65520 rounds up to inf, 70000.0 must be inf (not NaN).
    for (v, want) in [
        (65504.0f32, 0x7bffu16),
        (65520.0f32, 0x7c00u16),
        (70000.0f32, 0x7c00u16),
        (-70000.0f32, 0xfc00u16),
        (f32::INFINITY, 0x7c00u16),
        (f32::NEG_INFINITY, 0xfc00u16),
        (0.0f32, 0x0000u16),
        (1.0f32, 0x3c00u16),
    ] {
        assert_eq!(ffi::f32_to_f16_rn(v), want, "value {v}");
    }
    assert!(half::f16::from_bits(ffi::f32_to_f16_rn(f32::NAN)).is_nan());
}
