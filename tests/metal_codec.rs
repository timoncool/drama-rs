//! Hand-checked CPU tests for the codec ops (feature `metal`): indexing,
//! history reads and ring-buffer behaviour — the things audio stats can't
//! pinpoint if the full run sounds wrong.
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

fn bf16_rn(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn f16_rn(v: f32) -> u16 {
    // test-side oracle (same rounding rule as the backend)
    let b = v.to_bits();
    let s = (b >> 16) & 0x8000;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7fffff;
    if e <= 0 {
        if e < -10 {
            return s as u16;
        }
        let m = m | 0x800000;
        let shift = (1 - e) as u32;
        let mut out = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rest > half || (rest == half && (out & 1) == 1) {
            out += 1;
        }
        return (s | out) as u16;
    }
    if e >= 31 {
        return if m == 0 { (s | 0x7c00) as u16 } else { (s | 0x7e00) as u16 };
    }
    let m = m + 0xfff + ((m >> 13) & 1);
    if m >= 0x800000 {
        return (s | ((e as u32 + 1) << 10)) as u16;
    }
    (s | ((e as u32) << 10) | (m >> 13)) as u16
}

#[test]
fn im2col_reads_history() {
    let s = st();
    // C=2, k=3, d=1, stride=1: out[t, j*2+c] = x[t-(2-j), c], history before x.
    let full = vec![10.0f32, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xp = alloc(full.len() * 4, s, Some(&bytes_of(&full)));
    let cur = unsafe { (xp as *const f32).add(4) }; // rows [1,2],[3,4],[5,6], history [10..13]
    let op = alloc(2 * 3 * 2 * 2, s, None);
    assert_eq!(unsafe { ffi::dr_im2col(cur, 2, 2, 3, 1, 1, std::ptr::null(), op as *mut u16, s) }, 0);
    let o: Vec<u16> = down(op, 2 * 3 * 2, s);
    // t=0 reads rows -2,-1,0 = [10,11],[12,13],[1,2]
    let want_f = vec![10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(o, want_f.iter().map(|&v| f16_rn(v)).collect::<Vec<_>>());
}

#[test]
fn tconv_gather_pairs_rows() {
    let s = st();
    // two=1: out[t] = [x[t] | x[t-1]], x[-1] must be readable.
    let full = vec![7.0f32, 8.0, 1.0, 2.0, 3.0, 4.0];
    let xp = alloc(full.len() * 4, s, Some(&bytes_of(&full)));
    let cur = unsafe { (xp as *const f32).add(2) };
    let op = alloc(2 * 4 * 2, s, None);
    assert_eq!(unsafe { ffi::dr_tconv_gather(cur, 2, 2, 1, std::ptr::null(), op as *mut u16, s) }, 0);
    let o: Vec<u16> = down(op, 8, s);
    let want = [1.0, 2.0, 7.0, 8.0, 3.0, 4.0, 1.0, 2.0];
    assert_eq!(o, want.iter().map(|&v| f16_rn(v)).collect::<Vec<_>>());
}

#[test]
fn epilogue_act_and_residual() {
    let s = st();
    let y = vec![0.0f32, 1.0, -1.0, 2.0];
    let bias = vec![0.5f32, -0.5];
    let scale = vec![2.0f32, 0.5];
    let res = vec![1.0f32, 1.0, 1.0, 1.0];
    let yp = alloc(16, s, Some(&bytes_of(&y)));
    let bp = alloc(8, s, Some(&bytes_of(&bias)));
    let sp = alloc(8, s, Some(&bytes_of(&scale)));
    let rp = alloc(16, s, Some(&bytes_of(&res)));
    let op = alloc(16, s, None);
    let o16 = alloc(8, s, None);
    assert_eq!(
        unsafe { ffi::dr_epilogue(yp as *const f32, 4, 2, bp as *const f32, sp as *const f32, rp as *const f32, 1, op as *mut f32, o16 as *mut u16, s) },
        0
    );
    let o: Vec<f32> = down(op, 4, s);
    let want: Vec<f32> = vec![(0.0 + 0.5f32).tanh() * 2.0 + 1.0, (1.0 - 0.5f32).tanh() * 0.5 + 1.0, (-1.0 + 0.5f32).tanh() * 2.0 + 1.0, (2.0 - 0.5f32).tanh() * 0.5 + 1.0];
    for (a, b) in o.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    let o16v: Vec<u16> = down(o16, 4, s);
    assert_eq!(o16v, want.iter().map(|&v| f16_rn(v)).collect::<Vec<_>>());
}

#[test]
fn vq_encode_picks_cosine_best() {
    let s = st();
    // D=4: r = [1,0,0,0]; z = in_w.r + b; codebook of 2 (L2-normalised).
    let r = vec![1.0f32, 0.0, 0.0, 0.0];
    let in_w = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let in_b = vec![0.0f32; 8];
    // code 0 direction ~ +x in z-space, code 1 ~ -x
    let cbn = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let table = vec![0.5f32, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0];
    let rp = alloc(16, s, Some(&bytes_of(&r)));
    let iwp = alloc(in_w.len() * 4, s, Some(&bytes_of(&in_w)));
    let ibp = alloc(32, s, Some(&bytes_of(&in_b)));
    let cbp = alloc(cbn.len() * 4, s, Some(&bytes_of(&cbn)));
    let tp = alloc(table.len() * 4, s, Some(&bytes_of(&table)));
    let cp = alloc(4, s, None);
    assert_eq!(
        unsafe { ffi::dr_vq_encode(rp as *mut f32, 1, 4, iwp as *const f32, ibp as *const f32, cbp as *const f32, 2, tp as *const f32, cp as *mut i32, 1, s) },
        0
    );
    let codes: Vec<i32> = down(cp, 1, s);
    assert_eq!(codes, vec![0]);
    let r2: Vec<f32> = down(rp, 4, s);
    assert_eq!(r2, vec![0.5, 0.0, 0.0, 0.0]); // residual minus table[0]
}

#[test]
fn codec_rope_and_attn_ring() {
    let s = st();
    // H=1, W=64. qkv rows: q = one-hot dim0, k = one-hot dim0, v = ramp.
    let h = 1usize;
    let w = 64;
    let inv: Vec<f32> = (0..32).map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / 64.0)).collect();
    let mut qkv = vec![0.0f32; 2 * 3 * w];
    qkv[0 * 3 * w + 0] = 1.0; // q row0 dim0
    qkv[0 * 3 * w + w + 0] = 1.0; // k row0 dim0
    for i in 0..w {
        qkv[0 * 3 * w + 2 * w + i] = i as f32; // v row0
        qkv[1 * 3 * w + 2 * w + i] = 100.0 + i as f32; // v row1
    }
    qkv[1 * 3 * w + 0] = 1.0;
    qkv[1 * 3 * w + w + 0] = 1.0;
    let qp = alloc(qkv.len() * 4, s, Some(&bytes_of(&qkv)));
    let ip = alloc(inv.len() * 4, s, Some(&bytes_of(&inv)));
    let qo = alloc(2 * w * 4, s, None);
    let kc = alloc(4 * w * 4, s, None);
    let vc = alloc(4 * w * 4, s, None);
    // t0=0, ring=4
    assert_eq!(unsafe { ffi::dr_codec_rope(qp as *const f32, 2, h as i32, ip as *const f32, 0, 4, qo as *mut f32, kc as *mut f32, vc as *mut f32, s) }, 0);
    let qv: Vec<f32> = down(qo, 2 * w, s);
    // rope at t=0 is identity (sin 0, cos 1)
    assert!((qv[0] - 1.0).abs() < 1e-5 && qv[1].abs() < 1e-5);
    // attention over window 2 at t0=1 sees rows 0..1; both keys one-hot dim0
    let ao = alloc(2 * w * 2, s, None);
    assert_eq!(unsafe { ffi::dr_codec_attn(qo as *const f32, kc as *const f32, vc as *const f32, 2, h as i32, 2, 0, 4, ao as *mut u16, s) }, 0);
    let _a: Vec<u16> = down(ao, 2 * w, s);
}

#[test]
fn dwconv_and_conv_out_histories() {
    let s = st();
    // dwconv k=2: out[t,c] = b + w0*x[t-1] + w1*x[t], history row before x.
    let full = vec![5.0f32, 6.0, 1.0, 2.0, 3.0, 4.0];
    let xp = alloc(full.len() * 4, s, Some(&bytes_of(&full)));
    let cur = unsafe { (xp as *const f32).add(2) };
    let w = vec![1.0f32, 1.0, 1.0, 1.0];
    let b = vec![0.0f32, 0.0];
    let wp = alloc(16, s, Some(&bytes_of(&w)));
    let bp = alloc(8, s, Some(&bytes_of(&b)));
    let op = alloc(2 * 2 * 4, s, None);
    assert_eq!(unsafe { ffi::dr_dwconv(cur, 2, 2, 2, wp as *const f32, bp as *const f32, op as *mut f32, s) }, 0);
    let o: Vec<f32> = down(op, 4, s);
    assert_eq!(o, vec![6.0, 8.0, 4.0, 6.0]); // [5+1, 6+2], [1+3, 2+4]
}
