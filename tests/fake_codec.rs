//! Synthetic codec weights (feature `metal`): real shapes/dtypes, random values.
//! Runs Codec::load -> stream -> decode_chunk, validating the whole decode
//! stack and the tensor-name inventory before the real weights arrive.
#![cfg(feature = "metal")]

use std::path::PathBuf;
use drama::codec::Codec;
use safetensors::tensor::TensorView;
use safetensors::Dtype;

const D: usize = 1024;
const FFN: usize = 3072;
const SEM: usize = 4096;
const RES: usize = 1024;
const NCB: usize = 10;
const DEC_IN: usize = 1536;

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
        (self.next() % 1000) as f32 / 1000.0 - 0.5
    }
}

fn f16_bits(v: f32) -> u16 {
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

struct File {
    tensors: Vec<(String, TensorView<'static>)>,
}

impl File {
    fn add_f32(&mut self, rng: &mut Rng, name: &str, shape: Vec<usize>, scale: f32) {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.f32() * scale).collect();
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4).to_vec() };
        std::mem::forget(v);
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        self.tensors.push((name.to_string(), TensorView::new(Dtype::F32, shape, leaked).unwrap()));
    }
    fn add_f16(&mut self, rng: &mut Rng, name: &str, shape: Vec<usize>, scale: f32) {
        let n: usize = shape.iter().product();
        let v: Vec<u16> = (0..n).map(|_| f16_bits(rng.f32() * scale)).collect();
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 2).to_vec() };
        std::mem::forget(v);
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        self.tensors.push((name.to_string(), TensorView::new(Dtype::F16, shape, leaked).unwrap()));
    }
}

fn build_codec(path: &std::path::Path) {
    let mut rng = Rng(0xc0dec);
    let mut f = File { tensors: Vec::new() };
    f.add_f32(&mut rng, "rvq.table", vec![SEM + 9 * RES, D], 0.01);
    for l in 0..8 {
        let p = format!("post.l{l}.");
        f.add_f32(&mut rng, &(p.clone() + "attn_norm"), vec![D], 1.0);
        f.add_f16(&mut rng, &(p.clone() + "wqkv"), vec![3 * D, D], 0.01);
        f.add_f16(&mut rng, &(p.clone() + "wo"), vec![D, D], 0.01);
        f.add_f32(&mut rng, &(p.clone() + "attn_scale"), vec![D], 0.01);
        f.add_f32(&mut rng, &(p.clone() + "ffn_norm"), vec![D], 1.0);
        f.add_f16(&mut rng, &(p.clone() + "w13"), vec![2 * FFN, D], 0.01);
        f.add_f16(&mut rng, &(p.clone() + "w2"), vec![D, FFN], 0.01);
        f.add_f32(&mut rng, &(p.clone() + "ffn_scale"), vec![D], 0.01);
    }
    f.add_f32(&mut rng, "post.norm", vec![D], 1.0);
    for u in 0..2 {
        let p = format!("up{u}.");
        f.add_f16(&mut rng, &(p.clone() + "w"), vec![2 * D, D], 0.01);
        f.add_f32(&mut rng, &(p.clone() + "b"), vec![2 * D], 0.1);
        let q = p + "cnx.";
        // Shapes must match the real codec (dr_* kernels read full rows):
        // dw_w is [D, 7] (depthwise kernel), pw1_b is [4*D] (expanded width).
        f.add_f32(&mut rng, &(q.clone() + "dw_w"), vec![D, 7], 0.1);
        for n in ["dw_b", "ln_w", "ln_b", "pw2_b", "gamma"] {
            f.add_f32(&mut rng, &(q.clone() + n), vec![D], if n.starts_with("ln") || n == "gamma" { 1.0 } else { 0.1 });
        }
        f.add_f32(&mut rng, &(q.clone() + "pw1_b"), vec![4 * D], 0.1);
        f.add_f16(&mut rng, &(q.clone() + "pw1_w"), vec![4 * D, D], 0.01);
        f.add_f16(&mut rng, &(q.clone() + "pw2_w"), vec![D, 4 * D], 0.01);
    }
    f.add_f16(&mut rng, "dec.in.w", vec![DEC_IN, 7 * D], 0.005);
    f.add_f32(&mut rng, "dec.in.b", vec![DEC_IN], 0.0);
    let mut ci = DEC_IN;
    for (b, &r) in [8usize, 8, 4, 2].iter().enumerate() {
        let co = ci / 2;
        let p = format!("dec.b{b}.");
        f.add_f32(&mut rng, &(p.clone() + "alpha"), vec![ci], 0.5);
        f.add_f16(&mut rng, &(p.clone() + "up.w"), vec![r * co, 2 * ci], 0.005);
        f.add_f32(&mut rng, &(p.clone() + "up.b"), vec![r * co], 0.0);
        for u in 0..3 {
            let q = format!("{p}ru{u}.");
            f.add_f32(&mut rng, &(q.clone() + "a1"), vec![co], 0.5);
            f.add_f16(&mut rng, &(q.clone() + "c1.w"), vec![co, 7 * co], 0.005);
            f.add_f32(&mut rng, &(q.clone() + "c1.b"), vec![co], 0.0);
            f.add_f32(&mut rng, &(q.clone() + "a2"), vec![co], 0.5);
            f.add_f16(&mut rng, &(q.clone() + "c2.w"), vec![co, co], 0.005);
            f.add_f32(&mut rng, &(q.clone() + "c2.b"), vec![co], 0.0);
        }
        ci = co;
    }
    f.add_f32(&mut rng, "dec.out.alpha", vec![ci], 0.5);
    f.add_f32(&mut rng, "dec.out.w", vec![7 * ci], 0.005);
    f.add_f32(&mut rng, "dec.out.b", vec![1], 0.0);
    let bytes = safetensors::serialize(f.tensors.iter().cloned(), &None).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn synthetic_codec_decodes() {
    let dir: PathBuf = std::env::temp_dir().join(format!("drama-codec-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("codec.safetensors");
    build_codec(&path);
    let codec = Codec::load(&path).unwrap();
    let mut ds = codec.stream().unwrap();
    // 3 frames of codes, then 2 more through the same stream (state continuity).
    let mut rng = Rng(7);
    let codes: Vec<i32> = (0..3 * NCB)
        .map(|i| if i % NCB == 0 { (rng.next() % SEM as u64) as i32 } else { (rng.next() % RES as u64) as i32 })
        .collect();
    let a = codec.decode_chunk(&mut ds, &codes).unwrap();
    assert_eq!(a.len(), 3 * drama::codec::FRAME_SAMPLES);
    assert!(a.iter().all(|v| v.is_finite()));
    let more: Vec<i32> = (0..2 * NCB)
        .map(|i| if i % NCB == 0 { (rng.next() % SEM as u64) as i32 } else { (rng.next() % RES as u64) as i32 })
        .collect();
    let b = codec.decode_chunk(&mut ds, &more).unwrap();
    assert_eq!(b.len(), 2 * drama::codec::FRAME_SAMPLES);
    assert!(b.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(&dir);
}
