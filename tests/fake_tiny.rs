//! End-to-end smoke test for the Metal backend (feature `metal`).
//!
//! Builds a tiny but shape-consistent model in a temp dir (all dimensions that
//! the engine hard-codes match the real model: H = 2048, I = 512, topk = 8,
//! hd = 256, tower dim = 2048) and runs prefill + decode steps through it.
//! This exercises every backbone/tower op: MSL kernels, CPU reference ops,
//! graph capture and replay. Values are random, so only shapes, finiteness
//! and plumbing are asserted — numeric parity with CUDA is covered by
//! tools/check_engine.py on a CUDA machine.
#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};

use drama::engine::{Engine, SampleParams};
use drama::model::Model;
use safetensors::tensor::TensorView;
use safetensors::Dtype;

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

fn bf16_rn(v: f32) -> u16 {
    let b = v.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn f16_bits(v: f32) -> u16 {
    // Round-to-nearest f16 for finite values in range.
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
    // Owns the bytes the views borrow.
    _blobs: Vec<Vec<u8>>,
}

impl File {
    fn new() -> Self {
        Self { tensors: Vec::new(), _blobs: Vec::new() }
    }
    // Leak the bytes so views can be 'static (test-only).
    fn push(&mut self, name: String, view: TensorView<'static>) {
        self.tensors.push((name, view));
    }
    fn add_raw(&mut self, name: &str, dtype: Dtype, shape: Vec<usize>, data: Vec<u8>) {
        let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
        let view = TensorView::new(dtype, shape, leaked).unwrap();
        self.push(name.to_string(), view);
    }
    fn add_f32(&mut self, rng: &mut Rng, name: &str, shape: Vec<usize>, scale: f32) {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.f32() * scale).collect();
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4).to_vec() };
        std::mem::forget(v);
        self.add_raw(name, Dtype::F32, shape, bytes);
    }
    fn add_bf16(&mut self, rng: &mut Rng, name: &str, shape: Vec<usize>, scale: f32) {
        let n: usize = shape.iter().product();
        let v: Vec<u16> = (0..n).map(|_| bf16_rn(rng.f32() * scale)).collect();
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 2).to_vec() };
        std::mem::forget(v);
        self.add_raw(name, Dtype::BF16, shape, bytes);
    }
    fn add_q8(&mut self, rng: &mut Rng, name: &str, n: usize, k: usize) {
        assert_eq!(k % 128, 0);
        let q: Vec<i8> = (0..n * k).map(|_| (rng.next() % 256) as u8 as i8).collect();
        let qb = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, n * k).to_vec() };
        std::mem::forget(q);
        // Positive scales around 0.02: exactly representable enough, finite.
        let s: Vec<u16> = (0..n * (k / 128)).map(|i| f16_bits(0.02 * (1.0 + (i % 7) as f32 * 0.1))).collect();
        let sb = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2).to_vec() };
        std::mem::forget(s);
        self.add_raw(&format!("{name}.q8"), Dtype::I8, vec![n, k], qb);
        self.add_raw(&format!("{name}.s"), Dtype::F16, vec![n, k / 128], sb);
    }
    fn add_q4(&mut self, rng: &mut Rng, name: &str, e: usize, n: usize, k: usize) {
        assert_eq!(k % 64, 0);
        let q: Vec<u8> = (0..e * n * (k / 2)).map(|_| (rng.next() % 256) as u8).collect();
        let qb = q.clone();
        std::mem::forget(q);
        let sm: Vec<u16> = (0..e * n * (k / 64) * 2)
            .map(|i| if i % 2 == 0 { f16_bits(0.03 * (1.0 + (i % 5) as f32 * 0.1)) } else { f16_bits(rng.f32() * 0.05) })
            .collect();
        let smb = unsafe { std::slice::from_raw_parts(sm.as_ptr() as *const u8, sm.len() * 2).to_vec() };
        std::mem::forget(sm);
        self.add_raw(&format!("{name}.q4"), Dtype::U8, vec![e, n, k / 2], qb);
        self.add_raw(&format!("{name}.sm"), Dtype::F16, vec![e, n, k / 64, 2], smb);
    }
    fn write(&self, path: &Path) {
        let bytes = safetensors::serialize(self.tensors.iter().cloned(), &None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}

// Model dimensions: the ones the engine hard-codes match the real model,
// everything else is shrunk.
const D: usize = 2048;
const MI: usize = 512;
const E: usize = 8;
const CB: usize = 64;
const NCB: usize = 2;
const NB: usize = 1;
const HD: usize = 256;
const SEM0: i32 = 100;
const SEM1: i32 = 163;
const EOS: i32 = 1;

fn layer_file(rng: &mut Rng, linear: bool) -> File {
    let mut f = File::new();
    f.add_f32(rng, "input_norm", vec![D], 1.0);
    f.add_f32(rng, "post_norm", vec![D], 1.0);
    if linear {
        // nk = nv = 2, lkd = lvd = 128: conv_c = 768, vd = 256, ps = 1028.
        f.add_q8(rng, "la.in_proj", 1028, D);
        f.add_q8(rng, "la.out_proj", D, 256);
        f.add_f32(rng, "la.conv_w", vec![768, 4], 0.2);
        f.add_f32(rng, "la.A_log", vec![2], 0.5);
        f.add_f32(rng, "la.dt_bias", vec![2], 0.5);
        f.add_f32(rng, "la.norm", vec![128], 1.0);
    } else {
        // 8 heads, 2 kv heads.
        f.add_q8(rng, "attn.qkv", 2 * D + 2 * 512, D);
        f.add_q8(rng, "attn.o", D, D);
        f.add_f32(rng, "attn.q_norm", vec![HD], 1.0);
        f.add_f32(rng, "attn.k_norm", vec![HD], 1.0);
    }
    f.add_bf16(rng, "moe.router", vec![E + 1, D], 0.3);
    f.add_q8(rng, "moe.shared_gate_up", 2 * MI, D);
    f.add_q8(rng, "moe.shared_down", D, MI);
    f.add_q4(rng, "moe.gate_up", E, 2 * MI, D);
    f.add_q4(rng, "moe.down", E, D, MI);
    f
}

fn build_tiny(dir: &Path) {
    let mut rng = Rng(0x5eed);
    layer_file(&mut rng, true).write(&dir.join("layer00.safetensors"));
    layer_file(&mut rng, false).write(&dir.join("layer01.safetensors"));

    let mut t = File::new();
    for l in 0..2 {
        let p = format!("t{l}.");
        t.add_bf16(&mut rng, &(p.clone() + "tok_emb"), vec![CB, D], 0.3);
        for n in ["enorm", "hnorm", "final_norm", "input_norm", "post_norm"] {
            t.add_f32(&mut rng, &(p.clone() + n), vec![D], 1.0);
        }
        t.add_f32(&mut rng, &(p.clone() + "q_norm"), vec![HD], 1.0);
        t.add_f32(&mut rng, &(p.clone() + "k_norm"), vec![HD], 1.0);
        t.add_q8(&mut rng, &(p.clone() + "eh_proj"), D, 2 * D);
        t.add_q8(&mut rng, &(p.clone() + "qkv"), 2 * D + 2 * 512, D);
        t.add_q8(&mut rng, &(p.clone() + "o"), D, D);
        t.add_q8(&mut rng, &(p.clone() + "gate_up"), 256, D);
        t.add_q8(&mut rng, &(p.clone() + "down"), D, 128);
    }
    t.add_q8(&mut rng, "heads", CB, D);
    t.add_bf16(&mut rng, "codebook_emb", vec![NCB * CB, D], 0.3);
    t.write(&dir.join("tower.safetensors"));

    let mut g = File::new();
    g.add_bf16(&mut rng, "lm_head_cand", vec![CB + 1, D], 0.3);
    g.add_f32(&mut rng, "final_norm", vec![D], 1.0);
    g.add_bf16(&mut rng, "embed_tokens", vec![200, D], 0.3);
    g.write(&dir.join("globals.safetensors"));

    let json = format!(
        r#"{{"format":"drama-rs-v1","q4_group":64,"q8_group":128,"config":{{
        "text_config":{{"hidden_size":{D},"num_hidden_layers":2,
        "layer_types":["linear_attention","full_attention"],
        "num_attention_heads":8,"num_key_value_heads":2,"head_dim":{HD},
        "rms_norm_eps":1e-5,"rope_parameters":{{"rope_theta":10000.0,"partial_rotary_factor":0.25}},
        "num_experts":{E},"num_experts_per_tok":8,"moe_intermediate_size":{MI},
        "shared_expert_intermediate_size":{MI},
        "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_value_head_dim":128,
        "linear_num_key_heads":2,"linear_num_value_heads":2}},
        "audio_decoder_config":{{"dim":{D},"n_head":4,"n_local_heads":2,"head_dim":{HD},
        "intermediate_size":128,"norm_eps":1e-5,"num_codebooks":{NCB},"vocab_size":{CB},
        "shared_enrichment_layers":1,"rope_base":10000.0,"rotary_percent":0.5,
        "vq_backbone_codebooks":{NB}}},
        "semantic_start_token_id":{SEM0},"semantic_end_token_id":{SEM1},"eos_token_id":{EOS}}}}}"#
    );
    std::fs::write(dir.join("drama.json"), json).unwrap();
}

fn tiny_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("drama-metal-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    build_tiny(&dir);
    dir
}

#[test]
fn prefill_and_steps_run() {
    let dir = tiny_dir("e2e");
    let stream = drama::gpu::Stream::new().unwrap();
    let model = Model::load(&dir, &stream).unwrap();
    let mut engine = Engine::new(model, 64, 16).unwrap();

    // 4 prompt rows, mains in the semantic band, width = 1 + num_codebooks.
    let frames: Vec<i32> = vec![100, 1, 2, 110, 3, 4, 120, 5, 6, 130, 7, 8];
    engine.prefill(&frames, SampleParams::default()).unwrap();

    engine.trace_logits = true;
    for step in 0..3 {
        let (frame, trace) = engine.step().unwrap();
        assert_eq!(frame.len(), NCB + 1, "step {step}");
        let trace = trace.expect("trace_logits");
        assert_eq!(trace.main_logits.len(), CB + 1);
        assert_eq!(trace.head_logits.len(), (NCB - 1) * CB);
        assert!(trace.main_logits.iter().all(|v| v.is_finite()), "step {step} main logits");
        assert!(trace.head_logits.iter().all(|v| v.is_finite()), "step {step} head logits");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tensor_inventory_matches_loader() {
    // Every tensor Model::load asks for exists with the dtype it expects.
    let dir = tiny_dir("inv");
    let names = drama::model::tensor_names(&dir).unwrap();
    assert_eq!(names.get("moe.router"), Some(&vec![E + 1, D]));
    assert_eq!(names.get("moe.gate_up.q4"), Some(&vec![E, 2 * MI, D / 2]));
    assert_eq!(names.get("t0.qkv.q8"), Some(&vec![2 * D + 2 * 512, D]));
    assert_eq!(names.get("lm_head_cand"), Some(&vec![CB + 1, D]));
    assert_eq!(names.get("embed_tokens"), Some(&vec![200, D]));
    let _ = std::fs::remove_dir_all(&dir);
}
