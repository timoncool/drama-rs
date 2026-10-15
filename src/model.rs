//! Model configuration and the converted weights, uploaded to the GPU.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::gpu::{DevBuf, HostBuf, Stream};

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeParameters,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub dim: usize,
    pub n_head: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub num_codebooks: usize,
    pub vocab_size: usize,
    pub shared_enrichment_layers: usize,
    pub rope_base: f64,
    pub rotary_percent: f64,
    pub vq_backbone_codebooks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub audio_decoder_config: DecoderConfig,
    pub semantic_start_token_id: i32,
    pub semantic_end_token_id: i32,
    pub eos_token_id: i32,
}

#[derive(Debug, Deserialize)]
struct Meta {
    format: String,
    q4_group: usize,
    q8_group: usize,
    config: Config,
}

impl Config {
    pub fn tower_layers(&self) -> usize {
        self.audio_decoder_config.shared_enrichment_layers + self.audio_decoder_config.num_codebooks - 1
    }
    pub fn frame_width(&self) -> usize {
        1 + self.audio_decoder_config.num_codebooks
    }
    pub fn n_candidates(&self) -> usize {
        self.audio_decoder_config.vocab_size + 1
    }
    pub fn is_linear(&self, layer: usize) -> bool {
        self.text_config.layer_types[layer] == "linear_attention"
    }
}

/// int8 weight, one f16 scale per 128 inputs of a row.
pub struct Q8 {
    pub q: DevBuf,
    pub s: DevBuf,
    pub n: usize,
    pub k: usize,
}

impl Q8 {
    /// Rows [row0, row0 + n) as raw pointers.
    pub fn rows(&self, row0: usize) -> (*const i8, *const u16) {
        (self.q.at::<i8>(row0 * self.k), self.s.at::<u16>(row0 * (self.k / 128)))
    }
}

/// Where the routed experts (about 80% of the weights) live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Experts {
    /// On the GPU with the rest of the model.
    Vram,
    /// In page-locked RAM; decoding keeps the experts in use in a VRAM cache (`Engine`).
    Ram,
}

pub enum Bank {
    Dev(DevBuf),
    Host(HostBuf),
    /// Left in the memory-mapped weights file (pageable): past the page-locking cap. Only the CPU reads it.
    Mapped(std::sync::Arc<Mmap>, *const u8),
}

unsafe impl Send for Bank {}

impl Bank {
    /// The address kernels read.
    pub fn dev(&self) -> *const u8 {
        match self {
            Bank::Dev(b) => b.ptr::<u8>(),
            Bank::Host(b) => b.dev::<u8>(),
            Bank::Mapped(..) => std::ptr::null(),
        }
    }

    /// The address a host-to-device copy reads (RAM banks only).
    pub fn host(&self) -> Option<*const u8> {
        match self {
            Bank::Dev(_) => None,
            Bank::Host(b) => Some(b.host::<u8>()),
            Bank::Mapped(_, p) => Some(*p),
        }
    }

    /// The GPU cannot read it (pageable memory).
    pub fn mapped(&self) -> bool {
        matches!(self, Bank::Mapped(..))
    }
}

/// Routed experts, int4 asymmetric with group 64: q [E, N, K/2], sm [E, N, K/64, 2].
pub struct Q4 {
    pub q: Bank,
    pub sm: Bank,
    pub n: usize,
    pub k: usize,
}

impl Q4 {
    /// Bytes of one expert in q and in sm.
    pub fn expert_bytes(&self) -> (usize, usize) {
        (self.n * self.k / 2, self.n * (self.k / 64) * 4)
    }

    pub fn expert(&self, e: usize) -> (*const u8, *const u16) {
        let (bq, bs) = self.expert_bytes();
        unsafe { (self.q.dev().add(e * bq), self.sm.dev().add(e * bs) as *const u16) }
    }
}

/// A dense tensor kept as uploaded (bf16 or f32).
pub struct Dense {
    pub buf: DevBuf,
    pub shape: Vec<usize>,
}

impl Dense {
    pub fn f32(&self) -> *const f32 {
        self.buf.ptr::<f32>()
    }
    pub fn bf16(&self) -> *const u16 {
        self.buf.ptr::<u16>()
    }
}

pub struct LinearAttn {
    pub in_proj: Q8,
    pub out_proj: Q8,
    pub conv_w: Dense,
    pub a_log: Dense,
    pub dt_bias: Dense,
    pub norm: Dense,
}

pub struct FullAttn {
    pub qkv: Q8,
    pub o: Q8,
    pub q_norm: Dense,
    pub k_norm: Dense,
}

pub enum Mixer {
    Linear(LinearAttn),
    Full(FullAttn),
}

pub struct Layer {
    pub input_norm: Dense,
    pub post_norm: Dense,
    pub mixer: Mixer,
    pub router: Dense,
    pub shared_gate_up: Q8,
    pub shared_down: Q8,
    pub gate_up: Q4,
    pub down: Q4,
}

pub struct TowerLayer {
    pub tok_emb: Dense,
    pub enorm: Dense,
    pub hnorm: Dense,
    pub final_norm: Dense,
    pub input_norm: Dense,
    pub post_norm: Dense,
    pub q_norm: Dense,
    pub k_norm: Dense,
    pub eh_proj: Q8,
    pub qkv: Q8,
    pub o: Q8,
    pub gate_up: Q8,
    pub down: Q8,
}

pub struct Model {
    pub cfg: Config,
    pub layers: Vec<Layer>,
    pub tower: Vec<TowerLayer>,
    pub heads: Q8,
    pub codebook_emb: Dense,
    pub lm_head_cand: Dense,
    pub cand_table: DevBuf,
    pub final_norm: Dense,
    pub text_inv_freq: Dense,
    pub tower_inv_freq: Dense,
    /// The full token embedding stays on the host; prefill gathers its rows.
    pub embed_tokens: HostTensor,
    pub experts: Experts,
}

pub struct HostTensor {
    _map: std::sync::Arc<Mmap>,
    ptr: *const u8,
    pub shape: Vec<usize>,
}

unsafe impl Send for HostTensor {}

impl HostTensor {
    /// Row `r` of a 2-D bf16 tensor.
    pub fn row_bf16(&self, r: usize) -> &[u16] {
        let d = self.shape[1];
        assert!(r < self.shape[0]);
        unsafe { std::slice::from_raw_parts((self.ptr as *const u16).add(r * d), d) }
    }
}

struct File {
    map: std::sync::Arc<Mmap>,
}

impl File {
    fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let map = std::sync::Arc::new(unsafe { Mmap::map(&f)? });
        Ok(Self { map })
    }
    fn st(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.map)?)
    }
}

struct Loader<'a> {
    st: SafeTensors<'a>,
    stream: &'a Stream,
    prefix: String,
    uploaded: &'a mut usize,
    experts: Experts,
    in_ram: &'a mut usize,
    map: Option<std::sync::Arc<Mmap>>,
    /// Bytes of experts that may still be page-locked; past it they stay in the mapped file.
    pin_left: &'a mut usize,
    in_file: &'a mut usize,
}

impl Loader<'_> {
    fn raw(&mut self, name: &str, dtype: Dtype) -> Result<(DevBuf, Vec<usize>)> {
        let t = self.st.tensor(name).map_err(|e| anyhow!("{}{name}: {e}", self.prefix))?;
        if t.dtype() != dtype {
            bail!("{}{name}: dtype {:?}, expected {dtype:?}", self.prefix, t.dtype());
        }
        let buf = DevBuf::from_bytes(t.data(), self.stream)?;
        *self.uploaded += t.data().len();
        Ok((buf, t.shape().to_vec()))
    }

    fn dense(&mut self, name: &str, dtype: Dtype) -> Result<Dense> {
        let (buf, shape) = self.raw(name, dtype)?;
        Ok(Dense { buf, shape })
    }

    fn f32(&mut self, name: &str) -> Result<Dense> {
        self.dense(name, Dtype::F32)
    }

    fn bf16(&mut self, name: &str) -> Result<Dense> {
        self.dense(name, Dtype::BF16)
    }

    fn q8(&mut self, name: &str) -> Result<Q8> {
        let (q, shape) = self.raw(&format!("{name}.q8"), Dtype::I8)?;
        let (s, _) = self.raw(&format!("{name}.s"), Dtype::F16)?;
        Ok(Q8 { q, s, n: shape[0], k: shape[1] })
    }

    fn bank(&mut self, name: &str, dtype: Dtype) -> Result<(Bank, Vec<usize>)> {
        if self.experts == Experts::Vram {
            let (buf, shape) = self.raw(name, dtype)?;
            return Ok((Bank::Dev(buf), shape));
        }
        let t = self.st.tensor(name).map_err(|e| anyhow!("{}{name}: {e}", self.prefix))?;
        if t.dtype() != dtype {
            bail!("{}{name}: dtype {:?}, expected {dtype:?}", self.prefix, t.dtype());
        }
        let len = t.data().len();
        if len <= *self.pin_left {
            // the driver may refuse before our cap (Windows limits page-locked memory per process): the rest stays mapped
            if let Ok(buf) = HostBuf::from_bytes(t.data()) {
                *self.pin_left -= len;
                *self.in_ram += len;
                return Ok((Bank::Host(buf), t.shape().to_vec()));
            }
            *self.pin_left = 0;
        }
        let map = self.map.clone().ok_or_else(|| anyhow!("{}{name}: no file mapping for unpinned experts", self.prefix))?;
        *self.in_file += len;
        Ok((Bank::Mapped(map, t.data().as_ptr()), t.shape().to_vec()))
    }

    fn q4(&mut self, name: &str) -> Result<Q4> {
        let (q, shape) = self.bank(&format!("{name}.q4"), Dtype::U8)?;
        let (sm, _) = self.bank(&format!("{name}.sm"), Dtype::F16)?;
        Ok(Q4 { q, sm, n: shape[1], k: shape[2] * 2 })
    }
}

/// Page-locked memory the experts may take: `DRAMA_PIN_CAP_GB`, else 45% of physical RAM on Windows (WDDM refuses
/// page-locking past about half of it per process) and 75% elsewhere.
fn pin_cap() -> usize {
    if let Some(gb) = std::env::var("DRAMA_PIN_CAP_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
        return (gb * (1u64 << 30) as f64) as usize;
    }
    let ram = physical_ram();
    if cfg!(windows) { ram / 100 * 45 } else { ram / 4 * 3 }
}

#[cfg(windows)]
fn physical_ram() -> usize {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_ext: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
    }
    let mut m = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, load: 0, total_phys: 0, avail_phys: 0, total_page: 0, avail_page: 0, total_virtual: 0, avail_virtual: 0, avail_ext: 0 };
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return usize::MAX;
    }
    m.total_phys as usize
}

#[cfg(not(windows))]
fn physical_ram() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemTotal:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok()))
        .map_or(usize::MAX, |kb| kb * 1024)
}

fn inv_freq(theta: f64, rot: usize) -> Vec<f32> {
    // Same arithmetic as torch: 1 / theta ** (arange(0, rot, 2) / rot) in f32.
    (0..rot / 2).map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / rot as f32)).collect()
}

/// VRAM the engine needs besides the weights and the codec with `Experts::Vram`: KV caches, decode and prefill buffers,
/// cuBLAS and the codec's working memory (measured 1.46 GiB at a 2048-row context: peak 21.83 GiB of the process).
const RUNTIME_RESERVE: u64 = 1792 << 20;

/// VRAM the whole model takes on the GPU with `Experts::Vram`, codec weights excluded.
pub fn full_vram(dir: &Path) -> Result<u64> {
    let mut weights = 0u64;
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let p = entry?.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("layer") || name == "tower.safetensors" {
            weights += std::fs::metadata(&p)?.len();
        }
    }
    Ok(weights + RUNTIME_RESERVE)
}

impl Model {
    pub fn load(dir: &Path, stream: &Stream, experts: Experts) -> Result<Self> {
        let progress = std::io::IsTerminal::is_terminal(&std::io::stderr());
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("drama.json")).context("read drama.json")?)?;
        if meta.format != "drama-rs-v1" || meta.q4_group != 64 || meta.q8_group != 128 {
            bail!("unsupported converted format in {}", dir.display());
        }
        let cfg = meta.config;
        let tc = &cfg.text_config;
        let dc = &cfg.audio_decoder_config;
        let t0 = std::time::Instant::now();
        let mut uploaded = 0usize;
        let mut in_ram = 0usize;
        let mut in_file = 0usize;
        let mut pin_left = pin_cap();

        let mut layers = Vec::with_capacity(tc.num_hidden_layers);
        for i in 0..tc.num_hidden_layers {
            let file = File::open(&dir.join(format!("layer{i:02}.safetensors")))?;
            let mut l = Loader {
                st: file.st()?,
                stream,
                prefix: format!("layer{i:02}."),
                uploaded: &mut uploaded,
                experts,
                in_ram: &mut in_ram,
                map: Some(file.map.clone()),
                pin_left: &mut pin_left,
                in_file: &mut in_file,
            };
            let mixer = if cfg.is_linear(i) {
                Mixer::Linear(LinearAttn {
                    in_proj: l.q8("la.in_proj")?,
                    out_proj: l.q8("la.out_proj")?,
                    conv_w: l.f32("la.conv_w")?,
                    a_log: l.f32("la.A_log")?,
                    dt_bias: l.f32("la.dt_bias")?,
                    norm: l.f32("la.norm")?,
                })
            } else {
                Mixer::Full(FullAttn {
                    qkv: l.q8("attn.qkv")?,
                    o: l.q8("attn.o")?,
                    q_norm: l.f32("attn.q_norm")?,
                    k_norm: l.f32("attn.k_norm")?,
                })
            };
            layers.push(Layer {
                input_norm: l.f32("input_norm")?,
                post_norm: l.f32("post_norm")?,
                mixer,
                router: l.bf16("moe.router")?,
                shared_gate_up: l.q8("moe.shared_gate_up")?,
                shared_down: l.q8("moe.shared_down")?,
                gate_up: l.q4("moe.gate_up")?,
                down: l.q4("moe.down")?,
            });
            stream.sync()?;
            if progress {
                eprint!("\rloading weights: layer {}/{}", i + 1, tc.num_hidden_layers);
            }
        }
        if progress {
            eprintln!();
        }

        let file = File::open(&dir.join("tower.safetensors"))?;
        let mut l = Loader { st: file.st()?, stream, prefix: "tower.".into(), uploaded: &mut uploaded, experts: Experts::Vram, in_ram: &mut in_ram, map: None, pin_left: &mut pin_left, in_file: &mut in_file };
        let mut tower = Vec::new();
        for t in 0..cfg.tower_layers() {
            let p = format!("t{t}.");
            tower.push(TowerLayer {
                tok_emb: l.bf16(&(p.clone() + "tok_emb"))?,
                enorm: l.f32(&(p.clone() + "enorm"))?,
                hnorm: l.f32(&(p.clone() + "hnorm"))?,
                final_norm: l.f32(&(p.clone() + "final_norm"))?,
                input_norm: l.f32(&(p.clone() + "input_norm"))?,
                post_norm: l.f32(&(p.clone() + "post_norm"))?,
                q_norm: l.f32(&(p.clone() + "q_norm"))?,
                k_norm: l.f32(&(p.clone() + "k_norm"))?,
                eh_proj: l.q8(&(p.clone() + "eh_proj"))?,
                qkv: l.q8(&(p.clone() + "qkv"))?,
                o: l.q8(&(p.clone() + "o"))?,
                gate_up: l.q8(&(p.clone() + "gate_up"))?,
                down: l.q8(&(p.clone() + "down"))?,
            });
        }
        let heads = l.q8("heads")?;
        let codebook_emb = l.bf16("codebook_emb")?;
        stream.sync()?;
        drop(file);

        let gfile = File::open(&dir.join("globals.safetensors"))?;
        let gst = gfile.st()?;
        let mut l = Loader { st: gfile.st()?, stream, prefix: "globals.".into(), uploaded: &mut uploaded, experts: Experts::Vram, in_ram: &mut in_ram, map: None, pin_left: &mut pin_left, in_file: &mut in_file };
        let lm_head_cand = l.bf16("lm_head_cand")?;
        let final_norm = l.f32("final_norm")?;
        stream.sync()?;

        // Candidate embedding rows [eos, semantic...] for decode-time lookups.
        let emb = gst.tensor("embed_tokens")?;
        let d = tc.hidden_size;
        let n_cand = cfg.n_candidates();
        let mut cand = Vec::<u8>::with_capacity(n_cand * d * 2);
        let row_bytes = d * 2;
        let ids: Vec<i32> = std::iter::once(cfg.eos_token_id).chain(cfg.semantic_start_token_id..=cfg.semantic_end_token_id).collect();
        for &id in &ids {
            let off = id as usize * row_bytes;
            cand.extend_from_slice(&emb.data()[off..off + row_bytes]);
        }
        let cand_table = DevBuf::from_bytes(&cand, stream)?;
        let emb_shape = emb.shape().to_vec();
        let emb_offset = emb.data().as_ptr() as usize - gfile.map.as_ptr() as usize;
        stream.sync()?;
        let embed_tokens = HostTensor { ptr: unsafe { gfile.map.as_ptr().add(emb_offset) }, shape: emb_shape, _map: gfile.map };

        let rot = (tc.head_dim as f64 * tc.rope_parameters.partial_rotary_factor) as usize;
        let text_inv_freq = Dense {
            buf: DevBuf::from_slice(&inv_freq(tc.rope_parameters.rope_theta, rot), stream)?,
            shape: vec![rot / 2],
        };
        let trot = (dc.head_dim as f64 * dc.rotary_percent) as usize;
        let tower_inv_freq = Dense { buf: DevBuf::from_slice(&inv_freq(dc.rope_base, trot), stream)?, shape: vec![trot / 2] };
        stream.sync()?;

        if in_ram + in_file > 0 {
            eprintln!(
                "loaded {:.2} GB of weights to the GPU and {:.2} GB of experts to page-locked RAM in {:.1}s",
                (uploaded + cand.len()) as f64 / 1e9,
                in_ram as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            );
            if in_file > 0 {
                eprintln!("{:.2} GB of experts past the page-locking cap stay in the mapped files; the CPU computes them", in_file as f64 / 1e9);
            }
        } else {
            eprintln!("loaded {:.2} GB of weights in {:.1}s", (uploaded + cand.len()) as f64 / 1e9, t0.elapsed().as_secs_f64());
        }
        Ok(Self {
            cfg,
            layers,
            tower,
            heads,
            codebook_emb,
            lm_head_cand,
            cand_table,
            final_norm,
            text_inv_freq,
            tower_inv_freq,
            embed_tokens,
            experts,
        })
    }
}

#[allow(dead_code)]
pub fn tensor_names(dir: &Path) -> Result<HashMap<String, Vec<usize>>> {
    let mut out = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        if p.extension().is_some_and(|e| e == "safetensors") {
            let f = File::open(&p)?;
            for (name, info) in f.st()?.tensors() {
                out.insert(name, info.shape().to_vec());
            }
        }
    }
    Ok(out)
}
