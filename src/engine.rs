//! Inference: chunked prefill with cuBLAS, then one CUDA-graph replay per audio frame.
//!
//! A decode step (one graph replay) takes the backbone's final hidden state at position P
//! and the frame that was fed at P, and
//!   1. samples the main token from the candidate logits (with RAS),
//!   2. runs the 13 tower layers at P, sampling codebooks 1..9 from the heads,
//!   3. makes that frame the input at P + 1 and runs the backbone on it.

use std::ffi::c_void;

use anyhow::{bail, Result};

use crate::ffi;
use crate::gpu::{launched, Cublas, DevBuf, Graph, Pinned, Stream};
use crate::model::{Dense, Mixer, Model, Q8};

pub const RAS_WINDOW: usize = 16;
const ROUTER_PREFETCH: bool = false;
const GEMV_PREFETCH: bool = true;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub pad: i32,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 30, top_p: 0.9, pad: 0, seed: 0 }
    }
}

const F32: usize = 4;
const BF16: usize = 2;
const I32: usize = 4;

/// Per-step buffers of the decode graph (one row each).
struct DecodeBufs {
    x: DevBuf,
    proj: DevBuf,
    conv_out: DevBuf,
    la_o: DevBuf,
    qkv: DevBuf,
    q: DevBuf,
    attn_o: DevBuf,
    part: DevBuf,
    rlog: DevBuf,
    idx: DevBuf,
    wts: DevBuf,
    sgate: DevBuf,
    act: DevBuf,
    counter: DevBuf,
    main_logits: DevBuf,
    tin: DevBuf,
    mixed: DevBuf,
    r: DevBuf,
    ff_act: DevBuf,
    r2: DevBuf,
    head_logits: DevBuf,
}

/// Buffers for a prefill chunk of up to `chunk` rows.
struct PrefillBufs {
    emb_host: Pinned<u16>,
    emb: DevBuf,
    x: DevBuf,
    hn_b: DevBuf,
    proj: DevBuf,
    conv_out: DevBuf,
    la_o: DevBuf,
    o_b: DevBuf,
    qkv: DevBuf,
    q: DevBuf,
    attn_o: DevBuf,
    rlog: DevBuf,
    idx: DevBuf,
    wts: DevBuf,
    sgate: DevBuf,
    shh: DevBuf,
    sh_act_b: DevBuf,
    sh_out: DevBuf,
    counts: DevBuf,
    offsets: DevBuf,
    fill: DevBuf,
    slot_of: DevBuf,
    xp: DevBuf,
    ap: DevBuf,
    yp: DevBuf,
    h_all: DevBuf,
    tin_b: DevBuf,
    mixed: DevBuf,
    an_b: DevBuf,
    gu: DevBuf,
    act_b: DevBuf,
    th: DevBuf,
    wbuf: DevBuf,
}

pub struct Engine {
    pub model: Model,
    st: Stream,
    blas: Cublas,
    pub max_len: usize,
    chunk: usize,
    kv: Vec<Option<(DevBuf, DevBuf)>>,
    tkv: Vec<(DevBuf, DevBuf)>,
    conv_state: Vec<Option<DevBuf>>,
    rec_state: Vec<Option<DevBuf>>,
    d_pos: DevBuf,
    d_pref_pos: DevBuf,
    d_step: DevBuf,
    d_in_frame: DevBuf,
    d_next_frame: DevBuf,
    d_ras: DevBuf,
    d_params: DevBuf,
    d_hist: DevBuf,
    d_hist_len: DevBuf,
    dec: DecodeBufs,
    pre: PrefillBufs,
    graph: Option<Graph>,
    frame_host: Pinned<i32>,
    pub trace_logits: bool,
    /// Set from any thread to end the running `generate` at its next frame with [`Cancelled`].
    pub stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// The error `generate` ends with when [`Engine::stop`] was set.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("generation cancelled")
    }
}

impl std::error::Error for Cancelled {}

pub struct StepTrace {
    pub main_logits: Vec<f32>,
    pub head_logits: Vec<f32>,
}

/// One recorded decode launch. `weights` is the region it streams (for the previous op to
/// prefetch); `run` gets the region of the next weight-streaming op to prefetch itself.
type Prefetch = (*const u8, usize);

struct Op<'a> {
    weights: Option<Prefetch>,
    run: Box<dyn Fn(*const std::ffi::c_char, usize) -> Result<()> + 'a>,
}

impl<'a> Op<'a> {
    fn plain(run: impl Fn(*const std::ffi::c_char, usize) -> Result<()> + 'a) -> Self {
        Self { weights: None, run: Box::new(run) }
    }
    fn weights(p: *const u8, bytes: usize, run: impl Fn(*const std::ffi::c_char, usize) -> Result<()> + 'a) -> Self {
        Self { weights: Some((p, bytes)), run: Box::new(run) }
    }
}

fn norm_args(norm: Option<(&Dense, f32)>) -> (*const f32, f32) {
    norm.map(|(w, eps)| (w.f32(), eps)).unwrap_or((std::ptr::null(), 0.0))
}

fn f(b: &DevBuf) -> *mut f32 {
    b.ptr::<f32>()
}
fn h(b: &DevBuf) -> *mut u16 {
    b.ptr::<u16>()
}
fn i(b: &DevBuf) -> *mut i32 {
    b.ptr::<i32>()
}

impl Engine {
    pub fn new(model: Model, max_len: usize, chunk: usize) -> Result<Self> {
        let st = Stream::new()?;
        let blas = Cublas::new(&st)?;
        launched(unsafe { ffi::dr_sample_init() }, "sample init")?;
        let cfg = model.cfg.clone();
        let tc = &cfg.text_config;
        let dc = &cfg.audio_decoder_config;
        let d = tc.hidden_size;
        let width = cfg.frame_width();
        let kv_row = tc.num_key_value_heads * tc.head_dim;
        let tkv_row = dc.n_local_heads * dc.head_dim;

        let mut kv = Vec::new();
        let mut conv_state = Vec::new();
        let mut rec_state = Vec::new();
        let conv_c = 2 * tc.linear_num_key_heads * tc.linear_key_head_dim + tc.linear_num_value_heads * tc.linear_value_head_dim;
        for l in 0..tc.num_hidden_layers {
            if cfg.is_linear(l) {
                kv.push(None);
                conv_state.push(Some(DevBuf::zeros(conv_c * (tc.linear_conv_kernel_dim - 1) * F32, &st)?));
                rec_state.push(Some(DevBuf::zeros(
                    tc.linear_num_value_heads * tc.linear_key_head_dim * tc.linear_value_head_dim * F32,
                    &st,
                )?));
            } else {
                kv.push(Some((DevBuf::alloc(max_len * kv_row * BF16)?, DevBuf::alloc(max_len * kv_row * BF16)?)));
                conv_state.push(None);
                rec_state.push(None);
            }
        }
        let mut tkv = Vec::new();
        for _ in 0..cfg.tower_layers() {
            tkv.push((DevBuf::alloc(max_len * tkv_row * BF16)?, DevBuf::alloc(max_len * tkv_row * BF16)?));
        }

        let in_proj_n = model
            .layers
            .iter()
            .find_map(|l| match &l.mixer {
                Mixer::Linear(la) => Some(la.in_proj.n),
                _ => None,
            })
            .unwrap_or(0);
        let qkv_n = (2 * tc.num_attention_heads * tc.head_dim + 2 * kv_row).max(2 * dc.n_head * dc.head_dim + 2 * tkv_row);
        let attn_w = (tc.num_attention_heads * tc.head_dim).max(dc.n_head * dc.head_dim);
        let la_o_n = tc.linear_num_value_heads * tc.linear_value_head_dim;
        let e = tc.num_experts;
        let topk = tc.num_experts_per_tok;
        let mi = tc.moe_intermediate_size;
        let si = tc.shared_expert_intermediate_size;
        let ti = dc.intermediate_size;
        let part_floats = unsafe { ffi::dr_attn_part_floats() } as usize;
        let heads = tc.num_attention_heads.max(dc.n_head);
        let cb = dc.vocab_size;

        let dec = DecodeBufs {
            x: DevBuf::alloc(d * F32)?,
            proj: DevBuf::alloc(in_proj_n.max(1) * F32)?,
            conv_out: DevBuf::alloc(conv_c * F32)?,
            la_o: DevBuf::alloc(la_o_n * F32)?,
            qkv: DevBuf::alloc(qkv_n * F32)?,
            q: DevBuf::alloc(attn_w * F32)?,
            attn_o: DevBuf::alloc(attn_w.max(la_o_n) * F32)?,
            part: DevBuf::alloc(heads * part_floats * F32)?,
            rlog: DevBuf::alloc((e + 1) * F32)?,
            idx: DevBuf::alloc(topk * I32)?,
            wts: DevBuf::alloc(topk * F32)?,
            sgate: DevBuf::alloc(F32)?,
            act: DevBuf::alloc((topk + 1) * mi.max(si) * F32)?,
            counter: DevBuf::zeros(16, &st)?,
            main_logits: DevBuf::alloc(cfg.n_candidates() * F32)?,
            tin: DevBuf::alloc(2 * d * F32)?,
            mixed: DevBuf::alloc(d * F32)?,
            r: DevBuf::alloc(d * F32)?,
            ff_act: DevBuf::alloc(ti * F32)?,
            r2: DevBuf::alloc(d * F32)?,
            head_logits: DevBuf::alloc((dc.num_codebooks - 1) * cb * F32)?,
        };

        // Largest dense matrix the prefill dequantises at once.
        let mut wmax = 0usize;
        let mut see = |q: &Q8| wmax = wmax.max(q.n * q.k);
        for l in &model.layers {
            match &l.mixer {
                Mixer::Linear(la) => {
                    see(&la.in_proj);
                    see(&la.out_proj);
                }
                Mixer::Full(fa) => {
                    see(&fa.qkv);
                    see(&fa.o);
                }
            }
            see(&l.shared_gate_up);
            see(&l.shared_down);
        }
        for t in &model.tower {
            for q in [&t.eh_proj, &t.qkv, &t.o, &t.gate_up, &t.down] {
                see(q);
            }
        }
        wmax = wmax.max(2 * mi * d).max(d * mi);

        let c = chunk;
        let pre = PrefillBufs {
            emb_host: Pinned::new(c * d)?,
            emb: DevBuf::alloc(c * d * BF16)?,
            x: DevBuf::alloc(c * d * F32)?,
            hn_b: DevBuf::alloc(c * d * BF16)?,
            proj: DevBuf::alloc(c * in_proj_n.max(1) * F32)?,
            conv_out: DevBuf::alloc(c * conv_c * F32)?,
            la_o: DevBuf::alloc(c * la_o_n.max(attn_w) * F32)?,
            o_b: DevBuf::alloc(c * la_o_n.max(attn_w).max(ti) * BF16)?,
            qkv: DevBuf::alloc(c * qkv_n * F32)?,
            q: DevBuf::alloc(c * attn_w * F32)?,
            attn_o: DevBuf::alloc(c * attn_w * F32)?,
            rlog: DevBuf::alloc(c * (e + 1) * F32)?,
            idx: DevBuf::alloc(c * topk * I32)?,
            wts: DevBuf::alloc(c * topk * F32)?,
            sgate: DevBuf::alloc(c * F32)?,
            shh: DevBuf::alloc(c * 2 * si * F32)?,
            sh_act_b: DevBuf::alloc(c * si * BF16)?,
            sh_out: DevBuf::alloc(c * d * F32)?,
            counts: DevBuf::alloc(e * I32)?,
            offsets: DevBuf::alloc(e * I32)?,
            fill: DevBuf::alloc(e * I32)?,
            slot_of: DevBuf::alloc(c * topk * I32)?,
            xp: DevBuf::alloc(c * topk * d * BF16)?,
            ap: DevBuf::alloc(c * topk * mi * BF16)?,
            yp: DevBuf::alloc(c * topk * d * F32)?,
            h_all: DevBuf::alloc(c * d * F32)?,
            tin_b: DevBuf::alloc(c * 2 * d * BF16)?,
            mixed: DevBuf::alloc(c * d * F32)?,
            an_b: DevBuf::alloc(c * d * BF16)?,
            gu: DevBuf::alloc(c * 2 * ti * F32)?,
            act_b: DevBuf::alloc(c * ti * BF16)?,
            th: DevBuf::alloc(c * d * F32)?,
            wbuf: DevBuf::alloc(wmax * BF16)?,
        };

        let mut engine = Self {
            st,
            blas,
            max_len,
            chunk,
            kv,
            tkv,
            conv_state,
            rec_state,
            d_pos: DevBuf::zeros(I32, &Stream::new()?)?,
            d_pref_pos: DevBuf::alloc(I32)?,
            d_step: DevBuf::alloc(8)?,
            d_in_frame: DevBuf::alloc(width * I32)?,
            d_next_frame: DevBuf::alloc(width * I32)?,
            d_ras: DevBuf::alloc(RAS_WINDOW * I32)?,
            d_params: DevBuf::alloc(std::mem::size_of::<SampleParams>())?,
            d_hist: DevBuf::alloc(max_len * width * I32)?,
            d_hist_len: DevBuf::alloc(I32)?,
            dec,
            pre,
            graph: None,
            frame_host: Pinned::new(width)?,
            trace_logits: false,
            stop: Default::default(),
            model,
        };
        engine.st.sync()?;
        // Everything a decode step touches lives at fixed addresses: capture it once.
        let graph = Graph::capture(&engine.st, || engine.record_step())?;
        engine.graph = Some(graph);
        Ok(engine)
    }

    // ------------------------------------------------------------------------------ helpers

    fn s(&self) -> ffi::Stream {
        self.st.raw
    }

    fn rmsnorm(&self, x: *const f32, w: &Dense, y: *mut f32, yb: *mut u16, rows: usize, eps: f32) -> Result<()> {
        let d = w.shape[0];
        launched(unsafe { ffi::dr_rmsnorm(x, w.f32(), y, yb, rows as i32, d as i32, eps, self.s()) }, "rmsnorm")
    }

    /// C = A . dequant(W)^T + beta C for a q8 weight (prefill).
    fn gemm_q8(&self, a: *const u16, w: &Q8, c: *mut f32, m: usize, beta: f32) -> Result<()> {
        let (q, s) = w.rows(0);
        launched(unsafe { ffi::dr_dequant_q8(q, s, h(&self.pre.wbuf), w.n as i32, w.k as i32, self.s()) }, "dequant_q8")?;
        self.blas.gemm(a, h(&self.pre.wbuf), c, m, w.n, w.k, beta)
    }


    // ------------------------------------------------------------------------------ prefill

    /// Run the prompt rows ([rows, width] row-major) and leave the engine ready to decode.
    pub fn prefill(&mut self, frames: &[i32], params: SampleParams) -> Result<()> {
        let cfg = self.model.cfg.clone();
        let width = cfg.frame_width();
        let rows = frames.len() / width;
        if rows < 2 {
            bail!("a prompt needs at least 2 rows");
        }
        if rows >= self.max_len {
            bail!("prompt of {rows} rows does not fit the context of {}", self.max_len);
        }
        // Fresh recurrent state.
        for s in self.conv_state.iter().chain(self.rec_state.iter()).flatten() {
            s.zero(&self.st)?;
        }
        let d_frames = DevBuf::from_slice(frames, &self.st)?;
        let mut start = 0;
        while start < rows {
            let n = (rows - start).min(self.chunk);
            self.prefill_chunk(&d_frames, frames, start, n, rows)?;
            start += n;
        }

        // Decode state: hidden of the last row, its frame, the RAS window.
        let d = cfg.text_config.hidden_size;
        let last = (rows - 1 - (rows - 1) / self.chunk * self.chunk) * d;
        unsafe {
            ffi::dr_d2d(f(&self.dec.x) as *mut c_void, self.pre.x.at::<f32>(last) as *const c_void, d * F32, self.s());
        }
        self.d_pos.upload_at(0, &[(rows - 1) as i32], &self.st)?;
        self.d_in_frame.upload_at(0, &frames[(rows - 1) * width..rows * width], &self.st)?;
        let mut ras = vec![-1i32; RAS_WINDOW];
        let tail: Vec<i32> = (rows.saturating_sub(RAS_WINDOW)..rows).map(|r| frames[r * width]).collect();
        ras[RAS_WINDOW - tail.len()..].copy_from_slice(&tail);
        self.d_ras.upload_at(0, &ras, &self.st)?;
        self.d_params.upload_at(0, &[params], &self.st)?;
        self.d_step.upload_at(0, &[0u64], &self.st)?;
        self.d_hist_len.upload_at(0, &[0i32], &self.st)?;
        self.st.sync()?;
        Ok(())
    }

    fn prefill_chunk(&self, d_frames: &DevBuf, frames: &[i32], start: usize, n: usize, rows: usize) -> Result<()> {
        let cfg = &self.model.cfg;
        let tc = &cfg.text_config;
        let dc = &cfg.audio_decoder_config;
        let d = tc.hidden_size;
        let width = cfg.frame_width();
        let eps = tc.rms_norm_eps;
        let p = &self.pre;
        let s = self.s();

        // Embeddings: gather the text rows on the host, mix codebook rows on the device.
        let emb = &self.model.embed_tokens;
        {
            let host = unsafe { std::slice::from_raw_parts_mut(p.emb_host.ptr(), n * d) };
            for r in 0..n {
                let tok = frames[(start + r) * width] as usize;
                host[r * d..(r + 1) * d].copy_from_slice(emb.row_bf16(tok));
            }
        }
        unsafe { ffi::dr_h2d(p.emb.ptr(), p.emb_host.ptr() as *const c_void, n * d * BF16, s) };
        self.d_pref_pos.upload_at(0, &[start as i32], &self.st)?;
        let fr = d_frames.at::<i32>(start * width);
        launched(
            unsafe {
                ffi::dr_embed(
                    h(&p.emb),
                    std::ptr::null(),
                    fr,
                    width as i32,
                    self.model.codebook_emb.bf16(),
                    f(&p.x),
                    n as i32,
                    d as i32,
                    cfg.semantic_start_token_id,
                    cfg.semantic_end_token_id,
                    cfg.eos_token_id,
                    dc.vocab_size as i32,
                    dc.vq_backbone_codebooks as i32,
                    s,
                )
            },
            "embed",
        )?;

        for (li, layer) in self.model.layers.iter().enumerate() {
            self.rmsnorm(f(&p.x), &layer.input_norm, std::ptr::null_mut(), h(&p.hn_b), n, eps)?;
            match &layer.mixer {
                Mixer::Linear(la) => {
                    let nk = tc.linear_num_key_heads;
                    let nv = tc.linear_num_value_heads;
                    let conv_c = 2 * nk * tc.linear_key_head_dim + nv * tc.linear_value_head_dim;
                    let vd = nv * tc.linear_value_head_dim;
                    self.gemm_q8(h(&p.hn_b), &la.in_proj, f(&p.proj), n, 0.0)?;
                    let ps = la.in_proj.n as i32;
                    launched(
                        unsafe {
                            ffi::dr_conv(f(&p.proj), ps, la.conv_w.f32(), f(self.conv_state[li].as_ref().unwrap()), f(&p.conv_out), n as i32, conv_c as i32, s)
                        },
                        "conv",
                    )?;
                    launched(
                        unsafe {
                            ffi::dr_gdn(
                                f(&p.conv_out),
                                f(&p.proj),
                                ps,
                                conv_c as i32,
                                (conv_c + vd) as i32,
                                (conv_c + vd + nv) as i32,
                                la.a_log.f32(),
                                la.dt_bias.f32(),
                                la.norm.f32(),
                                f(self.rec_state[li].as_ref().unwrap()),
                                f(&p.la_o),
                                n as i32,
                                nk as i32,
                                nv as i32,
                                tc.linear_key_head_dim as i32,
                                tc.linear_value_head_dim as i32,
                                eps,
                                s,
                            )
                        },
                        "gdn",
                    )?;
                    launched(unsafe { ffi::dr_f32_to_bf16(f(&p.la_o), h(&p.o_b), n * vd, s) }, "to_bf16")?;
                    self.gemm_q8(h(&p.o_b), &la.out_proj, f(&p.x), n, 1.0)?;
                }
                Mixer::Full(fa) => {
                    let (hh, hkv, hd) = (tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim);
                    let (kc, vc) = self.kv[li].as_ref().unwrap();
                    self.attention_prefill(
                        &fa.qkv, &fa.o, &fa.q_norm, &fa.k_norm, &self.model.text_inv_freq, false, kc, vc, hh, hkv, hd, eps, h(&p.hn_b), f(&p.x), n,
                    )?;
                }
            }
            // MoE
            self.rmsnorm(f(&p.x), &layer.post_norm, std::ptr::null_mut(), h(&p.hn_b), n, eps)?;
            self.moe_prefill(layer, n)?;
        }
        self.rmsnorm(f(&p.x), &self.model.final_norm, f(&p.h_all), std::ptr::null_mut(), n, eps)?;

        // Tower over this chunk's rows, except the prompt's last row (the first decode step's).
        let m = if start + n == rows { n - 1 } else { n };
        if m > 0 {
            self.tower_prefill(d_frames, start, m)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill(
        &self, qkv_w: &Q8, o_w: &Q8, q_norm: &Dense, k_norm: &Dense, inv_freq: &Dense, interleaved: bool, kc: &DevBuf, vc: &DevBuf,
        hh: usize, hkv: usize, hd: usize, eps: f32, input_b: *const u16, resid: *mut f32, n: usize,
    ) -> Result<()> {
        let p = &self.pre;
        let s = self.s();
        let rot = inv_freq.shape[0] * 2;
        self.gemm_q8(input_b, qkv_w, f(&p.qkv), n, 0.0)?;
        let rs = qkv_w.n as i32;
        launched(
            unsafe {
                ffi::dr_qk_norm_rope(
                    f(&p.qkv), rs, n as i32, hh as i32, hkv as i32, hd as i32, q_norm.f32(), k_norm.f32(), inv_freq.f32(), rot as i32,
                    interleaved as i32, i(&self.d_pref_pos), 0, f(&p.q), h(kc), h(vc), eps, s,
                )
            },
            "qk_norm_rope",
        )?;
        launched(
            unsafe {
                ffi::dr_attn_prefill(
                    f(&p.q), h(kc), h(vc), p.qkv.at::<f32>(hh * hd), rs, n as i32, hh as i32, hkv as i32, hd as i32,
                    i(&self.d_pref_pos), 0, f(&p.attn_o), s,
                )
            },
            "attn_prefill",
        )?;
        launched(unsafe { ffi::dr_f32_to_bf16(f(&p.attn_o), h(&p.o_b), n * hh * hd, s) }, "to_bf16")?;
        self.gemm_q8(h(&p.o_b), o_w, resid, n, 1.0)
    }

    fn moe_prefill(&self, layer: &crate::model::Layer, n: usize) -> Result<()> {
        let tc = &self.model.cfg.text_config;
        let p = &self.pre;
        let s = self.s();
        let d = tc.hidden_size;
        let e = tc.num_experts;
        let topk = tc.num_experts_per_tok;
        let mi = tc.moe_intermediate_size;
        let si = tc.shared_expert_intermediate_size;

        self.blas.gemm(h(&p.hn_b), layer.router.bf16(), f(&p.rlog), n, e + 1, d, 0.0)?;
        launched(
            unsafe { ffi::dr_moe_topk(f(&p.rlog), n as i32, e as i32, topk as i32, (e + 1) as i32, i(&p.idx), f(&p.wts), f(&p.sgate), s) },
            "moe_topk",
        )?;
        // Shared expert.
        self.gemm_q8(h(&p.hn_b), &layer.shared_gate_up, f(&p.shh), n, 0.0)?;
        launched(unsafe { ffi::dr_swiglu_rows(f(&p.shh), h(&p.sh_act_b), n as i32, si as i32, s) }, "swiglu")?;
        self.gemm_q8(h(&p.sh_act_b), &layer.shared_down, f(&p.sh_out), n, 0.0)?;

        // Group (row, k) pairs by expert.
        p.counts.zero(&self.st)?;
        launched(unsafe { ffi::dr_moe_count(i(&p.idx), (n * topk) as i32, i(&p.counts), s) }, "moe_count")?;
        let counts: Vec<i32> = p.counts.download(e, &self.st)?;
        let mut offsets = vec![0i32; e];
        let mut acc = 0;
        for (k, c) in counts.iter().enumerate() {
            offsets[k] = acc;
            acc += c;
        }
        p.offsets.upload_at(0, &offsets, &self.st)?;
        p.fill.zero(&self.st)?;
        launched(
            unsafe {
                ffi::dr_moe_permute(i(&p.idx), n as i32, topk as i32, i(&p.offsets), i(&p.fill), i(&p.slot_of), h(&p.hn_b), h(&p.xp), d as i32, s)
            },
            "moe_permute",
        )?;
        let (gq, gs) = layer.gate_up.expert(0);
        let (dq, ds) = layer.down.expert(0);
        launched(
            unsafe {
                ffi::dr_moe_prefill(gq, gs, dq, ds, h(&p.xp), i(&p.offsets), i(&p.counts), h(&p.ap), f(&p.yp), e as i32, d as i32, mi as i32, s)
            },
            "moe_prefill",
        )?;
        launched(
            unsafe {
                ffi::dr_moe_combine(f(&p.x), f(&p.yp), i(&p.slot_of), f(&p.wts), f(&p.sgate), f(&p.sh_out), n as i32, topk as i32, d as i32, s)
            },
            "moe_combine",
        )
    }

    /// Teacher-forced tower over prompt rows [start, start + m): each row's labels are the
    /// next row's frame, and its first-layer extra input is the row's own frame.
    fn tower_prefill(&self, d_frames: &DevBuf, start: usize, m: usize) -> Result<()> {
        let cfg = &self.model.cfg;
        let dc = &cfg.audio_decoder_config;
        let p = &self.pre;
        let s = self.s();
        let d = dc.dim;
        let width = cfg.frame_width();
        let eps = dc.norm_eps;
        let ns = dc.shared_enrichment_layers;
        let labels = d_frames.at::<i32>((start + 1) * width);
        let inputs = d_frames.at::<i32>(start * width);
        let mut hin = f(&p.h_all);
        for (t, tl) in self.model.tower.iter().enumerate() {
            let code_col = if t < ns { 1 } else { 1 + (t - ns) };
            launched(
                unsafe {
                    ffi::dr_tower_in(
                        hin,
                        labels,
                        width as i32,
                        code_col as i32,
                        if t == 0 { inputs } else { std::ptr::null() },
                        width as i32,
                        tl.tok_emb.bf16(),
                        self.model.codebook_emb.bf16(),
                        tl.enorm.f32(),
                        tl.hnorm.f32(),
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        h(&p.tin_b),
                        m as i32,
                        d as i32,
                        cfg.semantic_start_token_id,
                        cfg.semantic_end_token_id,
                        dc.vocab_size as i32,
                        dc.num_codebooks as i32,
                        dc.vq_backbone_codebooks as i32,
                        eps,
                        s,
                    )
                },
                "tower_in",
            )?;
            self.gemm_q8(h(&p.tin_b), &tl.eh_proj, f(&p.mixed), m, 0.0)?;
            self.rmsnorm(f(&p.mixed), &tl.input_norm, std::ptr::null_mut(), h(&p.an_b), m, eps)?;
            let (kc, vc) = &self.tkv[t];
            self.attention_prefill(
                &tl.qkv, &tl.o, &tl.q_norm, &tl.k_norm, &self.model.tower_inv_freq, true, kc, vc, dc.n_head, dc.n_local_heads, dc.head_dim,
                eps, h(&p.an_b), f(&p.mixed), m,
            )?;
            // mixed now holds r = attn + mixed.
            self.rmsnorm(f(&p.mixed), &tl.post_norm, std::ptr::null_mut(), h(&p.an_b), m, eps)?;
            self.gemm_q8(h(&p.an_b), &tl.gate_up, f(&p.gu), m, 0.0)?;
            launched(unsafe { ffi::dr_swiglu_rows(f(&p.gu), h(&p.act_b), m as i32, dc.intermediate_size as i32, s) }, "swiglu")?;
            self.gemm_q8(h(&p.act_b), &tl.down, f(&p.mixed), m, 1.0)?;
            self.rmsnorm(f(&p.mixed), &tl.final_norm, f(&p.th), std::ptr::null_mut(), m, eps)?;
            hin = f(&p.th);
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------ decode

    /// Record one decode step: tower + sampling at P, frame bookkeeping, backbone at P + 1.
    /// Ops are collected first so every weight-streaming kernel can prefetch the weights of
    /// the next one into L2 (the last op prefetches the next step's first).
    fn record_step(&self) -> Result<()> {
        let mut ops: Vec<Op<'_>> = Vec::new();
        self.ops_tower(&mut ops);
        let cfg = &self.model.cfg;
        let width = cfg.frame_width();
        let s = self.s();
        ops.push(Op::plain(move |_, _| {
            launched(
                unsafe {
                    ffi::dr_frame_end(
                        i(&self.d_next_frame),
                        i(&self.d_in_frame),
                        i(&self.d_ras),
                        RAS_WINDOW as i32,
                        i(&self.d_pos),
                        self.d_step.ptr::<u64>(),
                        i(&self.d_hist),
                        i(&self.d_hist_len),
                        self.max_len as i32,
                        width as i32,
                        s,
                    )
                },
                "frame_end",
            )
        }));
        self.ops_backbone(&mut ops);
        let first = ops.iter().find_map(|o| o.weights);
        for k in 0..ops.len() {
            let next = ops[k + 1..].iter().find_map(|o| o.weights).or(first);
            let (p, n) = if GEMV_PREFETCH { next.unwrap_or((std::ptr::null(), 0)) } else { (std::ptr::null(), 0) };
            (ops[k].run)(p as *const std::ffi::c_char, n)?;
        }
        Ok(())
    }

    fn op_gemv_q8<'a>(&'a self, w: &'a Q8, row0: usize, n: usize, x: *const f32, norm: Option<(&'a Dense, f32)>, res: *const f32, y: *mut f32) -> Op<'a> {
        let (q, sc) = w.rows(row0);
        let (nw, eps) = norm_args(norm);
        let k = w.k;
        let st = self.s();
        Op::weights(q as *const u8, n * k, move |pf, pfb| {
            launched(unsafe { ffi::dr_gemv_q8(q, sc, x, res, y, n as i32, k as i32, nw, eps, pf, pfb, st) }, "gemv_q8")
        })
    }

    fn op_swiglu<'a>(&'a self, w: &'a Q8, x: *const f32, norm: Option<(&'a Dense, f32)>, y: *mut f32) -> Op<'a> {
        let (q, sc) = w.rows(0);
        let (nw, eps) = norm_args(norm);
        let (n, k) = (w.n, w.k);
        let st = self.s();
        Op::weights(q as *const u8, n * k, move |pf, pfb| {
            launched(unsafe { ffi::dr_gemv_q8_swiglu(q, sc, x, y, (n / 2) as i32, k as i32, nw, eps, pf, pfb, st) }, "gemv_swiglu")
        })
    }

    fn ops_tower<'a>(&'a self, ops: &mut Vec<Op<'a>>) {
        let cfg = &self.model.cfg;
        let dc = &cfg.audio_decoder_config;
        let b = &self.dec;
        let s = self.s();
        let d = dc.dim;
        let eps = dc.norm_eps;
        let teps = cfg.text_config.rms_norm_eps;
        let ns = dc.shared_enrichment_layers;
        let cb = dc.vocab_size;
        let n_cand = cfg.n_candidates();
        let prm = self.d_params.ptr::<c_void>() as *const c_void;
        let step = self.d_step.ptr::<u64>() as *const u64;
        let final_norm = &self.model.final_norm;
        let lm = &self.model.lm_head_cand;

        // b.x holds the backbone residual; its final norm is fused into the consumers.
        ops.push(Op::weights(lm.bf16() as *const u8, n_cand * d * 2, move |pf, pfb| {
            launched(
                unsafe { ffi::dr_gemv_bf16(lm.bf16(), f(&b.x), f(&b.main_logits), n_cand as i32, d as i32, final_norm.f32(), teps, pf, pfb, s) },
                "lm_head",
            )
        }));
        ops.push(Op::plain(move |_, _| {
            launched(
                unsafe {
                    ffi::dr_sample_main(
                        f(&b.main_logits),
                        n_cand as i32,
                        prm,
                        i(&self.d_ras),
                        RAS_WINDOW as i32,
                        step,
                        cfg.eos_token_id,
                        cfg.semantic_start_token_id,
                        cb as i32,
                        i(&self.d_next_frame),
                        s,
                    )
                },
                "sample_main",
            )
        }));
        let rot = self.model.tower_inv_freq.shape[0] * 2;
        for (t, tl) in self.model.tower.iter().enumerate() {
            let code_col = if t < ns { 1 } else { 1 + (t - ns) };
            // h is the previous layer's (or the backbone's) un-normalised output.
            let (hin, pre_norm) = if t == 0 { (f(&b.x), final_norm) } else { (f(&b.r2), &self.model.tower[t - 1].final_norm) };
            let in_frame = if t == 0 { i(&self.d_in_frame) } else { std::ptr::null_mut() };
            ops.push(Op::plain(move |_, _| {
                launched(
                    unsafe {
                        ffi::dr_tower_in(
                            hin,
                            i(&self.d_next_frame),
                            0,
                            code_col as i32,
                            in_frame,
                            0,
                            tl.tok_emb.bf16(),
                            self.model.codebook_emb.bf16(),
                            tl.enorm.f32(),
                            tl.hnorm.f32(),
                            pre_norm.f32(),
                            f(&b.tin),
                            std::ptr::null_mut(),
                            1,
                            d as i32,
                            cfg.semantic_start_token_id,
                            cfg.semantic_end_token_id,
                            cb as i32,
                            dc.num_codebooks as i32,
                            dc.vq_backbone_codebooks as i32,
                            eps,
                            s,
                        )
                    },
                    "tower_in",
                )
            }));
            ops.push(self.op_gemv_q8(&tl.eh_proj, 0, d, f(&b.tin), None, std::ptr::null(), f(&b.mixed)));
            let (kc, vc) = &self.tkv[t];
            self.ops_attention(
                ops, &tl.qkv, &tl.o, &tl.q_norm, &tl.k_norm, rot, true, kc, vc, dc.n_head, dc.n_local_heads, dc.head_dim, eps, f(&b.mixed), &tl.input_norm,
                f(&b.mixed), f(&b.r),
            );
            ops.push(self.op_swiglu(&tl.gate_up, f(&b.r), Some((&tl.post_norm, eps)), f(&b.ff_act)));
            ops.push(self.op_gemv_q8(&tl.down, 0, d, f(&b.ff_act), None, f(&b.r), f(&b.r2)));
            if t >= ns {
                let r = t - ns;
                let logits = b.head_logits.at::<f32>(r * cb);
                ops.push(self.op_gemv_q8(&self.model.heads, r * cb, cb, f(&b.r2), Some((&tl.final_norm, eps)), std::ptr::null(), logits));
                ops.push(Op::plain(move |_, _| {
                    launched(
                        unsafe { ffi::dr_sample_head(logits, cb as i32, prm, (2 + r) as i32, step, i(&self.d_next_frame), (2 + r) as i32, s) },
                        "sample_head",
                    )
                }));
            }
        }
    }

    /// q/k norm + rope + cache write + attention + gated output projection (+ residual) for
    /// one decode row: y = res + o_proj(attention(norm(input))).
    #[allow(clippy::too_many_arguments)]
    fn ops_attention<'a>(
        &'a self, ops: &mut Vec<Op<'a>>, qkv_w: &'a Q8, o_w: &'a Q8, q_norm: &'a Dense, k_norm: &'a Dense, rot: usize, interleaved: bool, kc: &'a DevBuf,
        vc: &'a DevBuf, hh: usize, hkv: usize, hd: usize, eps: f32, input: *const f32, in_norm: &'a Dense, res: *const f32, y: *mut f32,
    ) {
        let b = &self.dec;
        let s = self.s();
        let inv = if interleaved { &self.model.tower_inv_freq } else { &self.model.text_inv_freq };
        ops.push(self.op_gemv_q8(qkv_w, 0, qkv_w.n, input, Some((in_norm, eps)), std::ptr::null(), f(&b.qkv)));
        let n = qkv_w.n;
        ops.push(Op::plain(move |_, _| {
            launched(
                unsafe {
                    ffi::dr_qk_norm_rope(
                        f(&b.qkv), n as i32, 1, hh as i32, hkv as i32, hd as i32, q_norm.f32(), k_norm.f32(), inv.f32(), rot as i32, interleaved as i32,
                        i(&self.d_pos), 0, f(&b.q), h(kc), h(vc), eps, s,
                    )
                },
                "qk_norm_rope",
            )?;
            launched(
                unsafe {
                    ffi::dr_attn_decode(f(&b.q), h(kc), h(vc), b.qkv.at::<f32>(hh * hd), hh as i32, hkv as i32, hd as i32, i(&self.d_pos), 0, f(&b.part), f(&b.attn_o), s)
                },
                "attn_decode",
            )
        }));
        ops.push(self.op_gemv_q8(o_w, 0, o_w.n, f(&b.attn_o), None, res, y));
    }

    fn ops_backbone<'a>(&'a self, ops: &mut Vec<Op<'a>>) {
        let cfg = &self.model.cfg;
        let tc = &cfg.text_config;
        let dc = &cfg.audio_decoder_config;
        let b = &self.dec;
        let s = self.s();
        let d = tc.hidden_size;
        let eps = tc.rms_norm_eps;
        ops.push(Op::plain(move |_, _| {
            launched(
                unsafe {
                    ffi::dr_embed(
                        std::ptr::null(),
                        self.model.cand_table.ptr::<u16>(),
                        i(&self.d_in_frame),
                        0,
                        self.model.codebook_emb.bf16(),
                        f(&b.x),
                        1,
                        d as i32,
                        cfg.semantic_start_token_id,
                        cfg.semantic_end_token_id,
                        cfg.eos_token_id,
                        dc.vocab_size as i32,
                        dc.vq_backbone_codebooks as i32,
                        s,
                    )
                },
                "embed",
            )
        }));
        let rot = self.model.text_inv_freq.shape[0] * 2;
        for (li, layer) in self.model.layers.iter().enumerate() {
            let in_norm = Some((&layer.input_norm, eps));
            match &layer.mixer {
                Mixer::Linear(la) => {
                    let nk = tc.linear_num_key_heads;
                    let nv = tc.linear_num_value_heads;
                    let conv_c = 2 * nk * tc.linear_key_head_dim + nv * tc.linear_value_head_dim;
                    let vd = nv * tc.linear_value_head_dim;
                    ops.push(self.op_gemv_q8(&la.in_proj, 0, la.in_proj.n, f(&b.x), in_norm, std::ptr::null(), f(&b.proj)));
                    let ps = la.in_proj.n as i32;
                    let conv_state = f(self.conv_state[li].as_ref().unwrap());
                    let rec_state = f(self.rec_state[li].as_ref().unwrap());
                    ops.push(Op::plain(move |_, _| {
                        launched(unsafe { ffi::dr_conv(f(&b.proj), ps, la.conv_w.f32(), conv_state, f(&b.conv_out), 1, conv_c as i32, s) }, "conv")?;
                        launched(
                            unsafe {
                                ffi::dr_gdn(
                                    f(&b.conv_out),
                                    f(&b.proj),
                                    ps,
                                    conv_c as i32,
                                    (conv_c + vd) as i32,
                                    (conv_c + vd + nv) as i32,
                                    la.a_log.f32(),
                                    la.dt_bias.f32(),
                                    la.norm.f32(),
                                    rec_state,
                                    f(&b.la_o),
                                    1,
                                    nk as i32,
                                    nv as i32,
                                    tc.linear_key_head_dim as i32,
                                    tc.linear_value_head_dim as i32,
                                    eps,
                                    s,
                                )
                            },
                            "gdn",
                        )
                    }));
                    ops.push(self.op_gemv_q8(&la.out_proj, 0, d, f(&b.la_o), None, f(&b.x), f(&b.x)));
                }
                Mixer::Full(fa) => {
                    let (kc, vc) = self.kv[li].as_ref().unwrap();
                    self.ops_attention(
                        ops, &fa.qkv, &fa.o, &fa.q_norm, &fa.k_norm, rot, false, kc, vc, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, eps,
                        f(&b.x), &layer.input_norm, f(&b.x), f(&b.x),
                    );
                }
            }
            let e = tc.num_experts;
            let topk = tc.num_experts_per_tok;
            let (gq, gs) = layer.gate_up.expert(0);
            let (dq, ds) = layer.down.expert(0);
            let (gu, dn) = (&layer.gate_up, &layer.down);
            let router = &layer.router;
            let post = &layer.post_norm;
            ops.push(Op::weights(router.bf16() as *const u8, (e + 1) * d * 2, move |_, _| {
                // The router warms L2 with its selected experts instead of a generic prefetch.
                launched(
                    unsafe {
                        ffi::dr_router_topk(
                            router.bf16(),
                            f(&b.x),
                            post.f32(),
                            eps,
                            f(&b.rlog),
                            e as i32,
                            topk as i32,
                            d as i32,
                            i(&b.idx),
                            f(&b.wts),
                            f(&b.sgate),
                            b.counter.ptr::<u32>(),
                            if ROUTER_PREFETCH { gq as *const std::ffi::c_char } else { std::ptr::null() },
                            gs as *const std::ffi::c_char,
                            dq as *const std::ffi::c_char,
                            ds as *const std::ffi::c_char,
                            gu.n * gu.k / 2,
                            gu.n * (gu.k / 64) * 4,
                            dn.n * dn.k / 2,
                            dn.n * (dn.k / 64) * 4,
                            s,
                        )
                    },
                    "router_topk",
                )
            }));
            let (sgq, sgs) = layer.shared_gate_up.rows(0);
            let (sdq, sds) = layer.shared_down.rows(0);
            let mi = tc.moe_intermediate_size;
            ops.push(Op::plain(move |pf, pfb| {
                launched(
                    unsafe {
                        ffi::dr_moe_decode(
                            gq,
                            gs,
                            dq,
                            ds,
                            sgq,
                            sgs,
                            sdq,
                            sds,
                            f(&b.x),
                            i(&b.idx),
                            f(&b.wts),
                            f(&b.sgate),
                            f(&b.x),
                            f(&b.act),
                            f(&b.x),
                            d as i32,
                            mi as i32,
                            topk as i32,
                            post.f32(),
                            eps,
                            pf,
                            pfb,
                            s,
                        )
                    },
                    "moe_decode",
                )
            }));
        }
    }

    /// Replay one decode step and return the sampled frame.
    pub fn step(&mut self) -> Result<(Vec<i32>, Option<StepTrace>)> {
        let width = self.model.cfg.frame_width();
        let g = self.graph.as_ref().expect("prefill first");
        g.launch(&self.st)?;
        unsafe {
            ffi::dr_d2h(self.frame_host.ptr() as *mut c_void, self.d_in_frame.ptr::<c_void>(), width * I32, self.s());
        }
        self.st.sync()?;
        let frame = self.frame_host.as_slice().to_vec();
        let trace = if self.trace_logits {
            let dc = &self.model.cfg.audio_decoder_config;
            Some(StepTrace {
                main_logits: self.dec.main_logits.download(self.model.cfg.n_candidates(), &self.st)?,
                head_logits: self.dec.head_logits.download((dc.num_codebooks - 1) * dc.vocab_size, &self.st)?,
            })
        } else {
            None
        };
        Ok((frame, trace))
    }


    /// Generate frames until end-of-speech or `max_new` frames.
    pub fn generate(
        &mut self, prompt: &[i32], params: SampleParams, max_new: usize, mut on_frame: impl FnMut(&[i32]),
    ) -> Result<Vec<Vec<i32>>> {
        let width = self.model.cfg.frame_width();
        let rows = prompt.len() / width;
        self.prefill(prompt, params)?;
        let room = self.max_len - rows;
        let mut out = Vec::new();
        for _ in 0..max_new.min(room) {
            if self.stop.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Cancelled.into());
            }
            let (frame, _) = self.step()?;
            if frame[0] == self.model.cfg.eos_token_id {
                break;
            }
            on_frame(&frame);
            out.push(frame);
        }
        Ok(out)
    }
}
