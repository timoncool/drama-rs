//! Drama backbone, gated DeltaNet, routed experts and audio tower on MLX.
use crate::{
    apple::{cat, rms, row, silu, slice, swiglu, vec_f32},
    model::{linear, tensor, Config, Layer, Model, TowerLayer},
};
use anyhow::{ensure, Context, Result};
use mlx_rs::{
    ops::{self, indexing::TryIndexOp},
    Array, Dtype,
};

pub const RAS_WINDOW: usize = 16;
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
        Self {
            temperature: 1.0,
            top_k: 30,
            top_p: 0.9,
            pad: 0,
            seed: 0,
        }
    }
}
pub struct StepTrace {
    pub main_logits: Vec<f32>,
    pub head_logits: Vec<f32>,
}

#[derive(Default)]
pub(crate) struct KvCache {
    pub k: Option<Array>,
    pub v: Option<Array>,
}
impl KvCache {
    pub(crate) fn append(&mut self, k: Array, v: Array, axis: i32) -> Result<(Array, Array)> {
        let k = if let Some(old) = &self.k {
            cat(&[old, &k], axis)?
        } else {
            k
        };
        let v = if let Some(old) = &self.v {
            cat(&[old, &v], axis)?
        } else {
            v
        };
        k.eval()?;
        v.eval()?;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}
#[derive(Default)]
struct Recurrent {
    conv: Option<Array>,
    state: Option<Array>,
}

pub struct Engine {
    pub model: Model,
    pub max_len: usize,
    pub trace_logits: bool,
    chunk: usize,
    kv: Vec<KvCache>,
    tkv: Vec<KvCache>,
    recurrent: Vec<Recurrent>,
    hidden: Option<Array>,
    input: Vec<i32>,
    history: Vec<i32>,
    params: SampleParams,
    pos: usize,
    step: u64,
}

/// Gated query attention. The checkpoint stores [query | gate | key | value].
#[allow(clippy::too_many_arguments)]
fn attention(
    x: &Array,
    ts: &std::collections::HashMap<String, Array>,
    ls: &std::collections::HashMap<String, crate::apple::Quantized>,
    prefix: &str,
    heads: usize,
    kvheads: usize,
    hd: usize,
    eps: f32,
    theta: f64,
    rot: usize,
    traditional: bool,
    pos: usize,
    cache: &mut KvCache,
) -> Result<Array> {
    let n = x.shape()[0];
    let p = linear(ls, &format!("{prefix}qkv"))?.forward(x)?;
    let qwidth = (heads * hd) as i32;
    let kwidth = (kvheads * hd) as i32;
    let q = slice(&p, 1, 0, qwidth)?.reshape(&[n, heads as i32, hd as i32])?;
    let gate = slice(&p, 1, qwidth, qwidth)?;
    let k = slice(&p, 1, 2 * qwidth, kwidth)?.reshape(&[n, kvheads as i32, hd as i32])?;
    let v = slice(&p, 1, 2 * qwidth + kwidth, kwidth)?
        .reshape(&[n, kvheads as i32, hd as i32])?
        .transpose_axes(&[1, 0, 2])?
        .expand_dims(0)?;
    let q = rms(&q, tensor(ts, &format!("{prefix}q_norm"))?, eps)?
        .transpose_axes(&[1, 0, 2])?
        .expand_dims(0)?;
    let k = rms(&k, tensor(ts, &format!("{prefix}k_norm"))?, eps)?
        .transpose_axes(&[1, 0, 2])?
        .expand_dims(0)?;
    let q = mlx_rs::fast::rope(
        q,
        rot as i32,
        traditional,
        theta as f32,
        1.0,
        pos as i32,
        None,
    )?;
    let k = mlx_rs::fast::rope(
        k,
        rot as i32,
        traditional,
        theta as f32,
        1.0,
        pos as i32,
        None,
    )?;
    let (k, v) = cache.append(k, v, 2)?;
    let y = mlx_rs::fast::scaled_dot_product_attention(
        q,
        k,
        v,
        1.0 / (hd as f32).sqrt(),
        Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal),
        None,
    )?;
    let y = y
        .squeeze_axes(&[0])?
        .transpose_axes(&[1, 0, 2])?
        .reshape(&[n, qwidth])?
        .multiply(ops::sigmoid(gate)?)?;
    linear(ls, &format!("{prefix}o"))?.forward(&y)
}

fn delta_net(x: &Array, layer: &Layer, cfg: &Config, rec: &mut Recurrent) -> Result<Array> {
    let tc = &cfg.text_config;
    let (nk, nv, dk, dv) = (
        tc.linear_num_key_heads as i32,
        tc.linear_num_value_heads as i32,
        tc.linear_key_head_dim as i32,
        tc.linear_value_head_dim as i32,
    );
    let cc = 2 * nk * dk + nv * dv;
    let n = x.shape()[0];
    let p = linear(&layer.linears, "la.in_proj")?.forward(x)?;
    let hist = match &rec.conv {
        Some(s) => s.clone(),
        None => Array::zeros::<f32>(&[(tc.linear_conv_kernel_dim - 1) as i32, cc])?,
    };
    let all = cat(&[&hist, &slice(&p, 1, 0, cc)?], 0)?;
    let kw = tc.linear_conv_kernel_dim as i32;
    let w = tensor(&layer.tensors, "la.conv_w")?;
    let mut conv = Array::zeros::<f32>(&[n, cc])?;
    for j in 0..kw {
        conv = conv.add(slice(&all, 0, j, n)?.multiply(w.try_index((.., j))?)?)?;
    }
    let conv = silu(&conv)?;
    rec.conv = Some(slice(&all, 0, n, kw - 1)?);
    rec.conv.as_ref().unwrap().eval()?;
    let map = Array::from_slice(&(0..nv).map(|h| h / (nv / nk)).collect::<Vec<_>>(), &[nv]);
    let mut state = match &rec.state {
        Some(s) => s.clone(),
        None => Array::zeros::<f32>(&[nv, dk, dv])?,
    };
    let mut outputs = Vec::new();
    for t in 0..n {
        let cr = slice(&conv, 0, t, 1)?;
        let q = slice(&cr, 1, 0, nk * dk)?
            .reshape(&[nk, dk])?
            .take_axis(&map, 0)?;
        let k = slice(&cr, 1, nk * dk, nk * dk)?
            .reshape(&[nk, dk])?
            .take_axis(&map, 0)?;
        let v = slice(&cr, 1, 2 * nk * dk, nv * dv)?.reshape(&[nv, dv])?;
        let q = q
            .divide(
                q.square()?
                    .sum_axis(1, true)?
                    .add(Array::from(1e-6f32))?
                    .sqrt()?,
            )?
            .divide(Array::from((dk as f32).sqrt()))?;
        let k = k.divide(
            k.square()?
                .sum_axis(1, true)?
                .add(Array::from(1e-6f32))?
                .sqrt()?,
        )?;
        let pr = row(&p, t)?;
        let z = slice(&pr, 0, cc, nv * dv)?.reshape(&[nv, dv])?;
        let beta = ops::sigmoid(slice(&pr, 0, cc + nv * dv, nv)?)?.reshape(&[nv, 1])?;
        let a = slice(&pr, 0, cc + nv * dv + nv, nv)?.add(tensor(&layer.tensors, "la.dt_bias")?)?;
        let decay = tensor(&layer.tensors, "la.A_log")?
            .exp()?
            .negative()?
            .multiply(ops::logaddexp(&a, Array::from(0.0f32))?)?
            .exp()?
            .reshape(&[nv, 1, 1])?;
        state = state.multiply(decay)?;
        let pred = state.multiply(k.expand_dims(-1)?)?.sum_axis(1, false)?;
        let delta = v.subtract(pred)?.multiply(beta)?;
        state = state.add(k.expand_dims(-1)?.multiply(delta.expand_dims(1)?)?)?;
        let out = state.multiply(q.expand_dims(-1)?)?.sum_axis(1, false)?;
        let out = rms(&out, tensor(&layer.tensors, "la.norm")?, tc.rms_norm_eps)?
            .multiply(silu(&z)?)?
            .reshape(&[1, nv * dv])?;
        outputs.push(out);
        // Materialize every output in the batch as well as the new state. Otherwise
        // unevaluated early outputs retain a separate large recurrent state per token.
        if t % 16 == 15 || t + 1 == n {
            mlx_rs::transforms::eval(std::iter::once(&state).chain(outputs.iter().rev().take(16)))?;
        }
    }
    rec.state = Some(state);
    linear(&layer.linears, "la.out_proj")?.forward(&ops::concatenate(&outputs, 0)?)
}

fn moe(x: &Array, layer: &Layer, cfg: &Config) -> Result<Array> {
    let tc = &cfg.text_config;
    let n = x.shape()[0] as usize;
    let e = tc.num_experts;
    let logits = vec_f32(
        &x.matmul(
            tensor(&layer.tensors, "moe.router")?
                .as_dtype(Dtype::Float32)?
                .t(),
        )?,
    )?;
    let mut rows: Vec<Vec<i32>> = vec![Vec::new(); e];
    let mut weights: Vec<Vec<f32>> = vec![Vec::new(); e];
    for r in 0..n {
        let l = &logits[r * (e + 1)..r * (e + 1) + e];
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
        order.truncate(tc.num_experts_per_tok);
        let z: f32 = order.iter().map(|&i| (l[i] - l[order[0]]).exp()).sum();
        for &i in &order {
            rows[i].push(r as i32);
            weights[i].push((l[i] - l[order[0]]).exp() / z);
        }
    }
    let shared = linear(&layer.linears, "moe.shared_down")?.forward(&swiglu(
        &linear(&layer.linears, "moe.shared_gate_up")?.forward(x)?,
    )?)?;
    let sg: Vec<f32> = (0..n)
        .map(|r| 1.0 / (1.0 + (-logits[r * (e + 1) + e]).exp()))
        .collect();
    let mut y = shared.multiply(Array::from_slice(&sg, &[n as i32, 1]))?;
    for i in 0..e {
        if rows[i].is_empty() {
            continue;
        }
        let ids = Array::from_slice(&rows[i], &[rows[i].len() as i32]);
        let xi = x.take_axis(&ids, 0)?;
        let yi = layer
            .down
            .expert(i as i32)?
            .forward(&swiglu(&layer.gate_up.expert(i as i32)?.forward(&xi)?)?)?
            .multiply(Array::from_slice(
                &weights[i],
                &[weights[i].len() as i32, 1],
            ))?;
        y = ops::indexing::scatter_add_single(&y, &ids, yi.expand_dims(1)?, 0)?;
        if i % 16 == 15 {
            y.eval()?;
        }
    }
    Ok(y)
}

impl Engine {
    pub fn new(model: Model, max_len: usize, chunk: usize) -> Result<Self> {
        ensure!(
            max_len >= 2 && max_len <= i32::MAX as usize && chunk > 0,
            "max_len must be at least 2 and chunk must be positive"
        );
        let layers = model.cfg.text_config.num_hidden_layers;
        let tower = model.cfg.tower_layers();
        Ok(Self {
            model,
            max_len,
            chunk,
            trace_logits: false,
            kv: (0..layers).map(|_| KvCache::default()).collect(),
            tkv: (0..tower).map(|_| KvCache::default()).collect(),
            recurrent: (0..layers).map(|_| Recurrent::default()).collect(),
            hidden: None,
            input: Vec::new(),
            history: Vec::new(),
            params: SampleParams::default(),
            pos: 0,
            step: 0,
        })
    }
    fn semantic(&self, t: i32) -> bool {
        (self.model.cfg.semantic_start_token_id..=self.model.cfg.semantic_end_token_id).contains(&t)
    }
    fn embeddings(&self, frames: &[i32]) -> Result<Array> {
        let cfg = &self.model.cfg;
        let dc = &cfg.audio_decoder_config;
        let mut out = Vec::new();
        for frame in frames.chunks_exact(cfg.frame_width()) {
            ensure!(
                frame[0] >= 0 && frame[0] < self.model.embed_tokens.shape()[0],
                "prompt token out of vocabulary"
            );
            let mut v = row(&self.model.embed_tokens, frame[0])?.as_dtype(Dtype::Float32)?;
            if self.semantic(frame[0]) {
                for c in 1..=dc.vq_backbone_codebooks {
                    v = v.add(
                        self.model
                            .codebook_emb
                            .try_index((c as i32, frame[c + 1].clamp(0, dc.vocab_size as i32 - 1)))?
                            .as_dtype(Dtype::Float32)?,
                    )?;
                }
                v = v.divide(Array::from(((1 + dc.vq_backbone_codebooks) as f32).sqrt()))?;
            }
            out.push(v.expand_dims(0)?);
        }
        Ok(ops::concatenate(&out, 0)?)
    }
    fn backbone(&mut self, frames: &[i32], pos: usize) -> Result<Array> {
        let cfg = &self.model.cfg;
        let tc = &cfg.text_config;
        let mut x = self.embeddings(frames)?;
        for (i, l) in self.model.layers.iter().enumerate() {
            let xn = rms(&x, tensor(&l.tensors, "input_norm")?, tc.rms_norm_eps)?;
            let mixed = if cfg.is_linear(i) {
                delta_net(&xn, l, cfg, &mut self.recurrent[i])?
            } else {
                attention(
                    &xn,
                    &l.tensors,
                    &l.linears,
                    "attn.",
                    tc.num_attention_heads,
                    tc.num_key_value_heads,
                    tc.head_dim,
                    tc.rms_norm_eps,
                    tc.rope_parameters.rope_theta,
                    (tc.head_dim as f64 * tc.rope_parameters.partial_rotary_factor) as usize,
                    false,
                    pos,
                    &mut self.kv[i],
                )?
            };
            x = x.add(mixed)?;
            let xn = rms(&x, tensor(&l.tensors, "post_norm")?, tc.rms_norm_eps)?;
            x = x.add(moe(&xn, l, cfg)?)?;
            x.eval()?;
        }
        rms(&x, &self.model.final_norm, tc.rms_norm_eps)
    }
    fn tower_embed(
        &self,
        tl: &TowerLayer,
        labels: &[i32],
        inputs: Option<&[i32]>,
        col: usize,
    ) -> Result<Array> {
        let cfg = &self.model.cfg;
        let dc = &cfg.audio_decoder_config;
        let width = cfg.frame_width();
        let mut out = Vec::new();
        for (r, f) in labels.chunks_exact(width).enumerate() {
            let mut v = if self.semantic(f[0]) {
                row(
                    tensor(&tl.tensors, "tok_emb")?,
                    f[col].clamp(0, dc.vocab_size as i32 - 1),
                )?
                .as_dtype(Dtype::Float32)?
            } else {
                Array::zeros::<f32>(&[dc.dim as i32])?
            };
            if let Some(inputs) = inputs {
                let fi = &inputs[r * width..(r + 1) * width];
                if self.semantic(fi[0]) {
                    for c in 1 + dc.vq_backbone_codebooks..dc.num_codebooks {
                        v = v.add(
                            self.model
                                .codebook_emb
                                .try_index((
                                    c as i32,
                                    fi[c + 1].clamp(0, dc.vocab_size as i32 - 1),
                                ))?
                                .as_dtype(Dtype::Float32)?,
                        )?;
                    }
                }
            }
            out.push(v.expand_dims(0)?);
        }
        Ok(ops::concatenate(&out, 0)?)
    }
    fn tower_block(&mut self, t: usize, h: &Array, e: &Array, pos: usize) -> Result<Array> {
        let dc = &self.model.cfg.audio_decoder_config;
        let tl = &self.model.tower[t];
        let eh = cat(
            &[
                &rms(e, tensor(&tl.tensors, "enorm")?, dc.norm_eps)?,
                &rms(h, tensor(&tl.tensors, "hnorm")?, dc.norm_eps)?,
            ],
            1,
        )?;
        let mut x = linear(&tl.linears, "eh_proj")?.forward(&eh)?;
        let xn = rms(&x, tensor(&tl.tensors, "input_norm")?, dc.norm_eps)?;
        x = x.add(attention(
            &xn,
            &tl.tensors,
            &tl.linears,
            "",
            dc.n_head,
            dc.n_local_heads,
            dc.head_dim,
            dc.norm_eps,
            dc.rope_base,
            (dc.head_dim as f64 * dc.rotary_percent) as usize,
            true,
            pos,
            &mut self.tkv[t],
        )?)?;
        let xn = rms(&x, tensor(&tl.tensors, "post_norm")?, dc.norm_eps)?;
        x = x.add(
            linear(&tl.linears, "down")?
                .forward(&swiglu(&linear(&tl.linears, "gate_up")?.forward(&xn)?)?)?,
        )?;
        let out = rms(&x, tensor(&tl.tensors, "final_norm")?, dc.norm_eps)?;
        out.eval()?;
        Ok(out)
    }
    pub fn prefill(&mut self, frames: &[i32], params: SampleParams) -> Result<()> {
        let width = self.model.cfg.frame_width();
        let rows = frames.len() / width;
        ensure!(
            frames.len() % width == 0 && rows >= 2 && rows < self.max_len,
            "prompt must contain at least 2 complete rows and fit the context"
        );
        ensure!(
            params.temperature.is_finite()
                && params.temperature >= 0.0
                && params.top_p.is_finite()
                && params.top_p > 0.0
                && params.top_p <= 1.0,
            "invalid sampling parameters"
        );
        for k in self.kv.iter_mut().chain(&mut self.tkv) {
            *k = KvCache::default();
        }
        for r in &mut self.recurrent {
            *r = Recurrent::default();
        }
        self.hidden = None;
        self.params = params;
        self.pos = rows - 1;
        self.step = 0;
        self.history = frames
            .chunks_exact(width)
            .rev()
            .take(RAS_WINDOW)
            .map(|f| f[0])
            .collect();
        self.history.reverse();
        for start in (0..rows).step_by(self.chunk) {
            let n = self.chunk.min(rows - start);
            let h = self.backbone(&frames[start * width..(start + n) * width], start)?;
            let m = if start + n == rows { n - 1 } else { n };
            if m > 0 {
                let mut th = slice(&h, 0, 0, m as i32)?;
                for t in 0..self.model.tower.len() {
                    let ns = self.model.cfg.audio_decoder_config.shared_enrichment_layers;
                    let col = if t < ns { 1 } else { 1 + t - ns };
                    let e = self.tower_embed(
                        &self.model.tower[t],
                        &frames[(start + 1) * width..(start + m + 1) * width],
                        (t == 0).then_some(&frames[start * width..(start + m) * width]),
                        col,
                    )?;
                    th = self.tower_block(t, &th, &e, start)?;
                }
            }
            if start + n == rows {
                self.hidden = Some(slice(&h, 0, n as i32 - 1, 1)?);
            }
        }
        self.input = frames[(rows - 1) * width..].to_vec();
        Ok(())
    }
    pub fn step(&mut self) -> Result<(Vec<i32>, Option<StepTrace>)> {
        ensure!(self.pos + 1 < self.max_len, "context is full");
        let h = self
            .hidden
            .as_ref()
            .context("call prefill before step")?
            .clone();
        let logits = vec_f32(&h.matmul(self.model.lm_head_cand.as_dtype(Dtype::Float32)?.t())?)?;
        let cfg = &self.model.cfg;
        let mut c = sample(
            &logits,
            self.params,
            uniform(self.params.seed, 0, self.step),
        );
        let token = if c == 0 {
            cfg.eos_token_id
        } else {
            cfg.semantic_start_token_id + c as i32 - 1
        };
        if c != 0 && self.history.contains(&token) {
            c = sample(
                &logits,
                SampleParams {
                    temperature: 1.0,
                    top_k: 30,
                    top_p: 0.95,
                    ..self.params
                },
                uniform(self.params.seed, 1, self.step),
            );
        }
        let mut frame = vec![0; cfg.frame_width()];
        frame[0] = if c == 0 {
            cfg.eos_token_id
        } else {
            cfg.semantic_start_token_id + c as i32 - 1
        };
        frame[1] = (frame[0] - cfg.semantic_start_token_id)
            .clamp(0, cfg.audio_decoder_config.vocab_size as i32 - 1);
        let mut th = h;
        let mut head_logits = Vec::new();
        let ns = self.model.cfg.audio_decoder_config.shared_enrichment_layers;
        let cb = self.model.cfg.audio_decoder_config.vocab_size;
        for t in 0..self.model.tower.len() {
            let col = if t < ns { 1 } else { 1 + t - ns };
            let e = self.tower_embed(
                &self.model.tower[t],
                &frame,
                (t == 0).then_some(self.input.as_slice()),
                col,
            )?;
            th = self.tower_block(t, &th, &e, self.pos)?;
            if t >= ns {
                let r = t - ns;
                let l = vec_f32(
                    &self
                        .model
                        .heads
                        .rows((r * cb) as i32, cb as i32)?
                        .forward(&th)?,
                )?;
                frame[2 + r] = sample(
                    &l,
                    self.params,
                    uniform(self.params.seed, (2 + r) as u64, self.step),
                ) as i32;
                if self.trace_logits {
                    head_logits.extend(l);
                }
            }
        }
        self.pos += 1;
        self.step += 1;
        self.hidden = Some(self.backbone(&frame, self.pos)?);
        self.input = frame.clone();
        self.history.push(frame[0]);
        if self.history.len() > RAS_WINDOW {
            self.history.remove(0);
        }
        Ok((
            frame,
            self.trace_logits.then_some(StepTrace {
                main_logits: logits,
                head_logits,
            }),
        ))
    }
    pub fn generate(
        &mut self,
        prompt: &[i32],
        params: SampleParams,
        max_new: usize,
        mut on_frame: impl FnMut(&[i32]),
    ) -> Result<Vec<Vec<i32>>> {
        self.prefill(prompt, params)?;
        let mut out = Vec::new();
        for _ in 0..max_new.min(self.max_len - prompt.len() / self.model.cfg.frame_width()) {
            let (f, _) = self.step()?;
            if f[0] == self.model.cfg.eos_token_id {
                break;
            }
            on_frame(&f);
            out.push(f);
        }
        Ok(out)
    }
}

// Philox4x32-10 uses the same seed/subsequence/offset convention as the CUDA sampler.
fn uniform(seed: u64, stream: u64, step: u64) -> f32 {
    let mut c = [
        step as u32,
        (step >> 32) as u32,
        stream as u32,
        (stream >> 32) as u32,
    ];
    let mut k = [seed as u32, (seed >> 32) as u32];
    for _ in 0..10 {
        let a = 0xd2511f53u64 * c[0] as u64;
        let b = 0xcd9e8d57u64 * c[2] as u64;
        c = [
            (b >> 32) as u32 ^ c[1] ^ k[0],
            b as u32,
            (a >> 32) as u32 ^ c[3] ^ k[1],
            a as u32,
        ];
        k[0] = k[0].wrapping_add(0x9e3779b9);
        k[1] = k[1].wrapping_add(0xbb67ae85);
    }
    (c[0] as f32 + 1.0) * (1.0 / 4294967296.0)
}
fn sample(logits: &[f32], p: SampleParams, u: f32) -> usize {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let inv = 1.0 / p.temperature.max(1e-5);
    let weights: Vec<f32> = order
        .iter()
        .map(|&i| ((logits[i] - logits[order[0]]) * inv).exp())
        .collect();
    let z: f32 = weights.iter().sum();
    let mut cum = 0.0;
    let mut kept = Vec::new();
    for (rank, &w) in weights.iter().enumerate() {
        cum += w / z;
        if rank == 0 || ((p.top_k <= 0 || rank < p.top_k as usize) && cum <= p.top_p) {
            kept.push((order[rank], w));
        }
    }
    let total: f32 = kept.iter().map(|(_, w)| w).sum();
    let mut draw = u * total;
    for &(i, w) in &kept {
        draw -= w;
        if draw <= 0.0 {
            return i;
        }
    }
    kept.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apple::Quantized;
    use std::collections::HashMap;
    fn dense(shape: &[i32], phase: f32) -> Array {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        Array::from_slice(
            &(0..n)
                .map(|i| ((i as f32) * 0.017 + phase).sin() * 0.1)
                .collect::<Vec<_>>(),
            shape,
        )
    }
    fn q8(n: i32, k: i32) -> Result<Quantized> {
        let v: Vec<i8> = (0..n * k)
            .map(|i| ((i * 7 + i / k * 3) % 17 - 8) as i8)
            .collect();
        Quantized::q8(
            Array::from_slice(&v, &[n, k]),
            Array::ones::<f32>(&[n, k / 128])?.multiply(Array::from(0.002f32))?,
        )
    }
    fn q4(n: i32, k: i32) -> Result<Quantized> {
        let v: Vec<u8> = (0..2 * n * k / 2).map(|i| ((i * 7) % 256) as u8).collect();
        let sm: Vec<f32> = (0..2 * n * (k / 64)).flat_map(|_| [0.01, -0.07]).collect();
        Quantized::q4(
            Array::from_slice(&v, &[2, n, k / 2]),
            Array::from_slice(&sm, &[2, n, k / 64, 2]),
        )
    }
    fn model() -> Result<Model> {
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "text_config":{"hidden_size":128,"num_hidden_layers":2,"layer_types":["full_attention","linear_attention"],
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":256,"rms_norm_eps":1e-6,
                "rope_parameters":{"rope_theta":10000.0,"partial_rotary_factor":0.5},"num_experts":2,"num_experts_per_tok":1,
                "moe_intermediate_size":128,"shared_expert_intermediate_size":128,"linear_conv_kernel_dim":4,
                "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":1,"linear_num_value_heads":1},
            "audio_decoder_config":{"dim":128,"n_head":1,"n_local_heads":1,"head_dim":256,"intermediate_size":128,"norm_eps":1e-6,
                "num_codebooks":2,"vocab_size":8,"shared_enrichment_layers":1,"rope_base":10000.0,"rotary_percent":1.0,"vq_backbone_codebooks":1},
            "semantic_start_token_id":5,"semantic_end_token_id":12,"eos_token_id":3
        }))?;
        let mut layers = Vec::new();
        for i in 0..2 {
            let mut ts = HashMap::new();
            for n in ["input_norm", "post_norm"] {
                ts.insert(n.to_string(), Array::ones::<f32>(&[128])?);
            }
            ts.insert("moe.router".into(), dense(&[3, 128], 0.2));
            let mut ls = HashMap::new();
            ls.insert("moe.shared_gate_up".into(), q8(256, 128)?);
            ls.insert("moe.shared_down".into(), q8(128, 128)?);
            if i == 0 {
                for n in ["attn.q_norm", "attn.k_norm"] {
                    ts.insert(n.into(), Array::ones::<f32>(&[256])?);
                }
                ls.insert("attn.qkv".into(), q8(1024, 128)?);
                ls.insert("attn.o".into(), q8(128, 256)?);
            } else {
                ts.insert("la.conv_w".into(), dense(&[384, 4], 0.4));
                ts.insert("la.A_log".into(), Array::from_slice(&[0.2f32], &[1]));
                ts.insert("la.dt_bias".into(), Array::from_slice(&[0.1f32], &[1]));
                ts.insert("la.norm".into(), Array::ones::<f32>(&[128])?);
                ls.insert("la.in_proj".into(), q8(514, 128)?);
                ls.insert("la.out_proj".into(), q8(128, 128)?);
            }
            layers.push(Layer {
                tensors: ts,
                linears: ls,
                gate_up: q4(256, 128)?,
                down: q4(128, 128)?,
            });
        }
        let mut tower = Vec::new();
        for _ in 0..2 {
            let mut ts = HashMap::new();
            for n in ["enorm", "hnorm", "final_norm", "input_norm", "post_norm"] {
                ts.insert(n.into(), Array::ones::<f32>(&[128])?);
            }
            for n in ["q_norm", "k_norm"] {
                ts.insert(n.into(), Array::ones::<f32>(&[256])?);
            }
            ts.insert("tok_emb".into(), dense(&[8, 128], 0.3));
            let mut ls = HashMap::new();
            for (n, out, k) in [
                ("eh_proj", 128, 256),
                ("qkv", 1024, 128),
                ("o", 128, 256),
                ("gate_up", 256, 128),
                ("down", 128, 128),
            ] {
                ls.insert(n.into(), q8(out, k)?);
            }
            tower.push(TowerLayer {
                tensors: ts,
                linears: ls,
            });
        }
        Ok(Model {
            cfg,
            layers,
            tower,
            heads: q8(8, 128)?,
            codebook_emb: dense(&[2, 8, 128], 0.8),
            lm_head_cand: dense(&[9, 128], 0.6),
            final_norm: Array::ones::<f32>(&[128])?,
            embed_tokens: dense(&[16, 128], 0.1),
        })
    }
    #[test]
    fn prefill_chunking_and_reset_preserve_decode_logits() -> Result<()> {
        crate::apple::init()?;
        let prompt: Vec<i32> = (0..19)
            .flat_map(|r| [if r % 4 == 0 { 1 } else { 5 + r % 8 }, r % 8, (r + 1) % 8])
            .collect();
        let params = SampleParams {
            top_k: 1,
            ..SampleParams::default()
        };
        let mut batched = Engine::new(model()?, 32, 19)?;
        batched.trace_logits = true;
        let mut streamed = Engine::new(model()?, 32, 3)?;
        streamed.trace_logits = true;
        batched.prefill(&prompt, params)?;
        streamed.prefill(&prompt, params)?;
        let (frame, a) = batched.step()?;
        let (_, b) = streamed.step()?;
        let (a, b) = (a.unwrap(), b.unwrap());
        for (a, b) in a
            .main_logits
            .iter()
            .chain(&a.head_logits)
            .zip(b.main_logits.iter().chain(&b.head_logits))
        {
            assert!((a - b).abs() < 1e-3, "prefill chunk mismatch: {a} != {b}");
        }
        batched.prefill(&prompt, params)?;
        let (again, _) = batched.step()?;
        assert_eq!(frame, again);
        Ok(())
    }
    #[test]
    fn philox_matches_published_zero_counter_vector() {
        // Random123 Philox4x32-10, counter/key all zero -> 0x6627e8d5 first word.
        assert_eq!(
            uniform(0, 0, 0),
            (0x6627e8d5u32 as f32 + 1.0) / 4294967296.0
        );
    }
}
