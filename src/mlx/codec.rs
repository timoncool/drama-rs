//! Fish codec on MLX, with bounded causal histories for exact streaming.
pub use crate::audio_io::{read_wav, read_wav_bytes, resample, write_wav};
use crate::engine::KvCache;
use crate::{
    apple::{cat, rms, slice, swiglu, vec_f32},
    model::tensor,
};
use anyhow::{ensure, Result};
use mlx_rs::{
    ops::{self, indexing::TryIndexOp},
    Array, Dtype,
};
use std::{collections::HashMap, path::Path, sync::Mutex};
pub const SAMPLE_RATE: u32 = 44100;
pub const FRAME_SAMPLES: usize = 2048;
const D: i32 = 1024;
const HEADS: i32 = 16;
const NCB: usize = 10;
const RATES: [i32; 4] = [8, 8, 4, 2];
const DILATIONS: [i32; 3] = [1, 3, 9];

struct Inner {
    t: HashMap<String, Array>,
}
/// MLX arrays may move across threads but may not be accessed concurrently. The mutex also
/// serializes reference encoding with the streaming decoder; no unsafe Sync impl is needed.
pub struct Codec {
    inner: Mutex<Inner>,
}
#[derive(Default)]
pub struct DecodeStream {
    frames_done: usize,
    caches: HashMap<String, KvCache>,
    history: HashMap<String, Array>,
}

impl Inner {
    fn w(&self, name: &str) -> Result<&Array> {
        tensor(&self.t, name)
    }
    fn gemm(&self, x: &Array, w: &str) -> Result<Array> {
        // Converted codec weights use f16; retain f32 accumulators and residuals.
        Ok(x.as_dtype(Dtype::Float16)?
            .as_dtype(Dtype::Float32)?
            .matmul(self.w(w)?.as_dtype(Dtype::Float32)?.t())?)
    }
    fn snake(&self, x: &Array, alpha: Option<&str>) -> Result<Array> {
        let Some(alpha) = alpha else {
            return Ok(x.clone());
        };
        let a = self.w(alpha)?;
        Ok(x.add(
            x.multiply(a)?
                .sin()?
                .square()?
                .divide(a.add(Array::from(1e-9f32))?)?,
        )?)
    }
    fn history_input(
        &self,
        ds: &mut DecodeStream,
        key: &str,
        x: &Array,
        ctx: i32,
    ) -> Result<Array> {
        if ctx == 0 {
            return Ok(x.clone());
        }
        let old = match ds.history.remove(key) {
            Some(a) => a,
            None => Array::zeros::<f32>(&[ctx, x.shape()[1]])?,
        };
        let all = cat(&[&old, x], 0)?;
        let hist = slice(&all, 0, all.shape()[0] - ctx, ctx)?;
        hist.eval()?;
        ds.history.insert(key.to_string(), hist);
        Ok(all)
    }
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        ds: &mut DecodeStream,
        key: &str,
        x: &Array,
        w: &str,
        b: &str,
        k: i32,
        dil: i32,
        stride: i32,
        alpha: Option<&str>,
    ) -> Result<Array> {
        let n = x.shape()[0];
        let c = x.shape()[1];
        let ctx = (k - 1) * dil + 1 - stride;
        ensure!(
            ctx >= 0 && n % stride == 0,
            "codec convolution shape is not stride aligned"
        );
        let input = self.history_input(ds, key, x, ctx)?;
        let input = self.snake(&input, alpha)?;
        let outc = self.w(w)?.shape()[0];
        let weight = self
            .w(w)?
            .reshape(&[outc, k, c])?
            .as_dtype(Dtype::Float32)?;
        let y = ops::conv1d(
            input
                .as_dtype(Dtype::Float16)?
                .as_dtype(Dtype::Float32)?
                .expand_dims(0)?,
            weight,
            stride,
            0,
            dil,
            1,
        )?;
        Ok(y.squeeze_axes(&[0])?.add(self.w(b)?)?)
    }
    fn tconv(
        &self,
        ds: &mut DecodeStream,
        key: &str,
        x: &Array,
        prefix: &str,
        rate: i32,
        two: bool,
        alpha: Option<&str>,
    ) -> Result<Array> {
        let n = x.shape()[0];
        let c = x.shape()[1];
        let cols = if two {
            let all = self.history_input(ds, key, x, 1)?;
            cat(
                &[
                    &self.snake(x, alpha)?,
                    &self.snake(&slice(&all, 0, 0, n)?, alpha)?,
                ],
                1,
            )?
        } else {
            self.snake(x, alpha)?
        };
        let w = format!("{prefix}.w");
        let b = format!("{prefix}.b");
        let outc = self.w(&w)?.shape()[0] / rate;
        let _ = c;
        Ok(self
            .gemm(&cols, &w)?
            .reshape(&[n * rate, outc])?
            .add(self.w(&b)?)?)
    }
    fn convnext(&self, ds: &mut DecodeStream, p: &str, x: &Array) -> Result<Array> {
        let n = x.shape()[0];
        let c = x.shape()[1];
        let all = self.history_input(ds, p, x, 6)?;
        let mut y = Array::zeros::<f32>(&[n, c])?;
        for j in 0..7 {
            y = y.add(
                slice(&all, 0, j, n)?
                    .multiply(self.w(&format!("{p}.dw_w"))?.try_index((.., j))?)?,
            )?;
        }
        y = y.add(self.w(&format!("{p}.dw_b"))?)?;
        y = mlx_rs::fast::layer_norm(
            y,
            Some(self.w(&format!("{p}.ln_w"))?),
            Some(self.w(&format!("{p}.ln_b"))?),
            1e-6,
        )?;
        y = self
            .gemm(&y, &format!("{p}.pw1_w"))?
            .add(self.w(&format!("{p}.pw1_b"))?)?;
        y = mlx_rs::nn::gelu(&y)?;
        y = self
            .gemm(&y, &format!("{p}.pw2_w"))?
            .add(self.w(&format!("{p}.pw2_b"))?)?
            .multiply(self.w(&format!("{p}.gamma"))?)?;
        Ok(x.add(y)?)
    }
    fn transformer(
        &self,
        ds: &mut DecodeStream,
        prefix: &str,
        layers: usize,
        mut x: Array,
        window: usize,
        t0: usize,
    ) -> Result<Array> {
        let n = x.shape()[0];
        for l in 0..layers {
            let p = format!("{prefix}.l{l}");
            let xn = rms(&x, self.w(&format!("{p}.attn_norm"))?, 1e-5)?;
            let qkv = self.gemm(&xn, &format!("{p}.wqkv"))?;
            let q = slice(&qkv, 1, 0, D)?
                .reshape(&[n, HEADS, 64])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let k = slice(&qkv, 1, D, D)?
                .reshape(&[n, HEADS, 64])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let v = slice(&qkv, 1, 2 * D, D)?
                .reshape(&[n, HEADS, 64])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let q = mlx_rs::fast::rope(q, 64, true, 10000.0, 1.0, t0 as i32, None)?;
            let k = mlx_rs::fast::rope(k, 64, true, 10000.0, 1.0, t0 as i32, None)?;
            let cache = ds.caches.entry(p.clone()).or_default();
            let old = cache.k.as_ref().map_or(0, |a| a.shape()[2]);
            let (k, v) = cache.append(k, v, 2)?;
            let total = old + n;
            let mask: Vec<f32> = (0..n)
                .flat_map(|i| {
                    (0..total).map(move |j| {
                        if j <= old + i && j > old + i - window as i32 {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Array::from_slice(&mask, &[n, total]);
            let a = mlx_rs::fast::scaled_dot_product_attention(q, &k, &v, 0.125, &mask, None)?;
            // Keep only the preceding window for the next chunk.
            let keep = (window as i32).min(total);
            cache.k = Some(slice(&k, 2, total - keep, keep)?);
            cache.v = Some(slice(&v, 2, total - keep, keep)?);
            cache.k.as_ref().unwrap().eval()?;
            cache.v.as_ref().unwrap().eval()?;
            let a = a
                .squeeze_axes(&[0])?
                .transpose_axes(&[1, 0, 2])?
                .reshape(&[n, D])?;
            x = x.add(
                self.gemm(&a, &format!("{p}.wo"))?
                    .multiply(self.w(&format!("{p}.attn_scale"))?)?,
            )?;
            let xn = rms(&x, self.w(&format!("{p}.ffn_norm"))?, 1e-5)?;
            let ff = swiglu(&self.gemm(&xn, &format!("{p}.w13"))?)?;
            x = x.add(
                self.gemm(&ff, &format!("{p}.w2"))?
                    .multiply(self.w(&format!("{p}.ffn_scale"))?)?,
            )?;
            x.eval()?;
        }
        rms(&x, self.w(&format!("{prefix}.norm"))?, 1e-5)
    }
    fn decode_part(&self, ds: &mut DecodeStream, codes: &[i32]) -> Result<Array> {
        let n = (codes.len() / NCB) as i32;
        let mut x = Array::zeros::<f32>(&[n, D])?;
        for c in 0..NCB {
            let size = if c == 0 { 4096 } else { 1024 };
            let off = if c == 0 {
                0
            } else {
                4096 + (c as i32 - 1) * 1024
            };
            let ids: Vec<i32> = codes
                .chunks_exact(NCB)
                .map(|f| off + f[c].clamp(0, size - 1))
                .collect();
            x = x.add(
                self.w("rvq.table")?
                    .take_axis(Array::from_slice(&ids, &[n]), 0)?,
            )?;
        }
        x = self.transformer(ds, "post", 8, x, 128, ds.frames_done)?;
        for u in 0..2 {
            x = self.tconv(ds, &format!("up{u}"), &x, &format!("up{u}"), 2, false, None)?;
            x = self.convnext(ds, &format!("up{u}.cnx"), &x)?;
        }
        ds.frames_done += n as usize;
        x = self.conv(ds, "dec.in", &x, "dec.in.w", "dec.in.b", 7, 1, 1, None)?;
        x.eval()?;
        for (b, &rate) in RATES.iter().enumerate() {
            let p = format!("dec.b{b}");
            x = self.tconv(
                ds,
                &p,
                &x,
                &format!("{p}.up"),
                rate,
                true,
                Some(&format!("{p}.alpha")),
            )?;
            for (u, &dil) in DILATIONS.iter().enumerate() {
                let p = format!("{p}.ru{u}");
                let y = self.conv(
                    ds,
                    &format!("{p}.c1"),
                    &x,
                    &format!("{p}.c1.w"),
                    &format!("{p}.c1.b"),
                    7,
                    dil,
                    1,
                    Some(&format!("{p}.a1")),
                )?;
                let y = self.conv(
                    ds,
                    &format!("{p}.c2"),
                    &y,
                    &format!("{p}.c2.w"),
                    &format!("{p}.c2.b"),
                    1,
                    1,
                    1,
                    Some(&format!("{p}.a2")),
                )?;
                x = x.add(y)?;
                x.eval()?;
            }
        }
        // Final output weights are f32 in the converted codec.
        let n = x.shape()[0];
        let c = x.shape()[1];
        let all = self.history_input(ds, "dec.out", &x, 6)?;
        let all = self.snake(&all, Some("dec.out.alpha"))?;
        let y = ops::conv1d(
            all.expand_dims(0)?,
            self.w("dec.out.w")?.reshape(&[1, 7, c])?,
            1,
            0,
            1,
            1,
        )?
        .reshape(&[n])?
        .add(self.w("dec.out.b")?)?;
        let y = ops::tanh(y)?;
        Ok(y)
    }
    fn encode(&self, audio: &[f32]) -> Result<Vec<i32>> {
        ensure!(!audio.is_empty(), "empty reference audio");
        let len = audio.len().div_ceil(FRAME_SAMPLES) * FRAME_SAMPLES;
        let mut padded = audio.to_vec();
        padded.resize(len, 0.0);
        let mut ds = DecodeStream::default();
        let mut x = Array::from_slice(&padded, &[len as i32, 1]);
        x = self.conv(&mut ds, "enc.in", &x, "enc.in.w", "enc.in.b", 7, 1, 1, None)?;
        for (b, &stride) in [2, 4, 8, 8].iter().enumerate() {
            for (u, &dil) in DILATIONS.iter().enumerate() {
                let p = format!("enc.b{b}.ru{u}");
                let y = self.conv(
                    &mut ds,
                    &format!("{p}.c1"),
                    &x,
                    &format!("{p}.c1.w"),
                    &format!("{p}.c1.b"),
                    7,
                    dil,
                    1,
                    Some(&format!("{p}.a1")),
                )?;
                let y = self.conv(
                    &mut ds,
                    &format!("{p}.c2"),
                    &y,
                    &format!("{p}.c2.w"),
                    &format!("{p}.c2.b"),
                    1,
                    1,
                    1,
                    Some(&format!("{p}.a2")),
                )?;
                x = x.add(y)?;
                x.eval()?;
            }
            let p = format!("enc.b{b}");
            x = self.conv(
                &mut ds,
                &p,
                &x,
                &format!("{p}.down.w"),
                &format!("{p}.down.b"),
                2 * stride,
                1,
                stride,
                Some(&format!("{p}.alpha")),
            )?;
            if self.t.contains_key(&format!("{p}.tr.norm")) {
                x = self.transformer(&mut ds, &format!("{p}.tr"), 4, x, 512, 0)?;
            }
        }
        x = self.conv(
            &mut ds,
            "enc.out",
            &x,
            "enc.out.w",
            "enc.out.b",
            3,
            1,
            1,
            Some("enc.out.alpha"),
        )?;
        for i in 0..2 {
            x = self.conv(
                &mut ds,
                &format!("down{i}"),
                &x,
                &format!("down{i}.w"),
                &format!("down{i}.b"),
                2,
                1,
                2,
                None,
            )?;
            x = self.convnext(&mut ds, &format!("down{i}.cnx"), &x)?;
        }
        x = self.transformer(&mut ds, "pre", 8, x, 128, 0)?;
        let n = x.shape()[0];
        let mut codes = vec![0; n as usize * NCB];
        for i in 0..NCB {
            let z = x
                .matmul(self.w(&format!("vq{i}.in_w"))?.t())?
                .add(self.w(&format!("vq{i}.in_b"))?)?;
            let score = z.matmul(self.w(&format!("vq{i}.codebook_n"))?.t())?;
            let ids = ops::indexing::argmax_axis(score, 1, false)?;
            let host = ids.as_dtype(Dtype::Int32)?.to_vec_exact::<i32>()?;
            for (r, &code) in host.iter().enumerate() {
                codes[r * NCB + i] = code;
            }
            let off = if i == 0 {
                0
            } else {
                4096 + (i as i32 - 1) * 1024
            };
            x = x.subtract(
                self.w("rvq.table")?
                    .take_axis(ids.add(Array::from(off))?, 0)?,
            )?;
        }
        Ok(codes)
    }
}
impl Codec {
    pub fn load(path: &Path) -> Result<Self> {
        crate::apple::init()?;
        let tensors = Array::load_safetensors(path)?;
        // Safetensors loading is lazy and refers to this thread's CPU stream.
        // Materialize those reads before the streaming decoder uses the arrays
        // on its own thread, where that loading stream is unavailable.
        mlx_rs::transforms::eval(tensors.values())?;
        Ok(Self {
            inner: Mutex::new(Inner { t: tensors }),
        })
    }
    pub fn stream(&self) -> Result<DecodeStream> {
        Ok(DecodeStream::default())
    }
    /// Same as `stream`: no buffers are sized by pass here (the CUDA codec sizes its buffers by it).
    pub fn stream_for(&self, _pass: usize) -> Result<DecodeStream> {
        self.stream()
    }
    pub fn encode(&self, audio: &[f32]) -> Result<Vec<i32>> {
        self.inner
            .lock()
            .map_err(|_| anyhow::anyhow!("codec mutex poisoned"))?
            .encode(audio)
    }
    pub fn decode_chunk(&self, ds: &mut DecodeStream, codes: &[i32]) -> Result<Vec<f32>> {
        ensure!(
            codes.len() % NCB == 0,
            "codec input must contain complete 10-codebook frames"
        );
        let inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("codec mutex poisoned"))?;
        let mut audio = Vec::new();
        // Small passes bound decoder activation memory independently of utterance length.
        for part in codes.chunks(16 * NCB) {
            audio.extend(vec_f32(&inner.decode_part(ds, part)?)?);
        }
        Ok(audio)
    }
    pub fn decode(&self, codes: &[i32]) -> Result<Vec<f32>> {
        self.decode_chunk(&mut self.stream()?, codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loaded_codec_weights_can_be_used_on_the_audio_thread() -> Result<()> {
        use safetensors::{tensor::TensorView, Dtype as SafeDtype};
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let view = TensorView::new(SafeDtype::F32, vec![2, 2], &bytes)?;
        let path = std::env::temp_dir().join(format!(
            "drama-codec-thread-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        safetensors::tensor::serialize_to_file([("w", view)], &None, &path)?;
        let codec = Codec::load(&path)?;
        std::fs::remove_file(path)?;
        let result = std::thread::spawn(move || -> Result<Vec<f32>> {
            let inner = codec
                .inner
                .lock()
                .map_err(|_| anyhow::anyhow!("codec mutex poisoned"))?;
            vec_f32(&inner.gemm(&Array::from_slice(&[1.0f32, 1.0], &[1, 2]), "w")?)
        })
        .join()
        .map_err(|_| anyhow::anyhow!("audio thread panicked"))??;
        assert_eq!(result, vec![3.0, 7.0]);
        Ok(())
    }

    #[test]
    fn causal_convolution_and_transpose_keep_chunk_histories() -> Result<()> {
        crate::apple::init()?;
        let w = Array::from_slice(
            &(0..36).map(|i| i as f32 / 100.0).collect::<Vec<_>>(),
            &[4, 9],
        );
        let tw = Array::from_slice(
            &(0..48).map(|i| i as f32 / 200.0).collect::<Vec<_>>(),
            &[8, 6],
        );
        let inner = Inner {
            t: HashMap::from([
                ("c.w".into(), w),
                ("c.b".into(), Array::zeros::<f32>(&[4])?),
                ("t.w".into(), tw),
                ("t.b".into(), Array::zeros::<f32>(&[4])?),
            ]),
        };
        let x = Array::from_slice(
            &(0..24).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>(),
            &[8, 3],
        );
        let full = inner.conv(
            &mut DecodeStream::default(),
            "c",
            &x,
            "c.w",
            "c.b",
            3,
            2,
            1,
            None,
        )?;
        let mut ds = DecodeStream::default();
        let a = inner.conv(
            &mut ds,
            "c",
            &slice(&x, 0, 0, 3)?,
            "c.w",
            "c.b",
            3,
            2,
            1,
            None,
        )?;
        let b = inner.conv(
            &mut ds,
            "c",
            &slice(&x, 0, 3, 5)?,
            "c.w",
            "c.b",
            3,
            2,
            1,
            None,
        )?;
        for (a, b) in vec_f32(&full)?.iter().zip(vec_f32(&cat(&[&a, &b], 0)?)?) {
            assert!((a - b).abs() < 1e-5);
        }
        let full = inner.tconv(&mut DecodeStream::default(), "t", &x, "t", 2, true, None)?;
        let mut ds = DecodeStream::default();
        let a = inner.tconv(&mut ds, "t", &slice(&x, 0, 0, 3)?, "t", 2, true, None)?;
        let b = inner.tconv(&mut ds, "t", &slice(&x, 0, 3, 5)?, "t", 2, true, None)?;
        for (a, b) in vec_f32(&full)?.iter().zip(vec_f32(&cat(&[&a, &b], 0)?)?) {
            assert!((a - b).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn windowed_transformer_streams_across_cache_eviction() -> Result<()> {
        crate::apple::init()?;
        let eye = ops::eye::<f32>(D, None, None)?;
        let mut ts = HashMap::new();
        for name in ["attn_norm", "ffn_norm"] {
            ts.insert(format!("tr.l0.{name}"), Array::ones::<f32>(&[D])?);
        }
        for name in ["attn_scale", "ffn_scale"] {
            ts.insert(
                format!("tr.l0.{name}"),
                Array::ones::<f32>(&[D])?.multiply(Array::from(0.1f32))?,
            );
        }
        ts.insert("tr.l0.wqkv".into(), cat(&[&eye, &eye, &eye], 0)?);
        ts.insert("tr.l0.wo".into(), eye);
        ts.insert("tr.l0.w13".into(), Array::zeros::<f32>(&[128, D])?);
        ts.insert("tr.l0.w2".into(), Array::zeros::<f32>(&[D, 64])?);
        ts.insert("tr.norm".into(), Array::ones::<f32>(&[D])?);
        let inner = Inner { t: ts };
        let values: Vec<f32> = (0..7 * D).map(|i| (i as f32 * 0.017).sin()).collect();
        let x = Array::from_slice(&values, &[7, D]);
        let full = inner.transformer(&mut DecodeStream::default(), "tr", 1, x.clone(), 2, 0)?;
        let mut ds = DecodeStream::default();
        let a = inner.transformer(&mut ds, "tr", 1, slice(&x, 0, 0, 3)?, 2, 0)?;
        let b = inner.transformer(&mut ds, "tr", 1, slice(&x, 0, 3, 4)?, 2, 3)?;
        for (a, b) in vec_f32(&full)?.iter().zip(vec_f32(&cat(&[&a, &b], 0)?)?) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
        assert_eq!(ds.caches["tr.l0"].k.as_ref().unwrap().shape()[2], 2);
        Ok(())
    }
}
