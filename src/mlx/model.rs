//! MLX weights in the existing drama-rs-v1 checkpoint format.
pub use crate::config::*;
use crate::{apple::Quantized, config::Meta, gpu::Stream};
use anyhow::{Context, Result};
use mlx_rs::Array;
use std::{collections::HashMap, path::Path};

#[derive(Clone)]
pub struct Layer {
    pub tensors: HashMap<String, Array>,
    pub linears: HashMap<String, Quantized>,
    pub gate_up: Quantized,
    pub down: Quantized,
}
#[derive(Clone)]
pub struct TowerLayer {
    pub tensors: HashMap<String, Array>,
    pub linears: HashMap<String, Quantized>,
}
pub struct Model {
    pub cfg: Config,
    pub layers: Vec<Layer>,
    pub tower: Vec<TowerLayer>,
    pub heads: Quantized,
    pub codebook_emb: Array,
    pub lm_head_cand: Array,
    pub final_norm: Array,
    pub embed_tokens: Array,
}

pub(crate) fn tensor<'a>(ts: &'a HashMap<String, Array>, name: &str) -> Result<&'a Array> {
    ts.get(name)
        .with_context(|| format!("checkpoint tensor {name} is missing"))
}
fn take(ts: &mut HashMap<String, Array>, name: &str) -> Result<Array> {
    ts.remove(name)
        .with_context(|| format!("checkpoint tensor {name} is missing"))
}
fn q8s(ts: &mut HashMap<String, Array>) -> Result<HashMap<String, Quantized>> {
    let names: Vec<_> = ts
        .keys()
        .filter_map(|s| s.strip_suffix(".q8").map(str::to_owned))
        .collect();
    names
        .into_iter()
        .map(|n| {
            let w = Quantized::q8(take(ts, &format!("{n}.q8"))?, take(ts, &format!("{n}.s"))?)?;
            Ok((n, w))
        })
        .collect()
}
pub(crate) fn linear<'a>(ts: &'a HashMap<String, Quantized>, name: &str) -> Result<&'a Quantized> {
    ts.get(name)
        .with_context(|| format!("checkpoint linear {name} is missing"))
}

impl Model {
    pub fn load(dir: &Path, stream: &Stream) -> Result<Self> {
        Self::load_with(dir, stream, &ExpertMode::Q4)
    }
    pub fn load_with(dir: &Path, _: &Stream, experts: &ExpertMode) -> Result<Self> {
        crate::apple::init()?;
        anyhow::ensure!(matches!(experts, ExpertMode::Q4), "the MLX backend currently uses the converted int4 experts; select --experts q4 (FP8 remains available with CUDA)");
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("drama.json"))?)?;
        anyhow::ensure!(
            meta.format == "drama-rs-v1" && meta.q4_group == 64 && meta.q8_group == 128,
            "unsupported converted model format"
        );
        let cfg = meta.config;
        let mut layers = Vec::new();
        for i in 0..cfg.text_config.num_hidden_layers {
            let mut ts = Array::load_safetensors(dir.join(format!("layer{i:02}.safetensors")))?;
            let linears = q8s(&mut ts)?;
            let gate_up = Quantized::q4(
                take(&mut ts, "moe.gate_up.q4")?,
                take(&mut ts, "moe.gate_up.sm")?,
            )?;
            let down = Quantized::q4(take(&mut ts, "moe.down.q4")?, take(&mut ts, "moe.down.sm")?)?;
            layers.push(Layer {
                tensors: ts,
                linears,
                gate_up,
                down,
            });
            eprint!(
                "\rloading MLX weights: layer {}/{}",
                i + 1,
                cfg.text_config.num_hidden_layers
            );
        }
        eprintln!();
        let mut ts = Array::load_safetensors(dir.join("tower.safetensors"))?;
        let mut tower = Vec::new();
        for i in 0..cfg.tower_layers() {
            let prefix = format!("t{i}.");
            let names: Vec<_> = ts
                .keys()
                .filter(|n| n.starts_with(&prefix))
                .cloned()
                .collect();
            let mut tensors: HashMap<_, _> = names
                .into_iter()
                .map(|n| {
                    let value = ts.remove(&n).unwrap();
                    (n[prefix.len()..].to_string(), value)
                })
                .collect();
            let linears = q8s(&mut tensors)?;
            tower.push(TowerLayer { tensors, linears });
        }
        let heads = Quantized::q8(take(&mut ts, "heads.q8")?, take(&mut ts, "heads.s")?)?;
        let codebook_emb = codebook_embeddings(
            take(&mut ts, "codebook_emb")?,
            cfg.audio_decoder_config.num_codebooks,
            cfg.audio_decoder_config.vocab_size,
            cfg.text_config.hidden_size,
        )?;
        let mut gs = Array::load_safetensors(dir.join("globals.safetensors"))?;
        Ok(Self {
            cfg,
            layers,
            tower,
            heads,
            codebook_emb,
            lm_head_cand: take(&mut gs, "lm_head_cand")?,
            final_norm: take(&mut gs, "final_norm")?,
            embed_tokens: take(&mut gs, "embed_tokens")?,
        })
    }
}

/// Converted checkpoints flatten the codebook and vocabulary axes to [NCB * V, D].
/// Normalize that table once so both backbone and tower gather by (codebook, code).
fn codebook_embeddings(data: Array, codebooks: usize, vocab: usize, dim: usize) -> Result<Array> {
    Ok(data.reshape(&[codebooks as i32, vocab as i32, dim as i32])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::TryIndexOp;

    #[test]
    fn converted_flat_codebook_rows_retain_codebook_offsets() -> Result<()> {
        crate::apple::init()?;
        let flat = Array::from_slice(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[6, 2]);
        let table = codebook_embeddings(flat, 2, 3, 2)?;
        assert_eq!(
            table.try_index((0, 2))?.to_vec_exact::<f32>()?,
            vec![4.0, 5.0]
        );
        assert_eq!(
            table.try_index((1, 2))?.to_vec_exact::<f32>()?,
            vec![10.0, 11.0]
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sizes {
    pub layers: usize,
    pub fp8_layer_bytes: u64,
    pub dense_bytes: u64,
}
pub fn sizes(dir: &Path) -> Result<Sizes> {
    let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("drama.json"))?)?;
    let tc = &meta.config.text_config;
    let (e, h, i) = (
        tc.num_experts as u64,
        tc.hidden_size as u64,
        tc.moe_intermediate_size as u64,
    );
    let fp8_layer_bytes =
        e * (2 * i * h + h * i) + 4 * e * ((2 * i / 128) * (h / 128) + (h / 128) * (i / 128));
    let mut dense_bytes = std::fs::metadata(dir.join("tower.safetensors")).map_or(0, |m| m.len());
    for l in 0..tc.num_hidden_layers {
        let file = std::fs::File::open(dir.join(format!("layer{l:02}.safetensors")))?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        for (name, t) in safetensors::SafeTensors::deserialize(&map)?.tensors() {
            if !name.starts_with("moe.gate_up") && !name.starts_with("moe.down") {
                dense_bytes += t.data().len() as u64;
            }
        }
    }
    Ok(Sizes {
        layers: tc.num_hidden_layers,
        fp8_layer_bytes,
        dense_bytes,
    })
}
