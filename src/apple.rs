//! Native MLX runtime and checkpoint quantization, using Apple unified memory.
use anyhow::{Context, Result};
use mlx_rs::{ops::indexing::TryIndexOp, Array, Dtype};

pub fn init() -> Result<()> {
    static INIT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        (|| -> Result<()> {
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    for path in [
                        dir.join("mlx.metallib"),
                        dir.join("../share/drama-rs/mlx.metallib"),
                    ] {
                        if path.is_file() {
                            mlx_rs::metal::set_metallib_path(path.to_string_lossy())?;
                            break;
                        }
                    }
                }
            }
            mlx_rs::Device::set_default(&mlx_rs::Device::new(mlx_rs::DeviceType::Gpu, 0));
            let output = std::process::Command::new("/usr/sbin/sysctl")
                .args(["-n", "hw.memsize"])
                .output()?;
            anyhow::ensure!(
                output.status.success(),
                "could not query unified memory size"
            );
            let total: usize = std::str::from_utf8(&output.stdout)?.trim().parse()?;
            let limit = match std::env::var("DRAMA_UNIFIED_MEMORY_GB") {
                Ok(v) => {
                    let gb: f64 = v
                        .parse()
                        .context("DRAMA_UNIFIED_MEMORY_GB must be a positive number")?;
                    anyhow::ensure!(
                        gb.is_finite() && gb > 0.0 && gb * (1u64 << 30) as f64 <= total as f64,
                        "DRAMA_UNIFIED_MEMORY_GB must fit physical unified memory"
                    );
                    (gb * (1u64 << 30) as f64) as usize
                }
                Err(_) => total / 5 * 4,
            };
            mlx_rs::memory::set_memory_limit(limit)?;
            mlx_rs::memory::set_cache_limit((limit / 16).min(512 << 20))?;
            Ok(())
        })()
        .map_err(|e| format!("{e:#}"))
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!(e.clone()))?;
    Ok(())
}

pub fn memory_info() -> Result<(usize, usize)> {
    init()?;
    let total = mlx_rs::memory::memory_limit()?;
    Ok((
        total.saturating_sub(mlx_rs::memory::active_memory()?),
        total,
    ))
}

pub(crate) fn slice(x: &Array, axis: i32, start: i32, len: i32) -> Result<Array> {
    let ids = Array::from_slice(&(start..start + len).collect::<Vec<_>>(), &[len]);
    Ok(x.take_axis(&ids, axis)?)
}
pub(crate) fn row(x: &Array, r: i32) -> Result<Array> {
    Ok(x.try_index(r)?)
}
pub(crate) fn cat(xs: &[&Array], axis: i32) -> Result<Array> {
    Ok(mlx_rs::ops::concatenate(xs, axis)?)
}
pub(crate) fn rms(x: &Array, w: &Array, eps: f32) -> Result<Array> {
    Ok(mlx_rs::fast::rms_norm(
        &x.as_dtype(Dtype::Float32)?,
        Some(&w.as_dtype(Dtype::Float32)?),
        eps,
    )?)
}
pub(crate) fn silu(x: &Array) -> Result<Array> {
    Ok(x.multiply(mlx_rs::ops::sigmoid(x)?)?)
}
pub(crate) fn swiglu(x: &Array) -> Result<Array> {
    let n = *x.shape().last().unwrap() / 2;
    Ok(silu(&slice(x, -1, 0, n)?)?.multiply(slice(x, -1, n, n)?)?)
}
pub(crate) fn vec_f32(x: &Array) -> Result<Vec<f32>> {
    Ok(x.as_dtype(Dtype::Float32)?
        .contiguous()?
        .reshape(&[-1])?
        .to_vec_exact::<f32>()?)
}

/// Original checkpoint values mapped losslessly to MLX affine quantization.
/// q4 is already low-nibble first; q8 is shifted from signed to unsigned with a bias.
#[derive(Clone)]
pub struct Quantized {
    pub q: Array,
    pub scales: Array,
    pub biases: Array,
    pub bits: i32,
    pub group: i32,
}
impl Quantized {
    pub fn q8(q: Array, scales: Array) -> Result<Self> {
        anyhow::ensure!(
            q.ndim() == 2 && q.shape()[1] % 128 == 0,
            "invalid q8 checkpoint shape"
        );
        let packed = q
            .as_dtype(Dtype::Int32)?
            .add(Array::from(128i32))?
            .as_dtype(Dtype::Uint8)?
            .view::<u32>()?;
        let scales = scales.as_dtype(Dtype::Float32)?;
        let biases = scales.multiply(Array::from(-128.0f32))?;
        packed.eval()?;
        scales.eval()?;
        biases.eval()?;
        Ok(Self {
            q: packed,
            scales,
            biases,
            bits: 8,
            group: 128,
        })
    }
    pub fn q4(q: Array, sm: Array) -> Result<Self> {
        anyhow::ensure!(
            q.ndim() == 3 && q.shape()[2] % 32 == 0,
            "invalid q4 checkpoint shape"
        );
        let packed = q.view::<u32>()?;
        let scales = sm
            .try_index((.., .., .., 0))?
            .contiguous()?
            .as_dtype(Dtype::Float32)?;
        let biases = sm
            .try_index((.., .., .., 1))?
            .contiguous()?
            .as_dtype(Dtype::Float32)?;
        packed.eval()?;
        scales.eval()?;
        biases.eval()?;
        Ok(Self {
            q: packed,
            scales,
            biases,
            bits: 4,
            group: 64,
        })
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::ops::quantized_matmul(
            &x.as_dtype(Dtype::Float32)?,
            &self.q,
            &self.scales,
            &self.biases,
            true,
            self.group,
            self.bits,
        )?)
    }
    pub fn expert(&self, e: i32) -> Result<Self> {
        Ok(Self {
            q: row(&self.q, e)?,
            scales: row(&self.scales, e)?,
            biases: row(&self.biases, e)?,
            bits: self.bits,
            group: self.group,
        })
    }
    pub fn rows(&self, start: i32, n: i32) -> Result<Self> {
        Ok(Self {
            q: slice(&self.q, 0, start, n)?,
            scales: slice(&self.scales, 0, start, n)?,
            biases: slice(&self.biases, 0, start, n)?,
            bits: self.bits,
            group: self.group,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_quantization_matches_dense_matmul() -> Result<()> {
        init()?;
        let signed: Vec<i8> = (0..256).map(|i| (i as i8).wrapping_mul(3)).collect();
        let scale = Array::from_slice(&[0.125f32, 0.25], &[2, 1]);
        let q8 = Quantized::q8(Array::from_slice(&signed, &[2, 128]), scale)?;
        let input: Vec<f32> = (0..384).map(|i| (i as f32 * 0.13).sin()).collect();
        let x = Array::from_slice(&input, &[3, 128]);
        let expected: Vec<f32> = signed
            .iter()
            .enumerate()
            .map(|(i, &v)| v as f32 * if i < 128 { 0.125 } else { 0.25 })
            .collect();
        let ref8 = x.matmul(Array::from_slice(&expected, &[2, 128]).t())?;
        for (a, b) in vec_f32(&q8.forward(&x)?)?.iter().zip(vec_f32(&ref8)?) {
            assert!((a - b).abs() < 2e-3, "q8 {a} != {b}");
        }

        let packed: Vec<u8> = (0..128).map(|i| i as u8).collect();
        let sm = Array::from_slice(
            &[0.25f32, -2.0, 0.5, -3.0, 0.125, 1.0, 0.375, -1.0],
            &[1, 2, 2, 2],
        );
        let q4 = Quantized::q4(Array::from_slice(&packed, &[1, 2, 64]), sm)?;
        let mut expected = Vec::new();
        let scales = [0.25, 0.5, 0.125, 0.375];
        let biases = [-2.0, -3.0, 1.0, -1.0];
        for (i, &b) in packed.iter().enumerate() {
            for nib in [b & 15, b >> 4] {
                let group = i / 32;
                expected.push(nib as f32 * scales[group] + biases[group]);
            }
        }
        let ref4 = x.matmul(Array::from_slice(&expected, &[2, 128]).t())?;
        for (a, b) in vec_f32(&q4.expert(0)?.forward(&x)?)?
            .iter()
            .zip(vec_f32(&ref4)?)
        {
            assert!((a - b).abs() < 2e-3, "q4 {a} != {b}");
        }
        Ok(())
    }
}
