use crate::codec::SAMPLE_RATE;
use anyhow::{anyhow, Result};
use std::path::Path;

pub fn write_wav(path: &Path, audio: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in audio {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99).
pub fn resample(x: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new {
        return x.to_vec();
    }
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let g = gcd(orig, new);
    let (orig, new) = ((orig / g) as usize, (new / g) as usize);
    let lpw = 6.0f64;
    let base = orig.min(new) as f64 * 0.99;
    let width = (lpw * orig as f64 / base).ceil() as usize;
    let klen = 2 * width + orig;
    let mut kernels = vec![0f32; new * klen];
    for i in 0..new {
        for m in 0..klen {
            let idx = (m as f64 - width as f64) / orig as f64;
            let mut t = (-(i as f64) / new as f64 + idx) * base;
            t = t.clamp(-lpw, lpw);
            let window = (t * std::f64::consts::PI / lpw / 2.0).cos().powi(2);
            t *= std::f64::consts::PI;
            let k = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernels[i * klen + m] = (k * window * base / orig as f64) as f32;
        }
    }
    let mut padded = vec![0f32; width];
    padded.extend_from_slice(x);
    padded.extend(std::iter::repeat_n(0.0, width + orig));
    let frames = (padded.len() - klen) / orig + 1;
    let target = (new as f64 * x.len() as f64 / orig as f64).ceil() as usize;
    let mut out = Vec::with_capacity(frames * new);
    for n in 0..frames {
        let seg = &padded[n * orig..n * orig + klen];
        for i in 0..new {
            let k = &kernels[i * klen..(i + 1) * klen];
            out.push(seg.iter().zip(k).map(|(a, b)| a * b).sum());
        }
    }
    out.truncate(target);
    out
}

/// Read a WAV file as mono f32 (channels averaged) and its sample rate.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let r = hound::WavReader::open(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    wav_samples(r)
}

/// [`read_wav`] from bytes in memory (an uploaded file).
pub fn read_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    wav_samples(hound::WavReader::new(std::io::Cursor::new(bytes))?)
}

fn wav_samples<R: std::io::Read>(mut r: hound::WavReader<R>) -> Result<(Vec<f32>, u32)> {
    let spec = r.spec();
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|v| v.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = samples
        .chunks(ch)
        .map(|c| c.iter().sum::<f32>() / ch as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}
