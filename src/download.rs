//! Finds the model, or downloads it from Hugging Face into a cache directory.
//!
//! Two directories make a model: the converted int4 build (`drama-3-q4`: dense weights, codec, tokenizer, int4
//! experts) and, optionally, the original `fishaudio/drama-3` checkpoint (`drama-3`), whose FP8 experts give higher
//! quality from host RAM. Both are looked up in `models/` near the working directory or binary, then in the cache
//! directory `$XDG_CACHE_HOME/drama-rs/` (else `~/.cache/drama-rs/`), where they can be symlinks. A missing int4 build is
//! downloaded to `drama-3-q4/` there (the old name `drama-q4/` is still found). Each file
//! is fetched to a `.part` sibling and renamed into place once complete, so a killed download never leaves a file
//! that looks done, and a rerun resumes with the files still missing.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::model::{ExpertMode, Sizes};

/// The Hugging Face repository the model comes from.
pub const HF_REPO: &str = "apiplant/drama-q4";
/// Transformer layers, one `layerNN.safetensors` each.
const LAYERS: usize = 40;
const FILES: &[&str] = &[
    "drama.json",
    "tokenizer.json",
    "globals.safetensors",
    "tower.safetensors",
    "codec.safetensors",
    "LICENSE",
    "NOTICE",
];

fn files() -> Vec<String> {
    FILES.iter().map(|s| s.to_string()).chain((0..LAYERS).map(|i| format!("layer{i:02}.safetensors"))).collect()
}

/// `$XDG_CACHE_HOME/drama-rs`, defaulting to `~/.cache/drama-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("drama-rs"))
}

/// Where [`download`] puts (or would put) the int4 model directory.
pub fn model_dir() -> Option<PathBuf> {
    Some(cache_dir()?.join(Q4_DIR))
}

/// Cache-directory names of the int4 build; the first is current, the second what 0.1.x downloaded into.
const Q4_DIR: &str = "drama-3-q4";
const Q4_DIR_LEGACY: &str = "drama-q4";
/// Cache-directory (and `models/`) name of the original checkpoint.
const ORIGINAL_DIR: &str = "drama-3";

fn is_model_dir(p: &Path) -> bool {
    p.join("drama.json").is_file() && p.join("globals.safetensors").is_file() && p.join("codec.safetensors").is_file()
}

/// The original `fishaudio/drama-3` checkpoint: has its safetensors index and no converted `drama.json`.
pub fn is_original_dir(p: &Path) -> bool {
    p.join("model.safetensors.index.json").is_file() && p.join("config.json").is_file() && !p.join("drama.json").is_file()
}

/// Directories to look in: the working directory, its parent, and up to three levels above the binary (the repo
/// root for `target/release/drama`).
fn search_roots() -> Vec<PathBuf> {
    let mut roots = vec![PathBuf::from("."), PathBuf::from("..")];
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        roots.extend(dir.ancestors().take(4).map(Path::to_path_buf));
    }
    roots
}

/// The int4 model to use when none is given: `models/drama-3-q4` or `drama-q4` in the search roots, then the cache
/// directory (`drama-3-q4`, `drama-q4`). `None` if none of them holds a complete model.
pub fn find_model() -> Option<PathBuf> {
    let local = search_roots().into_iter().flat_map(|r| [r.join("models").join(Q4_DIR), r.join(Q4_DIR_LEGACY)]);
    let cached = cache_dir().into_iter().flat_map(|c| [c.join(Q4_DIR), c.join(Q4_DIR_LEGACY)]);
    local.chain(cached).find(|p| is_model_dir(p) && files().iter().all(|f| f.ends_with("LICENSE") || f.ends_with("NOTICE") || p.join(f).is_file()))
}

/// The original checkpoint (for FP8 experts): `models/drama-3` in the search roots, then `drama-3` in the cache
/// directory. `None` if there is none.
pub fn find_original() -> Option<PathBuf> {
    let local = search_roots().into_iter().map(|r| r.join("models").join(ORIGINAL_DIR));
    local.chain(cache_dir().map(|c| c.join(ORIGINAL_DIR))).find(|p| is_original_dir(p))
}

/// [`find_model`], else download into the cache directory (about 21 GB).
pub fn default_model() -> Result<PathBuf> {
    match find_model() {
        Some(dir) => Ok(dir),
        None => download(),
    }
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string();
    let part = dest.with_file_name(format!("{name}.part"));

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut written, mut last_report) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if written - last_report >= 16 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!("\r  {name} {:.0}% ({} / {} MiB)  ", written as f64 * 100.0 / len as f64, written >> 20, len >> 20),
                None => eprint!("\r  {name} {} MiB  ", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    if written >= 16 << 20 {
        eprintln!("\r  {name} done ({} MiB)          ", written >> 20);
    }
    Ok(())
}

/// Downloads the model into the cache directory unless it is already there, and returns the directory.
pub fn download() -> Result<PathBuf> {
    let dir = model_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    let missing: Vec<String> = files().into_iter().filter(|f| !dir.join(f).is_file()).collect();
    if missing.is_empty() {
        return Ok(dir);
    }
    eprintln!("Downloading {HF_REPO} (about 21 GB) from https://huggingface.co/{HF_REPO} into {}", dir.display());
    for f in missing {
        fetch_to_file(&format!("https://huggingface.co/{HF_REPO}/resolve/main/{f}"), &dir.join(&f))?;
    }
    Ok(dir)
}

/// Which routed experts to run, and where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// FP8 from the original checkpoint if it is available and fits (VRAM + RAM), else int4.
    Auto,
    /// The converted int4 experts, on the GPU.
    Q4,
    /// Original FP8 experts, placed by the hardware: as many layers in VRAM as fit, the rest in host RAM.
    Fp8,
    /// Original FP8 experts, all in VRAM (needs a big GPU).
    Fp8Vram,
    /// Original FP8 experts, all in pinned host RAM, read over PCIe.
    Fp8Ram,
}

/// The user's choices.
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub want: Want,
    /// Layers whose FP8 experts go to VRAM, instead of as many as fit (with `Auto` or `Fp8`).
    pub vram_layers: Option<usize>,
    /// VRAM kept free for the KV caches, scratch buffers, codec and CUDA context.
    pub reserve_bytes: u64,
}

pub const DEFAULT_RESERVE: u64 = 3 << 30;

impl Default for Choice {
    fn default() -> Self {
        Self { want: Want::Auto, vram_layers: None, reserve_bytes: DEFAULT_RESERVE }
    }
}

/// Free memory, where known.
#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    pub vram: Option<u64>,
    pub ram: Option<u64>,
}

impl Hardware {
    pub fn detect() -> Self {
        Self { vram: crate::gpu::mem_info().ok().map(|(free, _)| free as u64), ram: available_ram() }
    }
}

/// A resolved model: the int4 directory (dense weights, codec, tokenizer) and the expert source.
#[derive(Debug)]
pub struct Resolved {
    pub dir: PathBuf,
    pub experts: ExpertMode,
    /// One line for the user: what was chosen.
    pub summary: String,
}

/// `MemAvailable` from /proc/meminfo; `None` where that is not readable.
fn available_ram() -> Option<u64> {
    let m = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = m.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kb << 10)
}

/// Find both directories and choose the experts. `model` and `original` are the user's `--model` and `--fp8-model`;
/// either may name the other kind of directory: `--model` may be the original checkpoint, in which case the int4 build
/// is found (or downloaded) separately for the dense weights.
pub fn resolve(model: Option<&Path>, original: Option<&Path>, choice: Choice) -> Result<Resolved> {
    #[cfg(feature = "mlx")]
    let choice = if choice.want == Want::Auto { Choice { want: Want::Q4, ..choice } } else { choice };
    let model_is_original = model.is_some_and(is_original_dir);
    let q4 = match model {
        Some(p) if !model_is_original => p.to_path_buf(),
        _ => default_model()?,
    };
    let orig = original.map(Path::to_path_buf).or_else(|| model.filter(|_| model_is_original).map(Path::to_path_buf)).or_else(find_original);
    let sizes = crate::model::sizes(&q4)?;
    resolve_with(q4, orig, choice, sizes, Hardware::detect())
}

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

/// [`resolve`] once the directories, model sizes and hardware are known.
pub fn resolve_with(dir: PathBuf, original: Option<PathBuf>, choice: Choice, sizes: Sizes, hw: Hardware) -> Result<Resolved> {
    let Choice { want, vram_layers, reserve_bytes } = choice;
    let q4 = |why: &str| {
        let summary = format!("int4 experts on the GPU{why}");
        Ok(Resolved { dir: dir.clone(), experts: ExpertMode::Q4, summary })
    };
    if want == Want::Q4 {
        if vram_layers.is_some() {
            bail!("--fp8-vram-layers only applies to FP8 experts");
        }
        return q4("");
    }
    if matches!(want, Want::Fp8Vram | Want::Fp8Ram) && vram_layers.is_some() {
        bail!("--fp8-vram-layers can't be combined with --experts fp8-vram or fp8-ram; use --experts fp8");
    }
    let Some(orig) = original else {
        if want == Want::Auto {
            eprintln!("note: original checkpoint not found (models/drama-3 or ~/.cache/drama-rs/drama-3): using the int4 experts, which are lower quality");
            return q4("");
        }
        bail!("FP8 experts need the original checkpoint (hf download fishaudio/drama-3): put it at models/drama-3 or ~/.cache/drama-rs/drama-3, or pass --fp8-model");
    };
    if !is_original_dir(&orig) {
        bail!("{} is not the original fishaudio/drama-3 checkpoint (no model.safetensors.index.json and config.json)", orig.display());
    }

    let layers = sizes.layers;
    let per = sizes.fp8_layer_bytes.max(1);
    // Layers whose experts fit in VRAM next to the dense weights and the reserve.
    let fit = hw.vram.map_or(0, |v| v.saturating_sub(sizes.dense_bytes + reserve_bytes) / per).min(layers as u64) as usize;
    let n = match (want, vram_layers) {
        (Want::Fp8Ram, _) => 0,
        (Want::Fp8Vram, _) => {
            if fit < layers {
                bail!(
                    "--experts fp8-vram needs ~{:.0} GB of free VRAM for the experts plus {:.0} GB for the rest; {:.0} GB free fits {fit} of {layers} layers (try --experts fp8)",
                    gb(per * layers as u64),
                    gb(sizes.dense_bytes + reserve_bytes),
                    gb(hw.vram.unwrap_or(0)),
                );
            }
            layers
        }
        (_, Some(v)) => {
            let v = v.min(layers);
            if v > fit {
                bail!("--fp8-vram-layers {v} needs more VRAM than is free: {fit} layers fit (free {:.1} GB, reserve {:.1} GB)", gb(hw.vram.unwrap_or(0)), gb(reserve_bytes));
            }
            v
        }
        (_, None) => fit,
    };
    let ram_needed = per * (layers - n) as u64;
    if hw.ram.is_some_and(|r| r < ram_needed) {
        let ram = hw.ram.unwrap_or(0);
        if want == Want::Auto {
            eprintln!("note: FP8 experts would need ~{:.0} GB of host RAM and only {:.0} GB is available: using the int4 experts", gb(ram_needed), gb(ram));
            return q4("");
        }
        eprintln!("warning: {} wants ~{:.0} GB of free RAM, only {:.0} GB available", "FP8 offloading", gb(ram_needed), gb(ram));
    }
    let summary = format!(
        "original FP8 experts: {n}/{layers} layers in VRAM ({:.1} GB), {} in host RAM ({:.1} GB)",
        gb(per * n as u64),
        layers - n,
        gb(ram_needed)
    );
    Ok(Resolved { dir, experts: ExpertMode::Fp8 { dir: orig, vram_layers: n }, summary })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("drama-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn original(name: &str) -> PathBuf {
        let d = tmp(name);
        std::fs::write(d.join("model.safetensors.index.json"), "{}").unwrap();
        std::fs::write(d.join("config.json"), "{}").unwrap();
        d
    }

    const GB: u64 = 1_000_000_000;
    // 40 layers of 0.8 GB experts, 3 GB dense: like the real model
    const SIZES: Sizes = Sizes { layers: 40, fp8_layer_bytes: 800_000_000, dense_bytes: 3 * GB };
    const SMALL: Hardware = Hardware { vram: Some(24 * GB), ram: Some(48 * GB) }; // 24 GB card
    const BIG: Hardware = Hardware { vram: Some(80 * GB), ram: Some(16 * GB) }; // 80 GB card

    fn run(o: Option<PathBuf>, want: Want, hw: Hardware) -> Result<Resolved> {
        resolve_with("q4".into(), o, Choice { want, ..Choice::default() }, SIZES, hw)
    }

    fn placement(r: &Resolved) -> Option<usize> {
        match r.experts {
            ExpertMode::Fp8 { vram_layers, .. } => Some(vram_layers),
            ExpertMode::Q4 | ExpertMode::Q4Ram { .. } => None,
        }
    }

    #[test]
    fn auto_places_by_hardware() {
        let o = original("auto");
        // 24 GB: 24 - 3 dense - 3.2 reserve = ~17.8 GB of experts = 22 layers in VRAM, the rest in RAM
        let n = placement(&run(Some(o.clone()), Want::Auto, SMALL).unwrap()).unwrap();
        assert!((20..=24).contains(&n), "{n}");
        // 80 GB: everything in VRAM, so little RAM is needed
        assert_eq!(placement(&run(Some(o.clone()), Want::Fp8, BIG).unwrap()), Some(40));
        assert_eq!(placement(&run(Some(o), Want::Auto, BIG).unwrap()), Some(40));
    }

    #[test]
    fn explicit_placements() {
        let o = original("explicit");
        assert_eq!(placement(&run(Some(o.clone()), Want::Fp8Ram, SMALL).unwrap()), Some(0));
        assert_eq!(placement(&run(Some(o.clone()), Want::Fp8Vram, BIG).unwrap()), Some(40));
        assert!(run(Some(o.clone()), Want::Fp8Vram, SMALL).is_err());
        assert!(placement(&run(Some(o), Want::Q4, SMALL).unwrap()).is_none());
    }

    #[test]
    fn explicit_layer_count() {
        let o = original("layers");
        let with = |want, n| resolve_with("q4".into(), Some(o.clone()), Choice { want, vram_layers: Some(n), ..Choice::default() }, SIZES, SMALL);
        assert_eq!(placement(&with(Want::Fp8, 10).unwrap()), Some(10));
        assert!(with(Want::Fp8, 40).is_err()); // doesn't fit
        assert!(with(Want::Fp8Ram, 10).is_err()); // conflicting controls
        assert!(with(Want::Q4, 10).is_err());
    }

    #[test]
    fn auto_falls_back_to_q4() {
        assert!(placement(&run(None, Want::Auto, SMALL).unwrap()).is_none());
        let o = original("lowram");
        let low = Hardware { vram: Some(24 * GB), ram: Some(8 * GB) };
        assert!(placement(&run(Some(o.clone()), Want::Auto, low).unwrap()).is_none());
        // explicit fp8 still runs when RAM looks short (it only warns)
        assert!(placement(&run(Some(o.clone()), Want::Fp8, low).unwrap()).is_some());
        // a big GPU needs no RAM at all
        let big_low = Hardware { vram: Some(80 * GB), ram: Some(2 * GB) };
        assert_eq!(placement(&run(Some(o), Want::Auto, big_low).unwrap()), Some(40));
        assert!(run(None, Want::Fp8, SMALL).is_err());
    }

    #[test]
    fn rejects_a_q4_dir_as_original() {
        let d = tmp("notorig");
        std::fs::write(d.join("drama.json"), "{}").unwrap();
        std::fs::write(d.join("model.safetensors.index.json"), "{}").unwrap();
        std::fs::write(d.join("config.json"), "{}").unwrap();
        assert!(!is_original_dir(&d));
        assert!(run(Some(d), Want::Auto, SMALL).is_err());
    }
}
