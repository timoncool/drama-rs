
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use drama::engine::{Engine, SampleParams};
use drama::{codec, gpu, tts};

#[derive(Parser)]
#[command(name = "drama", version, about = "Fish Audio Drama 3 text-to-speech")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::ValueEnum, Clone, Copy, PartialEq)]
enum Experts {
    Auto,
    Q4,
    #[cfg_attr(feature = "mlx", value(hide = true))]
    Fp8,
    #[cfg_attr(feature = "mlx", value(hide = true))]
    Fp8Vram,
    #[cfg_attr(feature = "mlx", value(hide = true))]
    Fp8Ram,
    #[cfg_attr(feature = "mlx", value(hide = true))]
    Q4Ram,
}

impl From<Experts> for drama::download::Want {
    fn from(e: Experts) -> Self {
        match e {
            Experts::Auto => Self::Auto,
            Experts::Q4 => Self::Q4,
            Experts::Fp8 => Self::Fp8,
            Experts::Fp8Vram => Self::Fp8Vram,
            Experts::Fp8Ram => Self::Fp8Ram,
            Experts::Q4Ram => Self::Q4,
        }
    }
}

#[derive(clap::Args)]
struct Common {
    /// The int4 build (tools/convert.py output; dense weights, codec, tokenizer), or the original fishaudio/drama-3
    /// checkpoint (then the int4 build is found separately). Default: `models/drama-3-q4` or `drama-q4` here, in the
    /// parent or next to the binary, then ~/.cache/drama-rs/drama-3-q4, else downloaded there (~21 GB)
    #[arg(long, env = "DRAMA_MODEL")]
    model: Option<PathBuf>,
    /// Routed experts. `auto` (default): original FP8 when the checkpoint is found and fits, else int4. `fp8`: original
    /// FP8, as many layers in VRAM as fit and the rest in pinned host RAM (read over PCIe). `fp8-vram`: all in VRAM
    /// (~35 GB card). `fp8-ram`: all in host RAM (~32 GB). `q4`: converted int4 on the GPU, ~8x faster. `q4-ram`: converted
    /// int4 in page-locked host RAM with a VRAM cache of the experts in use (the card's free VRAM or --vram-gb); `auto`
    /// takes it when int4 does not fit
    #[cfg_attr(feature = "mlx", arg(help = "Routed experts: auto and q4 use converted int4 weights on Apple Silicon"))]
    #[arg(long, value_enum, default_value_t = Experts::Auto)]
    experts: Experts,
    /// Advanced: how many layers' FP8 experts to put in VRAM (default: as many as fit), the rest in host RAM
    #[cfg_attr(feature = "mlx", arg(hide = true))]
    #[arg(long)]
    fp8_vram_layers: Option<usize>,
    /// Advanced: VRAM (GB) kept free for the KV caches, scratch, codec and CUDA context when placing FP8 experts
    #[cfg_attr(feature = "mlx", arg(hide = true))]
    #[arg(long, default_value_t = 3.0)]
    vram_reserve_gb: f64,
    /// The original fishaudio/drama-3 checkpoint for fp8 experts. Default: `models/drama-3` (searched like the
    /// model), then ~/.cache/drama-rs/drama-3
    #[cfg_attr(feature = "mlx", arg(hide = true))]
    #[arg(long, env = "DRAMA_FP8_MODEL")]
    fp8_model: Option<PathBuf>,
    /// Context length (prompt rows + generated frames)
    #[arg(long, default_value_t = 4096)]
    max_len: usize,
    /// Prefill chunk size in rows
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 30)]
    top_k: i32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Most VRAM to take, in GiB (0 = what is free less 1 GiB). When the whole model does not fit, the routed experts
    /// stay in RAM and a VRAM cache of the ones in use takes the rest of the limit: less VRAM, slower decoding.
    #[arg(long, default_value_t = 0.0)]
    vram_gb: f64,
    /// Prefill every prompt whole instead of starting from the saved state of its voice part (A/B tests)
    #[arg(long)]
    no_voice_reuse: bool,
}

impl Common {
    /// Find the model directories and choose the experts.
    fn resolve(&self) -> Result<drama::download::Resolved> {
        #[cfg(feature = "mlx")]
        anyhow::ensure!(matches!(self.experts, Experts::Auto | Experts::Q4) && self.fp8_vram_layers.is_none() && self.fp8_model.is_none(),
            "Apple Silicon uses converted int4 experts: select --experts q4; FP8 placement is available with the CUDA build");
        let choice = drama::download::Choice {
            want: self.experts.into(),
            vram_layers: self.fp8_vram_layers,
            reserve_bytes: (self.vram_reserve_gb * (1u64 << 30) as f64) as u64,
        };
        let r = drama::download::resolve(self.model.as_deref(), self.fp8_model.as_deref(), choice)?;
        eprintln!("experts: {} (dense weights from {})", r.summary, r.dir.display());
        Ok(r)
    }
    fn params(&self) -> SampleParams {
        SampleParams { temperature: self.temperature, top_k: self.top_k, top_p: self.top_p, pad: 0, seed: self.seed }
    }
    fn vram_limit(&self) -> usize {
        (self.vram_gb * (1u64 << 30) as f64) as usize
    }
    /// The int4 experts in RAM for `q4-ram`, and for `auto` when int4 does not fit the card or --vram-gb.
    fn placed(&self, r: &drama::download::Resolved) -> Result<drama::model::ExpertMode> {
        use drama::model::ExpertMode;
        match (&r.experts, self.experts) {
            #[cfg(not(feature = "mlx"))]
            (ExpertMode::Q4, Experts::Q4Ram) => Ok(ExpertMode::Q4Ram { vram_limit: self.vram_limit() }),
            #[cfg(not(feature = "mlx"))]
            (ExpertMode::Q4, Experts::Auto) => tts::Tts::fitting(&r.dir, self.vram_limit()),
            (e, _) => Ok(e.clone()),
        }
    }
    fn engine(&self) -> Result<Engine> {
        let r = self.resolve()?;
        let experts = self.placed(&r)?;
        let engine = if matches!(experts, drama::model::ExpertMode::Q4Ram { .. }) {
            tts::Tts::load_with(&r.dir, self.max_len, self.chunk, &experts)?.engine
        } else {
            let st = gpu::Stream::new()?;
            let model = drama::model::Model::load_with(&r.dir, &st, &experts)?;
            Engine::new(model, self.max_len, self.chunk)?
        };
        let (free, total) = gpu::mem_info()?;
        eprintln!("GPU memory: {:.2} GiB used of {:.2} GiB", (total - free) as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64);
        Ok(engine)
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Synthesize speech (optionally cloning a reference voice) to a WAV file and/or speakers.
    Speak {
        #[command(flatten)]
        common: Common,
        /// Text to speak (with --interactive, lines are read from stdin instead)
        #[arg(long)]
        text: Option<String>,
        /// Output WAV file
        #[arg(long)]
        out: Option<PathBuf>,
        /// Stream audio to the speakers (ffplay on macOS, PipeWire on Linux)
        #[arg(long)]
        play: bool,
        /// Read lines from stdin and speak each one (keeps the model loaded)
        #[arg(long)]
        interactive: bool,
        /// Reference voice clip(s) to clone (WAV); pair each with --ref-text
        #[arg(long)]
        ref_wav: Vec<PathBuf>,
        /// Transcript of each reference clip
        #[arg(long)]
        ref_text: Vec<String>,
        /// Write the generated frames here (raw little-endian i32, [frames][11])
        #[arg(long)]
        codes_out: Option<PathBuf>,
        /// Write the prompt rows here (raw little-endian i32, [rows][11]; input for `trace`)
        #[arg(long)]
        prompt_out: Option<PathBuf>,
        #[arg(long, default_value_t = 2048)]
        max_new: usize,
        /// Frames in the first streamed audio chunk, and in each later one
        #[arg(long, default_value_t = 4)]
        first_chunk: usize,
        #[arg(long, default_value_t = 16)]
        stream_chunk: usize,
        /// Generate the whole text at once instead of sentence by sentence
        #[arg(long)]
        no_split: bool,
    },
    /// Run raw prompt rows and dump frames + logits per step (engine tests).
    Trace {
        #[command(flatten)]
        common: Common,
        /// Raw little-endian i32 prompt rows [rows][11]
        #[arg(long)]
        rows: PathBuf,
        #[arg(long, default_value_t = 8)]
        steps: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Decode raw codes ([frames][10] little-endian i32) to a WAV file.
    Decode {
        /// codec.safetensors (default: the one in the model directory)
        #[arg(long = "codec", env = "DRAMA_CODEC")]
        codec_path_opt: Option<PathBuf>,
        #[arg(long)]
        codes: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Feed the streaming decoder this many frames at a time (0 = all at once)
        #[arg(long, default_value_t = 0)]
        stream_chunk: usize,
    },
    /// Encode a WAV file to codes ([frames][10] little-endian i32).
    Encode {
        /// codec.safetensors (default: the one in the model directory)
        #[arg(long = "codec", env = "DRAMA_CODEC")]
        codec_path_opt: Option<PathBuf>,
        #[arg(long)]
        wav: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Serve an HTTP API: POST /v1/speak renders text to WAV (see `drama::serve`).
    Serve {
        #[command(flatten)]
        common: Common,
        /// Address to listen on
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Voice to preload as NAME=WAV=TRANSCRIPT (repeatable); requests pick it with "voice": "NAME"
        #[arg(long)]
        voice: Vec<String>,
        #[arg(long, default_value_t = 2048)]
        max_new: usize,
        #[arg(long, default_value_t = 4)]
        first_chunk: usize,
        #[arg(long, default_value_t = 16)]
        stream_chunk: usize,
        /// Generate the whole text at once instead of sentence by sentence (requests can override with "split")
        #[arg(long)]
        no_split: bool,
    },
    /// Measure decode speed.
    Bench {
        #[command(flatten)]
        common: Common,
        #[arg(long, default_value_t = 200)]
        steps: usize,
        #[arg(long, default_value_t = 64)]
        prompt_rows: usize,
    },
}

fn print_cache_stats(engine: &Engine) -> Result<()> {
    if let Some((hits, misses)) = engine.cache_stats()? {
        let all = (hits + misses).max(1);
        eprintln!("expert cache: {:.1}% of {all} expert reads hit, {misses} copied from RAM", 100.0 * hits as f64 / all as f64);
    }
    Ok(())
}

fn codec_file(p: Option<PathBuf>) -> Result<PathBuf> {
    match p {
        Some(p) => Ok(p),
        None => Ok(drama::download::default_model()?.join("codec.safetensors")),
    }
}

fn read_i32(path: &PathBuf) -> Result<Vec<i32>> {
    let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn write_le<T: Copy>(w: &mut impl Write, v: &[T]) -> Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    w.write_all(bytes)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Speak { common, text, out, play, interactive, ref_wav, ref_text, codes_out, prompt_out, max_new, first_chunk, stream_chunk, no_split } => {
            if ref_wav.len() != ref_text.len() {
                anyhow::bail!("give one --ref-text per --ref-wav");
            }
            if text.is_none() && !interactive {
                anyhow::bail!("give --text, or --interactive to read lines from stdin");
            }
            let t_load = Instant::now();
            let r = common.resolve()?;
            let experts = common.placed(&r)?;
            let mut tts = tts::Tts::load_with(&r.dir, common.max_len, common.chunk, &experts)?;
            tts.engine.reuse_voice = !common.no_voice_reuse;
            let (free, total) = gpu::mem_info()?;
            eprintln!(
                "ready in {:.1}s, GPU memory {:.2} / {:.2} GiB",
                t_load.elapsed().as_secs_f64(),
                (total - free) as f64 / (1u64 << 30) as f64,
                total as f64 / (1u64 << 30) as f64
            );
            let mut refs = Vec::new();
            for (wav, transcript) in ref_wav.iter().zip(&ref_text) {
                let t1 = Instant::now();
                let r = tts.reference(wav, transcript)?;
                eprintln!("reference {}: {} frames in {:.0} ms", wav.display(), r.codes[0].len(), t1.elapsed().as_secs_f64() * 1e3);
                refs.push(r);
            }
            let opts = tts::SpeakOpts { params: common.params(), max_new, first_chunk, chunk: stream_chunk, split: !no_split, keep_stalled: true };
            let mut player = if play { Some(tts::Player::new()?) } else { None };
            let lines: Box<dyn Iterator<Item = String>> = match text {
                Some(t) if !interactive => Box::new(std::iter::once(t)),
                _ => Box::new(std::io::stdin().lines().map_while(Result::ok)),
            };
            for (n, line) in lines.enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let spoken = tts.speak(line, &refs, &opts, |a| match player.as_mut() {
                    Some(p) => p.write(a),
                    None => Ok(()),
                })?;
                let audio_s = spoken.audio.len() as f64 / codec::SAMPLE_RATE as f64;
                eprintln!(
                    "{} sentence(s), {} prompt rows -> {} frames ({audio_s:.2}s of audio) in {:.2}s ({:.1}x real time), first audio after {:.0} ms",
                    spoken.segments,
                    spoken.prompt_rows,
                    spoken.frames.len(),
                    spoken.elapsed.as_secs_f64(),
                    audio_s / spoken.elapsed.as_secs_f64(),
                    spoken.first_audio.map(|d| d.as_secs_f64() * 1e3).unwrap_or(f64::NAN),
                );
                print_cache_stats(&tts.engine)?;
                if let Some(path) = &out {
                    let path = if interactive { path.with_file_name(format!("{}-{n}.wav", path.file_stem().unwrap().to_string_lossy())) } else { path.clone() };
                    codec::write_wav(&path, &spoken.audio)?;
                    eprintln!("wrote {}", path.display());
                }
                if let Some(path) = &prompt_out {
                    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
                    write_le(&mut f, &spoken.prompt)?;
                }
                if let Some(path) = &codes_out {
                    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
                    for fr in &spoken.frames {
                        write_le(&mut f, fr)?;
                    }
                }
            }
            if let Some(p) = player {
                p.finish()?;
            }
        }
        Cmd::Serve { common, addr, voice, max_new, first_chunk, stream_chunk, no_split } => {
            let defaults = drama::serve::Defaults { params: common.params(), max_new, first_chunk, chunk: stream_chunk, split: !no_split };
            let r = common.resolve()?;
            let (dir, experts, max_len, chunk) = (r.dir, r.experts, common.max_len, common.chunk);
            let load = move || -> Result<_> {
                let t = Instant::now();
                let tts = tts::Tts::load_with(&dir, max_len, chunk, &experts)?;
                let mut voices = std::collections::HashMap::new();
                for v in &voice {
                    let mut it = v.splitn(3, '=');
                    let (Some(name), Some(wav), Some(text)) = (it.next(), it.next(), it.next()) else {
                        anyhow::bail!("--voice wants NAME=WAV=TRANSCRIPT, got {v:?}");
                    };
                    voices.insert(name.to_string(), tts.reference(wav.as_ref(), text)?);
                }
                eprintln!("ready in {:.1}s", t.elapsed().as_secs_f64());
                Ok((tts, voices))
            };
            drama::serve::run(load, defaults, &addr)?;
        }
        Cmd::Decode { codec_path_opt, codes, out, stream_chunk } => {
            let codec_path = codec_file(codec_path_opt)?;
            let raw = read_i32(&codes)?;
            let codec = codec::Codec::load(&codec_path)?;
            let t1 = Instant::now();
            let audio = if stream_chunk == 0 {
                codec.decode(&raw)?
            } else {
                let mut ds = codec.stream()?;
                let mut audio = Vec::new();
                for part in raw.chunks(stream_chunk * 10) {
                    audio.extend(codec.decode_chunk(&mut ds, part)?);
                }
                audio
            };
            eprintln!("decoded {} samples in {:.0} ms", audio.len(), t1.elapsed().as_secs_f64() * 1e3);
            codec::write_wav(&out, &audio)?;
        }
        Cmd::Encode { codec_path_opt, wav, out } => {
            let codec_path = codec_file(codec_path_opt)?;
            let codec = codec::Codec::load(&codec_path)?;
            let (audio, sr) = codec::read_wav(&wav)?;
            let audio = codec::resample(&audio, sr, codec::SAMPLE_RATE);
            let t1 = Instant::now();
            let codes = codec.encode(&audio)?;
            eprintln!("encoded {} frames in {:.0} ms", codes.len() / 10, t1.elapsed().as_secs_f64() * 1e3);
            let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
            write_le(&mut f, &codes)?;
        }
        Cmd::Trace { common, rows, steps, out } => {
            let mut engine = common.engine()?;
            engine.trace_logits = true;
            let prompt = read_i32(&rows)?;
            engine.prefill(&prompt, common.params(), 0)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
            for _ in 0..steps {
                let (frame, trace) = engine.step()?;
                let trace = trace.unwrap();
                write_le(&mut f, &frame)?;
                write_le(&mut f, &trace.main_logits)?;
                write_le(&mut f, &trace.head_logits)?;
            }
            eprintln!("wrote {}", out.display());
        }
        Cmd::Bench { common, steps, prompt_rows } => {
            let mut engine = common.engine()?;
            let cfg = engine.model.cfg.clone();
            let width = cfg.frame_width();
            let mut rows = vec![0i32; prompt_rows * width];
            for r in 0..prompt_rows {
                rows[r * width] = 1000 + r as i32;
            }
            let t0 = Instant::now();
            engine.prefill(&rows, common.params(), 0)?;
            eprintln!("prefill {prompt_rows} rows: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
            for _ in 0..5 {
                engine.step()?;
            }
            let t0 = Instant::now();
            for _ in 0..steps {
                engine.step()?;
            }
            let per = t0.elapsed().as_secs_f64() / steps as f64;
            eprintln!("decode: {:.2} ms/frame, {:.1} frames/s, {:.1}x real time", per * 1e3, 1.0 / per, 1.0 / per / 21.533);
            print_cache_stats(&engine)?;
        }
    }
    Ok(())
}
