
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

#[derive(clap::Args)]
struct Common {
    /// Directory produced by tools/convert.py
    #[arg(long)]
    model: PathBuf,
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
    fn params(&self) -> SampleParams {
        SampleParams { temperature: self.temperature, top_k: self.top_k, top_p: self.top_p, pad: 0, seed: self.seed }
    }
    fn vram_limit(&self) -> usize {
        (self.vram_gb * (1u64 << 30) as f64) as usize
    }
    fn engine(&self) -> Result<Engine> {
        let tts = tts::Tts::load_with(&self.model, self.max_len, self.chunk, self.vram_limit())?;
        let engine = tts.engine;
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
        /// Stream the audio to the speakers (PipeWire) while it is generated
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
        #[arg(long)]
        codec: PathBuf,
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
        #[arg(long)]
        codec: PathBuf,
        #[arg(long)]
        wav: PathBuf,
        #[arg(long)]
        out: PathBuf,
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
            let mut tts = tts::Tts::load_with(&common.model, common.max_len, common.chunk, common.vram_limit())?;
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
        Cmd::Decode { codec: codec_path, codes, out, stream_chunk } => {
            let raw = read_i32(&codes)?;
            let codec = codec::Codec::load(&codec_path, false)?;
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
        Cmd::Encode { codec: codec_path, wav, out } => {
            let codec = codec::Codec::load(&codec_path, false)?;
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
