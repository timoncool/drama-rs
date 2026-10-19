//! Text -> audio: prompt building, LM generation and streaming codec decode, overlapped.
//!
//! The LM runs on the calling thread; sampled frames go over a channel to a codec thread
//! (its own CUDA stream) that decodes them in small chunks and hands the audio to a sink
//! (speakers and/or a buffer) while generation continues.

use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use tokenizers::Tokenizer;

use crate::codec::{self, Codec};
use crate::engine::{Engine, SampleParams};
use crate::model::{ExpertMode, Model};
use crate::prompt::{self, Reference};

/// VRAM left to other programs when the engine sizes itself to the free VRAM: a sixteenth of it, 256 MiB to 1 GiB.
pub fn auto_margin(free: usize) -> usize {
    (free / 16).clamp(256 << 20, 1 << 30)
}
/// VRAM the codec allocates while it encodes references and decodes speech.
const CODEC_WORK: usize = 768 << 20;

pub struct Tts {
    pub engine: Engine,
    pub codec: Codec,
    tok: Tokenizer,
}

pub struct SpeakOpts {
    pub params: SampleParams,
    /// Most frames per sentence with `split`, for the whole text without it.
    pub max_new: usize,
    /// Frames in the first streamed chunk (small = low latency) and in later ones.
    pub first_chunk: usize,
    pub chunk: usize,
    /// Generate sentence by sentence (see `Tts::speak`).
    pub split: bool,
    /// A sentence that ran into its cap and is said again keeps its stalled audio in the output (it was already
    /// streamed). False: a sentence reaches the decoder only once it ended, so the output has no stalled audio.
    pub keep_stalled: bool,
}

pub struct Spoken {
    pub audio: Vec<f32>,
    pub frames: Vec<Vec<i32>>,
    /// Prompt rows prefilled, over all sentences.
    pub prompt_rows: usize,
    /// The first sentence's prompt rows [rows][1 + num_codebooks].
    pub prompt: Vec<i32>,
    /// Sentences generated (1 without splitting).
    pub segments: usize,
    pub first_audio: Option<Duration>,
    pub elapsed: Duration,
}

/// Raw f32 audio to the default output through PipeWire.
pub struct Player {
    child: Child,
}

impl Player {
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "macos")]
        let child = Command::new("ffplay")
            .args(["-nodisp", "-autoexit", "-loglevel", "error", "-f", "f32le", "-ar", &codec::SAMPLE_RATE.to_string(), "-ac", "1", "-i", "pipe:0"])
            .stdin(Stdio::piped()).spawn().context("starting ffplay for --play (install ffmpeg)")?;
        #[cfg(not(target_os = "macos"))]
        let child = Command::new("pw-play")
            .args(["--raw", "--rate", &codec::SAMPLE_RATE.to_string(), "--channels", "1", "--format", "f32", "-"])
            .stdin(Stdio::piped())
            .spawn()
            .context("starting pw-play (PipeWire) for --play")?;
        Ok(Self { child })
    }

    pub fn write(&mut self, audio: &[f32]) -> Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(audio.as_ptr() as *const u8, std::mem::size_of_val(audio)) };
        self.child.stdin.as_mut().unwrap().write_all(bytes)?;
        Ok(())
    }

    /// Close the stream and wait until everything has been played.
    pub fn finish(mut self) -> Result<()> {
        drop(self.child.stdin.take());
        self.child.wait()?;
        Ok(())
    }
}

impl Tts {
    pub fn load(model_dir: &Path, max_len: usize, chunk: usize) -> Result<Self> {
        Self::load_with(model_dir, max_len, chunk, &ExpertMode::Q4)
    }

    #[cfg(not(feature = "mlx"))]
    /// The int4 experts on the GPU when the whole model fits `vram_limit` bytes (0: the free VRAM), else in RAM with a
    /// VRAM expert cache ([`ExpertMode::Q4Ram`]).
    pub fn fitting(model_dir: &Path, vram_limit: usize) -> Result<ExpertMode> {
        let fit_limit = if vram_limit == 0 { crate::gpu::mem_info()?.0 } else { vram_limit };
        let (enc_bytes, dec_bytes) = Codec::sizes(&model_dir.join("codec.safetensors"))?;
        let whole = enc_bytes + dec_bytes + crate::model::full_vram(model_dir)? <= fit_limit as u64;
        Ok(if whole { ExpertMode::Q4 } else { ExpertMode::Q4Ram { vram_limit } })
    }

    /// [`Tts::load`] with the routed experts from `experts` (int4 on the GPU or in host RAM, or the original FP8 in host
    /// RAM).
    pub fn load_with(model_dir: &Path, max_len: usize, chunk: usize, experts: &ExpertMode) -> Result<Self> {
        let (engine, codec) = match experts {
            #[cfg(not(feature = "mlx"))]
            ExpertMode::Q4Ram { vram_limit } => Self::load_offloaded(model_dir, max_len, chunk, *vram_limit, experts)?,
            _ => {
                let codec = Codec::load(&model_dir.join("codec.safetensors"))?;
                let st = crate::gpu::Stream::new()?;
                let model = Model::load_with(model_dir, &st, experts)?;
                (Engine::new(model, max_len, chunk)?, codec)
            }
        };
        let tok = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(Self { engine, codec, tok })
    }

    #[cfg(not(feature = "mlx"))]
    /// The experts in RAM: the codec's encoder stays in RAM too, and what `vram_limit` leaves after the weights and the
    /// codec's working memory goes to the expert cache.
    fn load_offloaded(model_dir: &Path, max_len: usize, chunk: usize, vram_limit: usize, experts: &ExpertMode) -> Result<(Engine, Codec)> {
        let (free0, _) = crate::gpu::mem_info()?;
        let limit = if vram_limit == 0 { free0.saturating_sub(auto_margin(free0)) } else { vram_limit };
        let codec = Codec::load_with(&model_dir.join("codec.safetensors"), true)?;
        let st = crate::gpu::Stream::new()?;
        let (free1, _) = crate::gpu::mem_info()?;
        let codec_vram = free0.saturating_sub(free1);
        let model = Model::load_with(model_dir, &st, experts)?;
        let (free2, _) = crate::gpu::mem_info()?;
        let used = free0.saturating_sub(free2);
        let keep_free = free2.saturating_sub(limit.saturating_sub(used)) + CODEC_WORK;
        let engine = Engine::with_cache(model, max_len, chunk, keep_free)?;
        let (free3, _) = crate::gpu::mem_info()?;
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "VRAM: codec {:.2} GiB, weights {:.2} GiB, engine (buffers and expert cache) {:.2} GiB",
            gib(codec_vram),
            gib(used.saturating_sub(codec_vram)),
            gib(free2.saturating_sub(free3))
        );
        Ok((engine, codec))
    }

    /// Encode a reference clip (any rate, mixed to mono) for voice cloning.
    pub fn reference(&self, wav: &Path, transcript: &str) -> Result<Reference> {
        let (audio, sr) = codec::read_wav(wav)?;
        self.reference_audio(&audio, sr, transcript)
    }

    /// [`Tts::reference`] from mono samples at `sr` Hz.
    pub fn reference_audio(&self, audio: &[f32], sr: u32, transcript: &str) -> Result<Reference> {
        let audio = codec::resample(audio, sr, codec::SAMPLE_RATE);
        let flat = self.codec.encode(&audio)?;
        let frames = flat.len() / 10;
        let codes = (0..10).map(|c| (0..frames).map(|t| flat[t * 10 + c]).collect()).collect();
        Ok(Reference { transcript: transcript.to_string(), codes })
    }

    /// Generate speech for `text`; `sink` receives audio chunks as soon as they are decoded.
    ///
    /// With `opts.split` the text is generated one sentence at a time (as the reference
    /// server's streaming endpoint does): without a voice reference the model often never
    /// ends after a sentence-final full stop and fills the rest with silence. A speaker
    /// with no reference gets their first sentence as the reference for the rest, so the
    /// voice stays the same across sentences.
    pub fn speak(&mut self, text: &str, refs: &[Reference], opts: &SpeakOpts, mut sink: impl FnMut(&[f32]) -> Result<()> + Send) -> Result<Spoken> {
        let dc = &self.engine.model.cfg.audio_decoder_config;
        let (ncb, cb) = (dc.num_codebooks, dc.vocab_size);
        let width = ncb + 1;
        let segments = if opts.split { prompt::split_sentences(text) } else { vec![text.to_string()] };
        // Build every prompt that does not depend on a generated voice up front, so bad input
        // fails before any audio is produced.
        prompt::build_prompt_rows(text, refs, &self.tok, ncb, cb)?;
        let covered: Vec<u32> = refs.iter().map(|r| prompt::leading_speaker(&r.transcript).unwrap_or(0)).collect();
        let t0 = Instant::now();
        let (tx, rx) = mpsc::channel::<Vec<i32>>();
        let codec = &self.codec;
        let engine = &mut self.engine;
        let tok = &self.tok;
        let (first, chunk) = (opts.first_chunk.max(1), opts.chunk.max(1));

        let (gen, decoded) = std::thread::scope(|scope| -> Result<_> {
            let worker = scope.spawn(move || -> Result<(Vec<f32>, Option<Duration>)> {
                let mut ds = codec.stream_for(first.max(chunk))?;
                let mut pending: Vec<i32> = Vec::new();
                let mut audio = Vec::new();
                let mut first_audio = None;
                let mut want = first;
                let mut flush = |pending: &mut Vec<i32>, audio: &mut Vec<f32>, first_audio: &mut Option<Duration>| -> Result<()> {
                    if pending.is_empty() {
                        return Ok(());
                    }
                    let a = codec.decode_chunk(&mut ds, pending)?;
                    pending.clear();
                    first_audio.get_or_insert_with(|| t0.elapsed());
                    sink(&a)?;
                    audio.extend_from_slice(&a);
                    Ok(())
                };
                while let Ok(codes) = rx.recv() {
                    pending.extend_from_slice(&codes);
                    // Take whatever else is already queued before decoding.
                    while let Ok(more) = rx.try_recv() {
                        pending.extend_from_slice(&more);
                    }
                    if pending.len() / 10 >= want {
                        flush(&mut pending, &mut audio, &mut first_audio)?;
                        want = chunk;
                    }
                }
                flush(&mut pending, &mut audio, &mut first_audio)?;
                Ok((audio, first_audio))
            });
            let gen = (|| -> Result<_> {
                let mut frames = Vec::new();
                let mut prompt_rows = 0;
                let mut first_prompt = Vec::new();
                // The given references, then each unreferenced speaker's first sentence.
                let mut voices: Vec<Reference> = refs.to_vec();
                let mut voiced: Vec<u32> = covered;
                for (i, seg) in segments.iter().enumerate() {
                    let (rows, voice_rows) = prompt::build_prompt(seg, &voices, tok, ncb, cb)?;
                    prompt_rows += rows.len() / width;
                    if i == 0 {
                        first_prompt = rows.clone();
                    }
                    let cap = if opts.split { frame_cap(seg) } else { usize::MAX };
                    // A sentence that runs into its cap has stalled (silence with no end); its
                    // audio is already out, so say it again with another seed rather than drop it.
                    let mut seg_frames = Vec::new();
                    let mut stalled = true;
                    for attempt in 0..=STALL_RETRIES {
                        let mut params = opts.params;
                        params.seed = params.seed.wrapping_add(i as u64).wrapping_add(attempt as u64 * 1000);
                        seg_frames = engine.generate(&rows, voice_rows, params, opts.max_new.min(cap), |frame| {
                            if opts.keep_stalled {
                                let _ = tx.send(frame[1..].to_vec());
                            }
                        })?;
                        stalled = seg_frames.len() >= cap;
                        if !stalled || (!opts.keep_stalled && attempt == STALL_RETRIES) {
                            if !opts.keep_stalled {
                                let _ = tx.send(seg_frames.iter().flat_map(|f| f[1..].iter().copied()).collect());
                            }
                            if stalled {
                                eprintln!("sentence {} ran into its {cap}-frame cap on every attempt", i + 1);
                            }
                            break;
                        }
                        eprintln!("sentence {} ran into its {cap}-frame cap without ending (attempt {})", i + 1, attempt + 1);
                        if opts.keep_stalled {
                            // Its audio was played, so it stays in the output.
                            frames.extend(seg_frames.drain(..));
                        } else {
                            seg_frames.clear();
                        }
                    }
                    if let Some(sp) = prompt::sole_speaker(seg) {
                        if opts.split && !stalled && !seg_frames.is_empty() && !voiced.contains(&sp) {
                            let codes = (0..ncb).map(|c| seg_frames.iter().map(|f| f[1 + c]).collect()).collect();
                            voices.push(Reference { transcript: seg.clone(), codes });
                            voiced.push(sp);
                        }
                    }
                    frames.extend(seg_frames);
                }
                Ok((frames, prompt_rows, first_prompt, segments.len()))
            })();
            drop(tx);
            let decoded = worker.join().map_err(|_| anyhow!("codec thread panicked"))?;
            Ok((gen?, decoded?))
        })?;
        let ((frames, prompt_rows, prompt, segments), (audio, first_audio)) = (gen, decoded);
        Ok(Spoken { audio, frames, prompt_rows, prompt, segments, first_audio, elapsed: t0.elapsed() })
    }
}

/// Extra attempts for a sentence that stalls.
const STALL_RETRIES: usize = 2;

/// Most frames a sentence may take: speech runs ~1-2 frames per Latin character (21.5 frames
/// per second); this allows slow, dramatic delivery and still cuts a stall within seconds.
fn frame_cap(sentence: &str) -> usize {
    let mut text = sentence.to_string();
    while let Some(i) = text.find("<|speaker:") {
        let end = text[i..].find("|>").map_or(text.len(), |e| i + e + 2);
        text.replace_range(i..end, "");
    }
    let text = text.trim();
    // In half frames: 2.5 frames per Latin character, 4 per CJK character.
    30 + text.chars().map(|c| if c as u32 >= 0x2E80 { 8 } else { 5 }).sum::<usize>() / 2
}

#[cfg(test)]
mod tests {
    #[test]
    fn frame_cap_ignores_speaker_tags() {
        assert_eq!(super::frame_cap("<|speaker:1|> Hi."), 30 + 3 * 5 / 2);
        assert_eq!(super::frame_cap("你好"), 30 + 8);
    }
}
