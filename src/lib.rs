//! Fish Audio Drama 3 text-to-speech on a single 24 GB GPU, in Rust + CUDA.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use drama::engine::SampleParams;
//! use drama::tts::{SpeakOpts, Tts};
//!
//! // a directory produced by tools/convert.py, or `hf download apiplant/drama-q4`
//! let mut tts = Tts::load("drama-q4".as_ref(), 4096, 512)?;
//! let voice = tts.reference("voice.wav".as_ref(), "Exactly what is said in voice.wav.")?;
//! let opts = SpeakOpts { params: SampleParams::default(), max_new: 2048, first_chunk: 4, chunk: 16, split: true, keep_stalled: true };
//! // the closure receives 44.1 kHz audio as it is generated (streaming)
//! let spoken = tts.speak("Hello there.", &[voice], &opts, |_chunk| Ok(()))?;
//! drama::codec::write_wav("hello.wav".as_ref(), &spoken.audio)?;
//! # Ok(()) }
//! ```

pub mod codec;
pub mod engine;
pub mod ffi;
pub mod gpu;
pub mod model;
pub mod prompt;
pub mod tts;
