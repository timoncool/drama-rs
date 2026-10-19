//! Fish Audio Drama 3 text-to-speech with CUDA or MLX on Apple Silicon.
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

mod config;
mod audio_io;
#[cfg(not(feature = "mlx"))]
pub mod codec;
#[cfg(feature = "mlx")]
#[path = "mlx/codec.rs"]
pub mod codec;
#[cfg(feature = "mlx")]
pub mod apple;
#[cfg(feature = "download")]
pub mod download;
#[cfg(not(feature = "mlx"))]
pub mod cpu_moe;
#[cfg(not(feature = "mlx"))]
pub mod engine;
#[cfg(feature = "mlx")]
#[path = "mlx/engine.rs"]
pub mod engine;
#[cfg(not(feature = "mlx"))]
pub mod ffi;
#[cfg(not(feature = "mlx"))]
pub mod gpu;
#[cfg(feature = "mlx")]
#[path = "mlx/gpu.rs"]
pub mod gpu;
#[cfg(not(feature = "mlx"))]
pub mod model;
#[cfg(feature = "mlx")]
#[path = "mlx/model.rs"]
pub mod model;
pub mod prompt;
#[cfg(feature = "serve")]
pub mod serve;
pub mod tts;
