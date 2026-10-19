/** Everything the home page says about the project, in one place. */

export interface Card {
  name: string;
  href: string;
  tagline: string;
  body: string;
}

export interface Feature {
  title: string;
  body: string;
}

export const SITE = {
  pkg: "drama-rs",
  hasCuda: false,
  bins: ["drama"],
  badges: ["Rust", "CUDA", "Apple Silicon", "int4", "Apache-2.0"],
  hero: { pre: "Drama 3 text-to-speech, ", accent: "on one GPU", post: "." },
  lead: "Fish Audio's 35B mixture-of-experts voice model on a single 24 GB NVIDIA card or an Apple Silicon Mac, in Rust, with hand-written CUDA and MLX. Clone a voice from a few seconds of audio, write multi-speaker dialogue, and stream it to the speakers faster than real time. No PyTorch.",
  heroNote: "Decoding at 8.9x real time on an RTX 4090, with the first audio 50 to 120 ms after the prompt is ready. Apple Silicon runs on MLX (a Mac with 32 GB or more).",
  demo: undefined as { href: string; label: string } | undefined,
  cargo: `cargo install drama-rs                          # needs the CUDA toolkit (nvcc)
DRAMA_CUDA_ARCH=sm_86 cargo install drama-rs     # for a GPU other than sm_89
cargo install drama-rs --no-default-features --features mlx,cli   # Apple Silicon (MLX)`,
  terminal: {
    title: "drama",
    command: `drama speak --model drama-q4 --out x.wav --text "<|speaker:0|> [sighs] Fine. If we miss it, it's on you."`,
    outputLang: "text",
    output: `loaded 21.06 GB of weights in 4.4s
ready in 5.3s, GPU memory 21.43 / 23.60 GiB
2 sentence(s), 132 prompt rows -> 96 frames (4.46s of audio) in 0.65s (6.8x real time), first audio after 138 ms
wrote x.wav`,
  },
  cardsTitle: "One binary, one crate",
  cardsLead: "A command-line speaker and the engine behind it.",
  cards: [
    {
      name: "drama",
      href: "/docs/cli",
      tagline: "Text to speech, files or speakers",
      body: "Write a WAV, or stream to PipeWire while generating. Voice cloning from reference clips, speaker tags, inline emotion tags, and an interactive mode that keeps the model loaded.",
    },
    {
      name: "drama-rs",
      href: "/docs/library",
      tagline: "The engine as a library",
      body: "Load the weights, build a reference voice, and get audio through a streaming callback. The CUDA kernels are built by the crate's build script.",
    },
  ] as Card[],
  featuresTitle: "35B parameters in 24 GB",
  featuresLead: "Routed experts in int4, everything else int8, and a decode step that is a single CUDA-graph replay.",
  features: [
    {
      title: "Fits one consumer GPU",
      body: "Backbone, codebook tower and codec take about 21.4 GiB, so an RTX 3090, 4090 or 5090 runs the full model.",
    },
    {
      title: "Faster than real time",
      body: "About 5.2 ms per frame, 190 frames a second: 8.9x real time on an RTX 4090.",
    },
    {
      title: "Voice cloning",
      body: "The Fish DAC encoder turns a reference clip and its transcript into prompt codes; pass several for several speakers.",
    },
    {
      title: "Streaming audio",
      body: "LM and codec run overlapped; the first audio chunk plays 50 to 120 ms after prefill.",
    },
    {
      title: "Stable long text",
      body: "Text is generated sentence by sentence, the first sentence becomes the reference voice, and a stalled sentence is retried.",
    },
    {
      title: "Hand-written kernels",
      body: "int8/int4 GEMVs with L2 prefetch, fused MoE, gated DeltaNet, attention and sampling, all in one captured graph.",
    },
    {
      title: "Checked against PyTorch",
      body: "Engine logits are compared per step against a plain-PyTorch reference; int4 against the original FP8 by KL and top-1.",
    },
    {
      title: "Exact codec",
      body: "The streaming decoder matches the full-sequence decoder, at about 200x real time.",
    },
    {
      title: "No Python at run time",
      body: "One binary, the CUDA runtime and cuBLAS. Python is only needed to convert weights yourself.",
    },
  ] as Feature[],
  lib: {
    lead: "The engine is a library: load the weights, optionally build a reference voice, and receive audio through a streaming callback. The CLI sits behind a feature.",
    add: `cargo add drama-rs --no-default-features`,
    caption: "src/main.rs",
    snippet: `use drama::engine::SampleParams;
use drama::tts::{SpeakOpts, Tts};

let mut tts = Tts::load("drama-q4".as_ref(), 4096, 512)?;
let voice = tts.reference("voice.wav".as_ref(), "Exactly what is said in voice.wav.")?;
let opts = SpeakOpts { params: SampleParams::default(), max_new: 2048, first_chunk: 4, chunk: 16, split: true };
let spoken = tts.speak("Hello there.", &[voice], &opts, |chunk| Ok(()))?;
drama::codec::write_wav("hello.wav".as_ref(), &spoken.audio)?;`,
  },
};
