# drama-rs

Fish Audio [Drama 3](https://huggingface.co/fishaudio/drama-3) text-to-speech on a single
24 GB GPU (RTX 4090), or from 6 GB with the routed experts in RAM, in Rust + CUDA. No PyTorch at runtime.

- **Backbone**: Qwen3.5-MoE (35B total / 3B active, 40 layers, gated-DeltaNet linear attention
  + full attention every 4th layer). Routed experts are 4-bit, everything else 8-bit.
- **Codebook tower**: 13 transformer layers that predict codebooks 1–9 of every frame.
- **Codec**: the Fish modded-DAC encoder (voice cloning) and decoder, streaming and exact.

## Footprint and speed (RTX 4090)

| | |
|---|---|
| weights on GPU (LM + codec) | ~21.4 GiB of 24 |
| decode | ~5.2 ms / frame ≈ 190 frames/s ≈ **8.9× real time** (21.5 frames = 1 s of audio) |
| prefill | ~140 ms for a 300-row prompt (text + 10 s reference voice) |
| time to first audio (streaming) | ~50–120 ms after the prompt is ready |
| codec decode | ~200× real time |

Each decode step is one CUDA-graph replay: lm_head → sampling (with RAS) → 13 tower layers
with 9 codebook samplers → backbone step for the new frame. Weight-streaming GEMVs prefetch
the next kernel's weights into L2 to hide launch gaps.

## Smaller GPUs: routed experts in RAM

`--vram-gb N` (C ABI: `drama_load_ex`) caps the VRAM the engine takes, codec included; 0 (the default) takes the
whole model when it fits the free VRAM. When it does not fit, the 18 GB of routed experts stay in page-locked RAM
and the rest of the limit becomes a VRAM cache of experts shared by all layers. A decode step uses 8 of a layer's
256 experts: a kernel inside the CUDA graph looks them up and copies the missing ones from RAM over PCIe, so the
step stays one graph replay. Eviction is GCLOCK (a hit raises a slot's count up to 3, the hand lowers it), all state
on the GPU. The prefill reads cached experts from their slots and streams the others, 64 at a time. A prompt whose
voice part (the reference) is the same as the last one starts from the saved state after it, so only the text is
prefilled. The codec's encoder (it runs once per reference voice) reads its weights from page-locked RAM too, which
leaves 0.4 GiB more to the cache. The output is bit-identical to the experts-in-VRAM run; a smaller cache only means
more copies.

Measured on an RTX 4090 (PCIe 4.0 x16) with a cloned voice, the second and third of three sentences (the cache is
warm, the voice part is reused), `--vram-gb` limiting the engine as a smaller card would:

| `--vram-gb` | expert cache | cache hits | speed | first audio |
|---|---|---|---|---|
| 0 (whole model) | – | – | 7.9× real time | 60 ms |
| 12 | 7.4 GiB | 85% | 4.7× | 220 ms |
| 8 | 3.4 GiB | 61% | 2.95× | 290 ms |
| 6 | 1.4 GiB | 39% | 2.25× | 325 ms |
| 5 | 0.4 GiB | 0% | 1.6× | 365 ms |

The engine needs about 4.8 GiB besides the cache (LM weights 2.8, codec decoder 0.4, buffers 0.7, the codec's working
memory 0.75), so a 6 GB card, with about 5 GiB free, runs it. With the default limit (0) the engine takes what is free
less a sixteenth of it (256 MiB to 1 GiB). A cache smaller than one step's 320 experts gets no hits: every expert of
the step is copied.

The first sentence after loading is slower (about 1.1 s to the first audio), as the empty cache fills.

**RAM: at least 32 GB.** The experts take 18.1 GB of page-locked memory. Windows page-locks only about half of the
RAM for one process, so by default the engine page-locks at most 45% of the RAM (75% on Linux; `DRAMA_PIN_CAP_GB`
sets it); the experts past that stay in the mapped model files, and the CPU (AVX2) computes the ones the cache does
not hold. On a 32 GB machine that is about 3 GB of experts: at an 8 GiB limit 2.2× real time, 335 ms to the first
audio (the CPU's arithmetic differs from the GPU's, so the audio is not bit-identical, the speech is the same).
With 16 GB of RAM the experts do not fit at all.

## Quantization

`tools/convert.py` reads the original checkpoint shard by shard:

| tensor | format |
|---|---|
| routed experts (32B params, FP8 in the checkpoint) | int4, asymmetric, group 64, f16 scale+min, MSE clip search |
| attention / DeltaNet / shared expert / tower linears | int8, symmetric, group 128 |
| lm_head | only the 4097 sampleable rows (EOS + semantic band), bf16 |
| embeddings, router | bf16 |

## Install

Prebuilt for Linux x86_64 (CUDA 12 runtime, Ampere or newer) on the
[releases page](https://github.com/apiplant/drama-rs/releases), and packaged:

```bash
brew install apiplant/tap/drama-rs     # Homebrew on Linux
sudo apt install drama-rs              # apt.apiplant.com
sudo pacman -S drama-rs                # apiplant.github.io/pacman
cargo install drama-rs                 # from crates.io (needs nvcc; DRAMA_CUDA_ARCH, default sm_89)
```

Repository setup for apt and pacman is in [`packaging/README.md`](packaging/README.md#using-the-packages).
As a library: `cargo add drama-rs --no-default-features` (drops clap); see `src/lib.rs` for an example.

## Setup

```bash
# 1. weights: download the prebuilt int4 build (no conversion needed)
hf download apiplant/drama-q4 --local-dir models/drama-3-q4

# 2. build (CUDA toolkit in /opt/cuda, or set CUDA_PATH; DRAMA_CUDA_ARCH defaults to sm_89,
#    a comma-separated list such as sm_86,sm_89 builds for several GPUs)
cargo build --release
```

### Converting from the original checkpoint (optional)

To rebuild the quantized weights yourself from the original FP8 checkpoint:

```bash
hf download fishaudio/drama-3 --local-dir models/drama-3   # 42 GB

# needs a CUDA PyTorch; ~1 min per layer is plenty
python tools/convert.py models/drama-3 models/drama-3-q4
python tools/convert_codec.py models/drama-3/codec.pth models/drama-3-q4/codec.safetensors
cp models/drama-3/tokenizer.json models/drama-3-q4/
```

## Usage

```bash
# speak to a file
./target/release/drama speak --model models/drama-3-q4 --text "Hello there." --out hello.wav

# stream to the speakers while generating (PipeWire / pw-play)
./target/release/drama speak --model models/drama-3-q4 --text "Hello there." --play

# clone a voice from a reference clip and its transcript
./target/release/drama speak --model models/drama-3-q4 --play \
  --ref-wav voice.wav --ref-text "Exactly what is said in voice.wav." \
  --text "<|speaker:0|> [sighs] Fine. If we miss it, it's on you."

# on a smaller card: at most 8 GiB of VRAM, the routed experts in RAM
./target/release/drama speak --model models/drama-3-q4 --vram-gb 8 --text "Hello there." --out hello.wav

# keep the model loaded and speak every line typed on stdin
./target/release/drama speak --model models/drama-3-q4 --interactive --play
```

Text is generated one sentence at a time, like the reference server's streaming endpoint.
Without a voice reference, Drama 3 often never ends after a sentence-final full stop when more
text follows. It fills the rest with silence: about a quarter of runs, measured on both the
original FP8 weights and this int4 build. Speaking each sentence on its own avoids that. A
speaker with no `--ref-wav` gets their own first sentence as the reference for the rest, so the
voice stays the same, and a sentence that still stalls is cut at a length cap and said again.
`--no-split` turns all of this off.

Sampling defaults follow the reference server: `--temperature 1.0 --top-k 30 --top-p 0.9`,
plus repetition-aware sampling over the last 16 frames. Multi-speaker text uses
`<|speaker:N|>` tags (N in 0..15); inline tags like `[laughs]` are passed through.

Other commands: `encode` / `decode` (codec only), `bench` (decode speed), `trace` (engine tests).

## Tests

All checks compare against a plain-PyTorch reference (`tools/reference.py`):

```bash
PY=python  # a CUDA PyTorch
$PY tools/make_tiny.py models/drama-3/config.json test/tiny-src --tokenizer models/drama-3/tokenizer.json
$PY tools/convert.py test/tiny-src test/tiny
$PY tools/check_engine.py test/tiny                     # engine logits vs reference, per step
# int4 vs original FP8 (KL, top-1), on a real prompt: random rows make the model emit EOS,
# which never happens mid-utterance in real use and skews the codebook-head comparison
./target/release/drama speak --model models/drama-3-q4 --text "..." --out x.wav --prompt-out test/work/prompt.bin
$PY tools/check_engine.py models/drama-3-q4 --original models/drama-3 --rows-file test/work/prompt.bin --steps 12
$PY tools/check_codec.py models/drama-3/codec.pth clip.wav test/work models/drama-3-q4/codec.safetensors
```

`tools/make_bench_model.py` writes a full-size random model for speed/memory tests without the
download.

## Layout

```
kernels/drama.cu   LM kernels: int8/int4 GEMVs, fused MoE, gated DeltaNet, attention, sampling
kernels/codec.cu   codec kernels: im2col, snake, ConvNeXt, windowed attention, VQ encode
src/engine.rs      prefill (cuBLAS + grouped MoE) and the graph-captured decode step
src/codec.rs       streaming codec decoder, encoder, resampler
src/tts.rs         prompt -> frames -> audio pipeline, LM and codec overlapped
ref/               upstream sources used as the reference (sglang-omni, codec)
```
