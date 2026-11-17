# MLX port (Apple Silicon)

Goal: build and run drama-rs on macOS with inference on MLX/Metal instead of CUDA.

## Stage 2 (done): Metal backend runs end-to-end (Apple Silicon)

Implementation choice: the `metal` crate (raw Metal) instead of `mlx-rs`.
Rationale: `mlx-rs` builds the whole MLX C++ library from source and still
would not provide the custom int4/int8 kernels this engine needs — Metal is
what MLX itself runs on, and custom MSL kernels are required either way.
The `metal` backend covers both roles: MSL for the hot kernels, CPU reference
code for the rest (buffers are host-visible unified memory, so both see the
same data).

What works (`cargo build --features metal`, macOS):

- Runtime: device buffers, copies, memset, streams, mem-info, graph capture
  (recorded as closure lists, replayed eagerly — engine semantics preserved).
- MSL kernels, each verified against CPU oracles in `tests/metal_kernels.rs`:
  `rmsnorm` (incl. null-output variants), `gemv_bf16` (plain + fused norm),
  `dequant_q8`, `f32_to_bf16`, `swiglu_rows`.
- CPU reference ops: `gemm`, `gemv_q8`, `gemv_q8_swiglu`, `router_topk`,
  `moe_decode`, `moe_topk/count/permute/combine`, `moe_prefill`, `embed`,
  `tower_in`, `qk_norm_rope`, `attn_prefill/decode`, `conv`, `gdn`,
  `sample_main/head`, `frame_end`.
- `tests/fake_tiny.rs`: a shape-consistent synthetic model runs full prefill
  + decode steps with finite logits (all backbone/tower ops exercised).
- `cargo test` (stub), `cargo test --features metal` (14 tests),
  `cargo check --features cuda` — all green; CUDA path untouched.

Deliberately deferred:

- Codec (`kernels/codec.cu`, 13 fns): `speak`/`decode`/`encode` fail with a
  clear error until ported. Next after the LM path is fully validated.
- Sampler RNG is distribution-equivalent to curand Philox, not bit-identical
  (validation compares logits, never samples).
- Performance: decode GEMVs and prefill GEMMs run on the CPU in v1 — correct
  but slow (~14 s for 3 tiny-model steps). Next: Metal `gemv_q8` (CPU code is
  the oracle), MPS/blocked prefill GEMM, then async command buffers.

## Stage 1 (done): the crate builds on machines without CUDA

- `cuda` cargo feature (off by default). With it: `build.rs` compiles
  `kernels/*.cu` with `nvcc` and links `cudart`/`cublas`, exactly as before.
  Without it: no native code, no CUDA link.
- `src/ffi.rs` without `cuda` provides **stub functions with identical
  signatures** (same names, params, return types). `engine.rs`, `codec.rs`,
  `model.rs` and `gpu.rs` compile **unchanged**; any inference attempt fails at
  runtime via the existing `check`/`launched` paths with:
  `CUDA backend not compiled into this binary (rebuild with --features cuda)`.
- Release builds/CI and docs pass `--features cuda`, so Linux artifacts are
  unaffected.

Build matrix:

| command | result |
|---|---|
| `cargo build` (macOS, no CUDA) | builds, runs, inference errors cleanly |
| `cargo build --features cuda` (Linux + nvcc) | full CUDA binary, as before |
| `cargo test` | host-only tests pass (no GPU needed) |

## Why the port is kernel-by-kernel

The decode loop (`Engine::record_step`) replays one captured CUDA graph per
frame; every heavy op is a hand-written kernel for a custom quantization:

- routed experts: **int4 asymmetric, group 64** (`Q4`)
- everything else linear: **int8 symmetric, group 128** (`Q8`)

Neither dtype exists in MLX, so the hot GEMV paths need **custom Metal
kernels** regardless — MLX only covers the "light" ops (norms, RoPE, softmax,
sampling, permutes). Each `dr_*` function in `src/ffi.rs` is one porting unit;
engine logic above it stays untouched.

## Function inventory (58 in `src/ffi.rs`)

### Runtime (map to MLX device / command buffer / unified memory)

| fn | MLX counterpart |
|---|---|
| `dr_malloc/free/memset` | `MTLBuffer` allocation (unified memory, host-visible) |
| `dr_h2d/d2h/d2d` | buffer copies / `synchronize` (often no-ops on unified memory) |
| `dr_host_alloc/free` | page-aligned host buffers (plain Rust `Vec` suffices) |
| `dr_mem_info` | `MTLDevice` recommended working-set query |
| `dr_stream_create/sync` | `MTLCommandQueue` + `waitUntilCompleted` |
| `dr_graph_begin/end/launch/destroy` | record command buffers eagerly, or `mlx::compile`; the prefetch args (`pf/pfb`) become prefetch hints or are dropped |
| `dr_cublas_create/gemm/gemm_f16` | `mlx::matmul` (bf16/f16) |
| `dr_error_string/last_error` | `MTLDevice` error strings / `NSError` |

### Backbone decode (hottest first)

- `dr_gemv_q8`, `dr_gemv_q8_swiglu`, `dr_gemv_bf16` — custom Metal kernels
  (weight-streaming + L2 prefetch strategy needs a Metal equivalent).
- `dr_router_topk`, `dr_moe_decode` — top-k on CPU/Metal + custom q4 MoE kernel.
- `dr_rmsnorm` — trivial Metal kernel or MLX op.
- `dr_qk_norm_rope`, `dr_attn_decode` — MLA-style decode attention; single-query
  Metal kernel, KV cache in unified memory.
- `dr_conv`, `dr_gdn` — DeltaNet recurrence, custom Metal kernel.
- `dr_embed`, `dr_tower_in` — gathers; Metal kernel.
- `dr_sample_main/head`, `dr_frame_end`, `dr_sample_init` — top-k/top-p + RAS on CPU.

### Prefill (correctness first, speed later)

`dr_dequant_q8`, `dr_f32_to_bf16`, `dr_swiglu_rows`, `dr_moe_topk/count/permute/combine`,
`dr_moe_prefill`, `dr_attn_prefill` — prefill already dequantizes + uses GEMM,
so most of it maps to `mlx::matmul` + small kernels; port after decode works.

### Codec (`kernels/codec.cu`)

`dr_rvq_sum/im2col/vq_encode/tconv_gather/epilogue/dwconv/layernorm/rmsnorm_plain/codec_rope/codec_attn/swiglu16/conv_out`
— the codec is small (1024-wide, chunked); plain MLX ops + a few gather kernels.

## Validation

The repo already has bit-exact references; use them per ported group:

```bash
python tools/make_tiny.py ...          # tiny weights for fast iteration
python tools/check_engine.py ...       # engine logits vs torch reference, per step
python tools/check_codec.py ...        # codec vs torch reference
```

Bring-up order: runtime → `gemv_bf16` (lm_head) → sampler → attention → MoE →
rest of backbone → prefill → codec. Each step must reproduce CUDA logits before
moving on.

## Memory note

The int4/int8 model is ~21.4 GiB on GPU; a 32 GB Mac holds it in unified memory
plus KV cache with room to spare. A dequantize-everything-to-f16 shortcut would
need ~70 GB and is **not** viable — the custom quant formats must be kept,
which is why the GEMV kernels stay custom Metal.

## Checklist

- [x] `cuda` feature; stub `ffi.rs`; macOS build + tests green
- [x] MLX runtime (`Stream`/`DevBuf`/`Graph` on Metal; closures replayed eagerly)
- [x] decode GEMV + sampler (MSL `gemv_bf16`; CPU `gemv_q8`/MoE/attention/GDN)
- [x] attention + MoE + DeltaNet (CPU reference, fake-tiny validated)
- [x] prefill path (CPU GEMM + dequant; slow, MPS later)
- [ ] codec encode/decode
- [ ] parity run vs CUDA on `check_engine.py` / `check_codec.py` (needs CUDA box)
- [ ] Metal `gemv_q8` + prefill GEMM acceleration + async execution
