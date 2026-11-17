#![allow(clippy::too_many_arguments)]
//! Raw bindings to the launchers and runtime wrappers in kernels/drama.cu.
//!
//! With the `cuda` feature these are the real `kernels/*.cu` entry points; with
//! `metal` they are Apple-Silicon implementations (`src/metal_backend.rs`); without
//! either (e.g. a plain macOS checkout) each one is a stub with an identical signature
//! that reports the missing backend, so the crate still builds and links.
//! See docs/MLX-PORT.md.

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("features `cuda` and `metal` are mutually exclusive backends");

use std::ffi::c_void;
#[cfg(any(feature = "cuda", all(not(feature = "cuda"), not(feature = "metal"))))]
use std::ffi::c_char;

pub type Stream = *mut c_void;
pub type GraphExec = *mut c_void;
pub type Bf16 = u16;
pub type Half = u16;

#[cfg(feature = "cuda")]
extern "C" {
    pub fn dr_error_string(e: i32) -> *const std::ffi::c_char;
    pub fn dr_last_error() -> i32;
    pub fn dr_malloc(p: *mut *mut c_void, n: usize) -> i32;
    pub fn dr_free(p: *mut c_void) -> i32;
    pub fn dr_host_alloc(p: *mut *mut c_void, n: usize) -> i32;
    pub fn dr_host_free(p: *mut c_void) -> i32;
    pub fn dr_h2d(d: *mut c_void, h: *const c_void, n: usize, st: Stream) -> i32;
    pub fn dr_d2h(h: *mut c_void, d: *const c_void, n: usize, st: Stream) -> i32;
    pub fn dr_d2d(d: *mut c_void, s: *const c_void, n: usize, st: Stream) -> i32;
    pub fn dr_memset(d: *mut c_void, v: i32, n: usize, st: Stream) -> i32;
    pub fn dr_stream_create(s: *mut Stream) -> i32;
    pub fn dr_stream_sync(s: Stream) -> i32;
    pub fn dr_mem_info(free: *mut usize, total: *mut usize) -> i32;
    pub fn dr_graph_begin(s: Stream) -> i32;
    pub fn dr_graph_end(s: Stream, exec: *mut GraphExec) -> i32;
    pub fn dr_graph_launch(exec: GraphExec, s: Stream) -> i32;
    pub fn dr_graph_destroy(exec: GraphExec) -> i32;
    pub fn dr_cublas_create(h: *mut *mut c_void, s: Stream) -> i32;
    pub fn dr_gemm(h: *mut c_void, a: *const Bf16, w: *const Bf16, c: *mut f32, m: i32, n: i32, k: i32, beta: f32) -> i32;

    pub fn dr_gemv_q8(
        w: *const i8, s: *const Half, x: *const f32, res: *const f32, y: *mut f32, n: i32, k: i32, nw: *const f32, eps: f32, pf: *const c_char, pf_bytes: usize,
        st: Stream,
    ) -> i32;
    pub fn dr_gemv_q8_swiglu(
        w: *const i8, s: *const Half, x: *const f32, y: *mut f32, i: i32, k: i32, nw: *const f32, eps: f32, pf: *const c_char, pf_bytes: usize, st: Stream,
    ) -> i32;
    pub fn dr_gemv_bf16(w: *const Bf16, x: *const f32, y: *mut f32, n: i32, k: i32, nw: *const f32, eps: f32, pf: *const c_char, pf_bytes: usize, st: Stream) -> i32;
    pub fn dr_moe_topk(logits: *const f32, rows: i32, e: i32, topk: i32, stride: i32, idx: *mut i32, wts: *mut f32, sgate: *mut f32, st: Stream) -> i32;
    pub fn dr_moe_decode(
        gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, sgu_q: *const i8, sgu_s: *const Half, sd_q: *const i8,
        sd_s: *const Half, x: *const f32, idx: *const i32, wts: *const f32, sgate: *const f32, res: *const f32, act: *mut f32, y: *mut f32,
        h: i32, i: i32, topk: i32, nw: *const f32, eps: f32, pf: *const c_char, pf_bytes: usize, st: Stream,
    ) -> i32;
    pub fn dr_router_topk(
        w: *const Bf16, x: *const f32, nw: *const f32, eps: f32, logits: *mut f32, e: i32, topk: i32, k: i32, idx: *mut i32, wts: *mut f32,
        sgate: *mut f32, counter: *mut u32, gu_q: *const c_char, gu_sm: *const c_char, d_q: *const c_char, d_sm: *const c_char, gu_q_b: usize,
        gu_sm_b: usize, d_q_b: usize, d_sm_b: usize, st: Stream,
    ) -> i32;
    pub fn dr_moe_prefill(
        gu_q: *const u8, gu_sm: *const Half, d_q: *const u8, d_sm: *const Half, xp: *const Bf16, offsets: *const i32, counts: *const i32,
        act: *mut Bf16, y: *mut f32, e: i32, h: i32, i: i32, st: Stream,
    ) -> i32;
    pub fn dr_moe_count(idx: *const i32, n: i32, counts: *mut i32, st: Stream) -> i32;
    pub fn dr_moe_permute(
        idx: *const i32, rows: i32, topk: i32, offsets: *const i32, fill: *mut i32, slot_of: *mut i32, x: *const Bf16, xp: *mut Bf16, h: i32,
        st: Stream,
    ) -> i32;
    pub fn dr_moe_combine(
        x: *mut f32, y: *const f32, slot_of: *const i32, wts: *const f32, sgate: *const f32, shared: *const f32, rows: i32, topk: i32, h: i32,
        st: Stream,
    ) -> i32;
    pub fn dr_dequant_q8(w: *const i8, s: *const Half, out: *mut Bf16, n: i32, k: i32, st: Stream) -> i32;
    pub fn dr_f32_to_bf16(x: *const f32, y: *mut Bf16, n: usize, st: Stream) -> i32;
    pub fn dr_swiglu_rows(h: *const f32, y: *mut Bf16, rows: i32, i: i32, st: Stream) -> i32;
    pub fn dr_rmsnorm(x: *const f32, w: *const f32, y: *mut f32, yb: *mut Bf16, rows: i32, d: i32, eps: f32, st: Stream) -> i32;
    pub fn dr_embed(
        rows_emb: *const Bf16, cand: *const Bf16, frames: *const i32, fstride: i32, cb_emb: *const Bf16, x: *mut f32, rows: i32, d: i32,
        sem0: i32, sem1: i32, eos: i32, cb: i32, nb: i32, st: Stream,
    ) -> i32;
    pub fn dr_tower_in(
        h: *const f32, frames: *const i32, fstride: i32, code_col: i32, in_frames: *const i32, in_stride: i32, tok_emb: *const Bf16,
        cb_emb: *const Bf16, enorm: *const f32, hnorm: *const f32, pre_norm: *const f32, y: *mut f32, yb: *mut Bf16, rows: i32, d: i32, sem0: i32, sem1: i32,
        cb: i32, ncb: i32, nb: i32, eps: f32, st: Stream,
    ) -> i32;
    pub fn dr_qk_norm_rope(
        qkv: *const f32, rs: i32, rows: i32, h: i32, hkv: i32, hd: i32, qn: *const f32, kn: *const f32, inv_freq: *const f32, rot: i32,
        interleaved: i32, pos_base: *const i32, pos_off: i32, q_out: *mut f32, kc: *mut Bf16, vc: *mut Bf16, eps: f32, st: Stream,
    ) -> i32;
    pub fn dr_attn_prefill(
        q: *const f32, kc: *const Bf16, vc: *const Bf16, gate: *const f32, gate_stride: i32, rows: i32, h: i32, hkv: i32, hd: i32,
        pos_base: *const i32, pos_off: i32, o: *mut f32, st: Stream,
    ) -> i32;
    pub fn dr_attn_decode(
        q: *const f32, kc: *const Bf16, vc: *const Bf16, gate: *const f32, h: i32, hkv: i32, hd: i32, pos_base: *const i32, pos_off: i32,
        part: *mut f32, o: *mut f32, st: Stream,
    ) -> i32;
    pub fn dr_attn_part_floats() -> i32;
    pub fn dr_conv(input: *const f32, in_stride: i32, w: *const f32, state: *mut f32, out: *mut f32, n: i32, c: i32, st: Stream) -> i32;
    pub fn dr_gdn(
        conv: *const f32, proj: *const f32, ps: i32, z_off: i32, b_off: i32, a_off: i32, a_log: *const f32, dt_bias: *const f32,
        norm_w: *const f32, state: *mut f32, o: *mut f32, n: i32, nk: i32, nv: i32, dk: i32, dv: i32, eps: f32, st: Stream,
    ) -> i32;
    pub fn dr_sample_init() -> i32;
    pub fn dr_sample_main(
        logits: *const f32, n: i32, prm: *const c_void, ras: *const i32, ras_len: i32, step: *const u64, eos: i32, sem0: i32,
        cb: i32, frame: *mut i32, st: Stream,
    ) -> i32;
    pub fn dr_sample_head(
        logits: *const f32, n: i32, prm: *const c_void, stream: i32, step: *const u64, frame: *mut i32, col: i32, st: Stream,
    ) -> i32;
    pub fn dr_frame_end(
        next: *const i32, in_frame: *mut i32, ras: *mut i32, ras_len: i32, pos: *mut i32, step: *mut u64, hist: *mut i32,
        hist_len: *mut i32, hist_cap: i32, width: i32, st: Stream,
    ) -> i32;
}

// codec kernels (kernels/codec.cu)
#[cfg(feature = "cuda")]
extern "C" {
    pub fn dr_rvq_sum(codes: *const i32, frames: i32, ncb: i32, table: *const f32, out: *mut f32, d: i32, sem_size: i32, res_size: i32, st: Stream) -> i32;
    pub fn dr_im2col(x: *const f32, n: i32, c: i32, k: i32, d: i32, s: i32, alpha: *const f32, out: *mut Half, st: Stream) -> i32;
    pub fn dr_vq_encode(r: *mut f32, rows: i32, d: i32, in_w: *const f32, in_b: *const f32, codebook_n: *const f32, n: i32, table: *const f32, codes: *mut i32, code_stride: i32, st: Stream) -> i32;
    pub fn dr_tconv_gather(x: *const f32, n: i32, c: i32, two: i32, alpha: *const f32, out: *mut Half, st: Stream) -> i32;
    pub fn dr_epilogue(
        y: *const f32, n: usize, c: i32, bias: *const f32, scale: *const f32, res: *const f32, act: i32, out: *mut f32, out16: *mut Half,
        st: Stream,
    ) -> i32;
    pub fn dr_dwconv(x: *const f32, t: i32, c: i32, k: i32, w: *const f32, b: *const f32, out: *mut f32, st: Stream) -> i32;
    pub fn dr_layernorm(x: *const f32, rows: i32, c: i32, w: *const f32, b: *const f32, eps: f32, out: *mut Half, st: Stream) -> i32;
    pub fn dr_rmsnorm_plain(x: *const f32, rows: i32, c: i32, w: *const f32, eps: f32, out: *mut f32, out16: *mut Half, st: Stream) -> i32;
    pub fn dr_codec_rope(qkv: *const f32, n: i32, h: i32, inv_freq: *const f32, t0: i32, ring: i32, q: *mut f32, kc: *mut f32, vc: *mut f32, st: Stream) -> i32;
    pub fn dr_codec_attn(q: *const f32, kc: *const f32, vc: *const f32, n: i32, h: i32, window: i32, t0: i32, ring: i32, out: *mut Half, st: Stream) -> i32;
    pub fn dr_swiglu16(h: *const f32, y: *mut Half, rows: i32, i: i32, st: Stream) -> i32;
    pub fn dr_conv_out(x: *const f32, n: i32, c: i32, k: i32, alpha: *const f32, w: *const f32, b: *const f32, out: *mut f32, st: Stream) -> i32;
    pub fn dr_gemm_f16(h: *mut c_void, a: *const Half, w: *const Half, c: *mut f32, m: i32, n: i32, k: i32, beta: f32) -> i32;
}

// ---------------------------------------------------------------------------
// Stubs used without the `cuda` feature (e.g. Apple Silicon): identical
// signatures, so engine.rs / codec.rs / gpu.rs compile unchanged; every call
// fails at runtime with the message from dr_error_string. The MLX port
// (docs/MLX-PORT.md) replaces these one group at a time.
// ---------------------------------------------------------------------------
#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
static UNAVAILABLE_MSG: &[u8] = b"CUDA backend not compiled into this binary (rebuild with --features cuda)\0";

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_error_string(_e: i32) -> *const std::ffi::c_char {
    UNAVAILABLE_MSG.as_ptr() as *const c_char
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_last_error() -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_malloc(_p: *mut *mut c_void, _n: usize) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_free(_p: *mut c_void) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_host_alloc(_p: *mut *mut c_void, _n: usize) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_host_free(_p: *mut c_void) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_h2d(_d: *mut c_void, _h: *const c_void, _n: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_d2h(_h: *mut c_void, _d: *const c_void, _n: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_d2d(_d: *mut c_void, _s: *const c_void, _n: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_memset(_d: *mut c_void, _v: i32, _n: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_stream_create(_s: *mut Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_stream_sync(_s: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_mem_info(_free: *mut usize, _total: *mut usize) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_graph_begin(_s: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_graph_end(_s: Stream, _exec: *mut GraphExec) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_graph_launch(_exec: GraphExec, _s: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_graph_destroy(_exec: GraphExec) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_cublas_create(_h: *mut *mut c_void, _s: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_gemm(_h: *mut c_void, _a: *const Bf16, _w: *const Bf16, _c: *mut f32, _m: i32, _n: i32, _k: i32, _beta: f32) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_gemv_q8(_w: *const i8, _s: *const Half, _x: *const f32, _res: *const f32, _y: *mut f32, _n: i32, _k: i32, _nw: *const f32, _eps: f32, _pf: *const c_char, _pf_bytes: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_gemv_q8_swiglu(_w: *const i8, _s: *const Half, _x: *const f32, _y: *mut f32, _i: i32, _k: i32, _nw: *const f32, _eps: f32, _pf: *const c_char, _pf_bytes: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_gemv_bf16(_w: *const Bf16, _x: *const f32, _y: *mut f32, _n: i32, _k: i32, _nw: *const f32, _eps: f32, _pf: *const c_char, _pf_bytes: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_moe_topk(_logits: *const f32, _rows: i32, _e: i32, _topk: i32, _stride: i32, _idx: *mut i32, _wts: *mut f32, _sgate: *mut f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_moe_decode(_gu_q: *const u8, _gu_sm: *const Half, _d_q: *const u8, _d_sm: *const Half, _sgu_q: *const i8, _sgu_s: *const Half, _sd_q: *const i8, _sd_s: *const Half, _x: *const f32, _idx: *const i32, _wts: *const f32, _sgate: *const f32, _res: *const f32, _act: *mut f32, _y: *mut f32, _h: i32, _i: i32, _topk: i32, _nw: *const f32, _eps: f32, _pf: *const c_char, _pf_bytes: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_router_topk(_w: *const Bf16, _x: *const f32, _nw: *const f32, _eps: f32, _logits: *mut f32, _e: i32, _topk: i32, _k: i32, _idx: *mut i32, _wts: *mut f32, _sgate: *mut f32, _counter: *mut u32, _gu_q: *const c_char, _gu_sm: *const c_char, _d_q: *const c_char, _d_sm: *const c_char, _gu_q_b: usize, _gu_sm_b: usize, _d_q_b: usize, _d_sm_b: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_moe_prefill(_gu_q: *const u8, _gu_sm: *const Half, _d_q: *const u8, _d_sm: *const Half, _xp: *const Bf16, _offsets: *const i32, _counts: *const i32, _act: *mut Bf16, _y: *mut f32, _e: i32, _h: i32, _i: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_moe_count(_idx: *const i32, _n: i32, _counts: *mut i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_moe_permute(_idx: *const i32, _rows: i32, _topk: i32, _offsets: *const i32, _fill: *mut i32, _slot_of: *mut i32, _x: *const Bf16, _xp: *mut Bf16, _h: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_moe_combine(_x: *mut f32, _y: *const f32, _slot_of: *const i32, _wts: *const f32, _sgate: *const f32, _shared: *const f32, _rows: i32, _topk: i32, _h: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_dequant_q8(_w: *const i8, _s: *const Half, _out: *mut Bf16, _n: i32, _k: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_f32_to_bf16(_x: *const f32, _y: *mut Bf16, _n: usize, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_swiglu_rows(_h: *const f32, _y: *mut Bf16, _rows: i32, _i: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_rmsnorm(_x: *const f32, _w: *const f32, _y: *mut f32, _yb: *mut Bf16, _rows: i32, _d: i32, _eps: f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_embed(_rows_emb: *const Bf16, _cand: *const Bf16, _frames: *const i32, _fstride: i32, _cb_emb: *const Bf16, _x: *mut f32, _rows: i32, _d: i32, _sem0: i32, _sem1: i32, _eos: i32, _cb: i32, _nb: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_tower_in(_h: *const f32, _frames: *const i32, _fstride: i32, _code_col: i32, _in_frames: *const i32, _in_stride: i32, _tok_emb: *const Bf16, _cb_emb: *const Bf16, _enorm: *const f32, _hnorm: *const f32, _pre_norm: *const f32, _y: *mut f32, _yb: *mut Bf16, _rows: i32, _d: i32, _sem0: i32, _sem1: i32, _cb: i32, _ncb: i32, _nb: i32, _eps: f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_qk_norm_rope(_qkv: *const f32, _rs: i32, _rows: i32, _h: i32, _hkv: i32, _hd: i32, _qn: *const f32, _kn: *const f32, _inv_freq: *const f32, _rot: i32, _interleaved: i32, _pos_base: *const i32, _pos_off: i32, _q_out: *mut f32, _kc: *mut Bf16, _vc: *mut Bf16, _eps: f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_attn_prefill(_q: *const f32, _kc: *const Bf16, _vc: *const Bf16, _gate: *const f32, _gate_stride: i32, _rows: i32, _h: i32, _hkv: i32, _hd: i32, _pos_base: *const i32, _pos_off: i32, _o: *mut f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_attn_decode(_q: *const f32, _kc: *const Bf16, _vc: *const Bf16, _gate: *const f32, _h: i32, _hkv: i32, _hd: i32, _pos_base: *const i32, _pos_off: i32, _part: *mut f32, _o: *mut f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_attn_part_floats() -> i32 {
    0
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_conv(_input: *const f32, _in_stride: i32, _w: *const f32, _state: *mut f32, _out: *mut f32, _n: i32, _c: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_gdn(_conv: *const f32, _proj: *const f32, _ps: i32, _z_off: i32, _b_off: i32, _a_off: i32, _a_log: *const f32, _dt_bias: *const f32, _norm_w: *const f32, _state: *mut f32, _o: *mut f32, _n: i32, _nk: i32, _nv: i32, _dk: i32, _dv: i32, _eps: f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_sample_init() -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_sample_main(_logits: *const f32, _n: i32, _prm: *const c_void, _ras: *const i32, _ras_len: i32, _step: *const u64, _eos: i32, _sem0: i32, _cb: i32, _frame: *mut i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_sample_head(_logits: *const f32, _n: i32, _prm: *const c_void, _stream: i32, _step: *const u64, _frame: *mut i32, _col: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_frame_end(_next: *const i32, _in_frame: *mut i32, _ras: *mut i32, _ras_len: i32, _pos: *mut i32, _step: *mut u64, _hist: *mut i32, _hist_len: *mut i32, _hist_cap: i32, _width: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_rvq_sum(_codes: *const i32, _frames: i32, _ncb: i32, _table: *const f32, _out: *mut f32, _d: i32, _sem_size: i32, _res_size: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_im2col(_x: *const f32, _n: i32, _c: i32, _k: i32, _d: i32, _s: i32, _alpha: *const f32, _out: *mut Half, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_vq_encode(_r: *mut f32, _rows: i32, _d: i32, _in_w: *const f32, _in_b: *const f32, _codebook_n: *const f32, _n: i32, _table: *const f32, _codes: *mut i32, _code_stride: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_tconv_gather(_x: *const f32, _n: i32, _c: i32, _two: i32, _alpha: *const f32, _out: *mut Half, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_epilogue(_y: *const f32, _n: usize, _c: i32, _bias: *const f32, _scale: *const f32, _res: *const f32, _act: i32, _out: *mut f32, _out16: *mut Half, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_dwconv(_x: *const f32, _t: i32, _c: i32, _k: i32, _w: *const f32, _b: *const f32, _out: *mut f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_layernorm(_x: *const f32, _rows: i32, _c: i32, _w: *const f32, _b: *const f32, _eps: f32, _out: *mut Half, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_rmsnorm_plain(_x: *const f32, _rows: i32, _c: i32, _w: *const f32, _eps: f32, _out: *mut f32, _out16: *mut Half, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_codec_rope(_qkv: *const f32, _n: i32, _h: i32, _inv_freq: *const f32, _t0: i32, _ring: i32, _q: *mut f32, _kc: *mut f32, _vc: *mut f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_codec_attn(_q: *const f32, _kc: *const f32, _vc: *const f32, _n: i32, _h: i32, _window: i32, _t0: i32, _ring: i32, _out: *mut Half, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_swiglu16(_h: *const f32, _y: *mut Half, _rows: i32, _i: i32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_conv_out(_x: *const f32, _n: i32, _c: i32, _k: i32, _alpha: *const f32, _w: *const f32, _b: *const f32, _out: *mut f32, _st: Stream) -> i32 {
    1
}

#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub unsafe fn dr_gemm_f16(_h: *mut c_void, _a: *const Half, _w: *const Half, _c: *mut f32, _m: i32, _n: i32, _k: i32, _beta: f32) -> i32 {
    1
}

// Metal backend (Apple Silicon): real implementations live in metal_backend.rs.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub use crate::metal_backend::*;
