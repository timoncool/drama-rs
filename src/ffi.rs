//! Raw bindings to the launchers and runtime wrappers in kernels/drama.cu.

#![allow(clippy::too_many_arguments)]

use std::ffi::{c_char, c_void};

pub type Stream = *mut c_void;
pub type GraphExec = *mut c_void;
pub type Bf16 = u16;
pub type Half = u16;

extern "C" {
    pub fn dr_error_string(e: i32) -> *const std::ffi::c_char;
    pub fn dr_last_error() -> i32;
    pub fn dr_malloc(p: *mut *mut c_void, n: usize) -> i32;
    pub fn dr_free(p: *mut c_void) -> i32;
    pub fn dr_host_alloc(p: *mut *mut c_void, n: usize) -> i32;
    pub fn dr_host_free(p: *mut c_void) -> i32;
    pub fn dr_host_alloc_mapped(p: *mut *mut c_void, dev: *mut *mut c_void, n: usize) -> i32;
    pub fn dr_host_func(s: Stream, f: unsafe extern "C" fn(*mut c_void), user: *mut c_void) -> i32;
    pub fn dr_moe_mask_misses(fill: *const i32, sidx: *const i32, wts: *const f32, zero_slot: i32, topk: i32, sidx2: *mut i32, wts2: *mut f32, st: Stream) -> i32;
    pub fn dr_add_vec(x: *mut f32, y: *const f32, n: i32, st: Stream) -> i32;
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
    pub fn dr_moe_decode_fp8(
        gu_w: *const u8, gu_s: *const f32, d_w: *const u8, d_s: *const f32, sgu_q: *const i8, sgu_s: *const Half, sd_q: *const i8,
        sd_s: *const Half, x: *const f32, idx: *const i32, wts: *const f32, sgate: *const f32, res: *const f32, act: *mut f32, y: *mut f32,
        h: i32, i: i32, topk: i32, nw: *const f32, eps: f32, pf: *const c_char, pf_bytes: usize, st: Stream,
    ) -> i32;
    pub fn dr_moe_prefill_fp8(
        gu_w: *const u8, gu_s: *const f32, d_w: *const u8, d_s: *const f32, xp: *const Bf16, offsets: *const i32, counts: *const i32,
        act: *mut Bf16, y: *mut f32, e: i32, h: i32, i: i32, st: Stream,
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
    pub fn dr_moe_prefill_where(
        c_gq: *const u8, c_gsm: *const Half, c_dq: *const u8, c_dsm: *const Half, s_gq: *const u8, s_gsm: *const Half, s_dq: *const u8,
        s_dsm: *const Half, where_: *const i32, xp: *const Bf16, offsets: *const i32, counts: *const i32, act: *mut Bf16, y: *mut f32, e: i32,
        h: i32, i: i32, st: Stream,
    ) -> i32;
    pub fn dr_moe_count(idx: *const i32, n: i32, counts: *mut i32, st: Stream) -> i32;
    pub fn dr_expert_slots(
        idx: *const i32, topk: i32, key0: i32, s: i32, map: *mut i32, owner: *mut i32, refb: *mut u32, tag: *mut u32, state: *mut u32, sidx: *mut i32,
        fill: *mut i32, max_ref: u32, insert: i32, st: Stream,
    ) -> i32;
    pub fn dr_expert_fill(
        idx: *const i32, fill: *const i32, topk: i32, g_q: *const c_void, g_sm: *const c_void, d_q: *const c_void, d_sm: *const c_void, c_gq: *mut c_void,
        c_gsm: *mut c_void, c_dq: *mut c_void, c_dsm: *mut c_void, b_gq: usize, b_gsm: usize, b_dq: usize, b_dsm: usize, st: Stream,
    ) -> i32;
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
