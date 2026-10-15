// CUDA kernels for drama-rs: Fish Audio Drama 3 inference on a single consumer GPU.
//
// Conventions
//   * Activations are f32 unless a name says bf16; weights are int8 (group 128, f16 scale),
//     int4 (group 64, f16 scale + f16 min), bf16 or f32.
//   * "decode" kernels read positions and lengths from device memory so that a whole
//     decode step can be captured once in a CUDA graph and replayed.
//   * Every launcher is extern "C" and takes the stream last.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

#define DEV __device__ __forceinline__
#define API extern "C"

// ------------------------------------------------------------------------------------------
// helpers

DEV float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

DEV float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over the block (blockDim.x a multiple of 32); every thread gets the total.
// red must hold 32 floats; the leading barrier lets callers reuse red back to back.
DEV float block_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float t = lane < nw ? red[lane] : 0.f;
  return warp_sum(t);
}

DEV float block_max(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float t = lane < nw ? red[lane] : -INFINITY;
  return warp_max(t);
}

DEV float silu(float x) { return x / (1.f + expf(-x)); }
DEV float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
DEV float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
DEV float bf_lo(int v) { return __int_as_float(v << 16); }
DEV float bf_hi(int v) { return __int_as_float(v & 0xffff0000); }
DEV int pick4(const int4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; }

// ------------------------------------------------------------------------------------------
// GEMV row dots: one warp computes dot(W[row], x) with x staged in shared memory as float4.

// int8, group 128: a 16-byte chunk holds 16 weights of one group.
template <int K>
DEV float dot_q8(const int8_t* __restrict__ W, const __half* __restrict__ S, size_t row,
                 const float4* xs4, int lane) {
  constexpr int CH = K / 16;
  constexpr int PL = (CH + 31) / 32;
  const int4* wr = reinterpret_cast<const int4*>(W + row * K);
  const __half* sr = S + row * (K / 128);
  int4 v[PL];
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int c = lane + 32 * i;
    if (c < CH) v[i] = __ldg(wr + c);
  }
  const int sw = lane >> 1;  // rotation that keeps the float4 reads bank-conflict free
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int c = lane + 32 * i;
    if (c < CH) {
      float d = 0.f;
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int j = (jj + sw) & 3;
        const int wv = pick4(v[i], j);
        const float4 xv = xs4[c * 4 + j];
        d += (float)(int8_t)(wv) * xv.x + (float)(int8_t)(wv >> 8) * xv.y +
             (float)(int8_t)(wv >> 16) * xv.z + (float)(int8_t)(wv >> 24) * xv.w;
      }
      acc += d * __half2float(sr[c >> 3]);
    }
  }
  return warp_sum(acc);
}

// bf16 weights: a 16-byte chunk holds 8 weights.
template <int K>
DEV float dot_bf16(const bf16* __restrict__ W, size_t row, const float4* xs4, int lane) {
  constexpr int CH = K / 8;
  constexpr int PL = (CH + 31) / 32;
  const int4* wr = reinterpret_cast<const int4*>(W + row * K);
  int4 v[PL];
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int c = lane + 32 * i;
    if (c < CH) v[i] = __ldg(wr + c);
  }
  const int sw = lane >> 2;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int c = lane + 32 * i;
    if (c < CH) {
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int j = (jj + sw) & 1;
        const int a = j ? v[i].z : v[i].x;
        const int b = j ? v[i].w : v[i].y;
        const float4 xv = xs4[c * 2 + j];
        acc += bf_lo(a) * xv.x + bf_hi(a) * xv.y + bf_lo(b) * xv.z + bf_hi(b) * xv.w;
      }
    }
  }
  return warp_sum(acc);
}

// int4 asymmetric, group 64: a 16-byte chunk holds 32 weights; two chunks per group.
// xcs holds the sum of x over each 32-wide chunk, so w = q*s + m costs one FMA per chunk.
template <int K>
DEV float dot_q4(const uint8_t* __restrict__ Q, const __half2* __restrict__ SM, size_t row,
                 const float4* xs4, const float* xcs, int lane) {
  constexpr int CH = K / 32;
  constexpr int PL = (CH + 31) / 32;
  const int4* wr = reinterpret_cast<const int4*>(Q + row * (K / 2));
  const __half2* sr = SM + row * (K / 64);
  int4 v[PL];
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int c = lane + 32 * i;
    if (c < CH) v[i] = __ldg(wr + c);
  }
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < PL; ++i) {
    const int c = lane + 32 * i;
    if (c < CH) {
      float qx = 0.f;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int j = (jj + lane) & 7;
        const int bits = pick4(v[i], j >> 1) >> ((j & 1) * 16);
        const float4 xv = xs4[c * 8 + j];
        qx += (float)(bits & 15) * xv.x + (float)((bits >> 4) & 15) * xv.y +
              (float)((bits >> 8) & 15) * xv.z + (float)((bits >> 12) & 15) * xv.w;
      }
      const float2 sm = __half22float2(sr[c >> 1]);
      acc += sm.x * qx + sm.y * xcs[c];
    }
  }
  return warp_sum(acc);
}

template <int K>
DEV void stage_x(const float* __restrict__ x, float4* xs4) {
  const float4* x4 = reinterpret_cast<const float4*>(x);
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) xs4[i] = x4[i];
  __syncthreads();
}

// Stage x, RMS-normalised with weight nw when nw is non-null (the weight has its "+1"
// folded in). Every block recomputes the norm, which saves a separate launch.
template <int K>
DEV void stage_x_norm(const float* __restrict__ x, const float* __restrict__ nw, float eps, float4* xs4) {
  if (!nw) {
    stage_x<K>(x, xs4);
    return;
  }
  __shared__ float red[32];
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float ss = 0.f;
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) {
    const float4 v = x4[i];
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  const float inv = rsqrtf(block_sum(ss, red) / K + eps);
  const float4* w4 = reinterpret_cast<const float4*>(nw);
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) {
    const float4 v = x4[i], w = w4[i];
    xs4[i] = make_float4(v.x * inv * w.x, v.y * inv * w.y, v.z * inv * w.z, v.w * inv * w.w);
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------
// L2 prefetch of the next kernel's weights: once a block has finished its own work it
// prefetches its share of [pf, pf + pf_bytes), so DRAM stays busy across the launch gap.

DEV void prefetch_share(const char* __restrict__ pf, size_t pf_bytes) {
  if (!pf) return;
  const unsigned nb = gridDim.x * gridDim.y;
  const unsigned b = blockIdx.y * gridDim.x + blockIdx.x;
  const size_t per = ((pf_bytes + nb - 1) / nb + 127) & ~(size_t)127;
  const size_t b0 = (size_t)b * per, b1 = min(pf_bytes, b0 + per);
  for (size_t o = b0 + threadIdx.x * 128; o < b1; o += (size_t)blockDim.x * 128)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + o));
}

// ------------------------------------------------------------------------------------------
// GEMV kernels (M = 1)

constexpr int GEMV_WARPS = 8;
constexpr int GEMV_RPW = 2;

// y[n] = W[n] . x (+ res[n]); y may alias res.
template <int K>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
    k_gemv_q8(const int8_t* __restrict__ W, const __half* __restrict__ S, const float* __restrict__ x,
              const float* res, float* y, int N, const float* __restrict__ nw, float eps, const char* pf,
              size_t pf_bytes) {
  __shared__ float4 xs4[K / 4];
  stage_x_norm<K>(x, nw, eps, xs4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * GEMV_RPW;
#pragma unroll
  for (int r = 0; r < GEMV_RPW; ++r) {
    const int row = row0 + r;
    if (row >= N) break;
    const float d = dot_q8<K>(W, S, row, xs4, lane);
    if (lane == 0) y[row] = d + (res ? res[row] : 0.f);
  }
  prefetch_share(pf, pf_bytes);
}

// y[j] = silu(W[j] . x) * (W[I + j] . x) for j < I.
template <int K>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
    k_gemv_q8_swiglu(const int8_t* __restrict__ W, const __half* __restrict__ S,
                     const float* __restrict__ x, float* __restrict__ y, int I, const float* __restrict__ nw,
                     float eps, const char* pf, size_t pf_bytes) {
  __shared__ float4 xs4[K / 4];
  stage_x_norm<K>(x, nw, eps, xs4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * GEMV_RPW;
#pragma unroll
  for (int r = 0; r < GEMV_RPW; ++r) {
    const int j = row0 + r;
    if (j >= I) break;
    const float g = dot_q8<K>(W, S, j, xs4, lane);
    const float u = dot_q8<K>(W, S, (size_t)I + j, xs4, lane);
    if (lane == 0) y[j] = silu(g) * u;
  }
  prefetch_share(pf, pf_bytes);
}

template <int K>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
    k_gemv_bf16(const bf16* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N,
                const float* __restrict__ nw, float eps, const char* pf, size_t pf_bytes) {
  __shared__ float4 xs4[K / 4];
  stage_x_norm<K>(x, nw, eps, xs4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * GEMV_RPW;
#pragma unroll
  for (int r = 0; r < GEMV_RPW; ++r) {
    const int row = row0 + r;
    if (row >= N) break;
    const float d = dot_bf16<K>(W, row, xs4, lane);
    if (lane == 0) y[row] = d;
  }
  prefetch_share(pf, pf_bytes);
}

static int gemv_blocks(int n) { return (n + GEMV_WARPS * GEMV_RPW - 1) / (GEMV_WARPS * GEMV_RPW); }

API int dr_gemv_q8(const int8_t* W, const __half* S, const float* x, const float* res, float* y, int N, int K,
                   const float* nw, float eps, const char* pf, size_t pf_bytes, cudaStream_t st) {
  const dim3 g(gemv_blocks(N)), b(GEMV_WARPS * 32);
  switch (K) {
    case 512: k_gemv_q8<512><<<g, b, 0, st>>>(W, S, x, res, y, N, nw, eps, pf, pf_bytes); break;
    case 2048: k_gemv_q8<2048><<<g, b, 0, st>>>(W, S, x, res, y, N, nw, eps, pf, pf_bytes); break;
    case 4096: k_gemv_q8<4096><<<g, b, 0, st>>>(W, S, x, res, y, N, nw, eps, pf, pf_bytes); break;
    case 6144: k_gemv_q8<6144><<<g, b, 0, st>>>(W, S, x, res, y, N, nw, eps, pf, pf_bytes); break;
    default: return -1;
  }
  return 0;
}

API int dr_gemv_q8_swiglu(const int8_t* W, const __half* S, const float* x, float* y, int I, int K,
                          const float* nw, float eps, const char* pf, size_t pf_bytes, cudaStream_t st) {
  const dim3 g(gemv_blocks(I)), b(GEMV_WARPS * 32);
  switch (K) {
    case 2048: k_gemv_q8_swiglu<2048><<<g, b, 0, st>>>(W, S, x, y, I, nw, eps, pf, pf_bytes); break;
    case 4096: k_gemv_q8_swiglu<4096><<<g, b, 0, st>>>(W, S, x, y, I, nw, eps, pf, pf_bytes); break;
    default: return -1;
  }
  return 0;
}

API int dr_gemv_bf16(const bf16* W, const float* x, float* y, int N, int K, const float* nw, float eps,
                     const char* pf, size_t pf_bytes, cudaStream_t st) {
  const dim3 g(gemv_blocks(N)), b(GEMV_WARPS * 32);
  switch (K) {
    case 2048: k_gemv_bf16<2048><<<g, b, 0, st>>>(W, x, y, N, nw, eps, pf, pf_bytes); break;
    default: return -1;
  }
  return 0;
}

// ------------------------------------------------------------------------------------------
// MoE

// Softmax over E router logits, top-k, renormalised weights; row r of logits has stride
// `stride` and holds the shared-expert gate logit at column E.
__global__ void __launch_bounds__(256)
    k_moe_topk(const float* __restrict__ logits, int E, int topk, int stride, int* __restrict__ idx,
               float* __restrict__ wts, float* __restrict__ sgate) {
  __shared__ float red[32];
  __shared__ float rv[32];
  __shared__ int ri[32];
  const int r = blockIdx.x, e = threadIdx.x;
  const float* l = logits + (size_t)r * stride;
  const float v = e < E ? l[e] : -INFINITY;
  const float m = block_max(v, red);
  const float ex = e < E ? expf(v - m) : 0.f;
  const float z = block_sum(ex, red);
  float p = e < E ? ex / z : -1.f;
  float total = 0.f;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int k = 0; k < topk; ++k) {
    float bv = p;
    int bi = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    __syncthreads();
    if (lane == 0) rv[w] = bv, ri[w] = bi;
    __syncthreads();
    bv = lane < nw ? rv[lane] : -2.f;
    bi = lane < nw ? ri[lane] : 1 << 30;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (e == bi) p = -1.f;
    total += bv;
    if (threadIdx.x == 0) {
      idx[r * topk + k] = bi;
      wts[r * topk + k] = bv;
    }
  }
  __syncthreads();
  if (threadIdx.x < topk) wts[r * topk + threadIdx.x] /= total;
  if (threadIdx.x == 0) sgate[r] = l[E];
}

// Router logits (bf16 weights, fused input RMSNorm) for E experts + the shared gate, then
// the last block to finish does the softmax / top-k. counter must start at 0.
template <int K>
__global__ void __launch_bounds__(256)
    k_router_topk(const bf16* __restrict__ W, const float* __restrict__ x, const float* __restrict__ nw, float eps,
                  float* __restrict__ logits, int E, int topk, int* __restrict__ idx, float* __restrict__ wts,
                  float* __restrict__ sgate, unsigned* __restrict__ counter, const char* gu_q, const char* gu_sm,
                  const char* d_q, const char* d_sm, size_t gu_q_b, size_t gu_sm_b, size_t d_q_b, size_t d_sm_b) {
  __shared__ float4 xs4[K / 4];
  __shared__ float red[32];
  __shared__ float rv[32];
  __shared__ int ri[32];
  __shared__ bool last;
  stage_x_norm<K>(x, nw, eps, xs4);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * GEMV_RPW;
  for (int r = 0; r < GEMV_RPW; ++r) {
    const int row = row0 + r;
    if (row > E) break;
    const float d = dot_bf16<K>(W, row, xs4, lane);
    if (lane == 0) logits[row] = d;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int e = threadIdx.x;
  const float v = e < E ? __ldcg(logits + e) : -INFINITY;
  const float m = block_max(v, red);
  const float ex = e < E ? expf(v - m) : 0.f;
  const float zsum = block_sum(ex, red);
  float p = e < E ? ex / zsum : -1.f;
  float total = 0.f;
  const int nw_ = blockDim.x >> 5;
  for (int k = 0; k < topk; ++k) {
    float bv = p;
    int bi = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    __syncthreads();
    if (lane == 0) rv[warp] = bv, ri[warp] = bi;
    __syncthreads();
    bv = lane < nw_ ? rv[lane] : -2.f;
    bi = lane < nw_ ? ri[lane] : 1 << 30;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (e == bi) p = -1.f;
    total += bv;
    if (threadIdx.x == 0) idx[k] = bi;
    if (threadIdx.x == k) wts[k] = bv;  // normalised below
  }
  __syncthreads();
  if (threadIdx.x < topk) wts[threadIdx.x] /= total;
  if (threadIdx.x == 0) {
    sgate[0] = __ldcg(logits + E);
    *counter = 0u;
  }
  __syncthreads();
  // Warm L2 with exactly the experts the next two kernels will read.
  if (gu_q) {
    for (int k = 0; k < topk; ++k) {
      const size_t ex = __ldcg(idx + k);
      for (size_t o = threadIdx.x * 128; o < gu_q_b; o += blockDim.x * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(gu_q + ex * gu_q_b + o));
      for (size_t o = threadIdx.x * 128; o < gu_sm_b; o += blockDim.x * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(gu_sm + ex * gu_sm_b + o));
      for (size_t o = threadIdx.x * 128; o < d_q_b; o += blockDim.x * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(d_q + ex * d_q_b + o));
      for (size_t o = threadIdx.x * 128; o < d_sm_b; o += blockDim.x * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(d_sm + ex * d_sm_b + o));
    }
  }
}

API int dr_router_topk(const bf16* W, const float* x, const float* nw, float eps, float* logits, int E, int topk, int K,
                       int* idx, float* wts, float* sgate, unsigned* counter, const char* gu_q, const char* gu_sm,
                       const char* d_q, const char* d_sm, size_t gu_q_b, size_t gu_sm_b, size_t d_q_b, size_t d_sm_b,
                       cudaStream_t st) {
  if (K != 2048 || E > 256 || topk > 32) return -1;
  k_router_topk<2048><<<gemv_blocks(E + 1), GEMV_WARPS * 32, 0, st>>>(
      W, x, nw, eps, logits, E, topk, idx, wts, sgate, counter, gu_q, gu_sm, d_q, d_sm, gu_q_b, gu_sm_b, d_q_b, d_sm_b);
  return 0;
}

API int dr_moe_topk(const float* logits, int rows, int E, int topk, int stride, int* idx, float* wts,
                    float* sgate, cudaStream_t st) {
  if (E > 256) return -1;
  k_moe_topk<<<rows, 256, 0, st>>>(logits, E, topk, stride, idx, wts, sgate);
  return 0;
}

constexpr int MOE_WARPS = 4;
constexpr int MOE_RPW = 2;

// Decode: act[k][j] = silu(G_e[j] . x) * (U_e[j] . x) for the k-th routed expert e; the extra
// grid row k == topk computes the shared expert (int8 weights) into act[topk].
template <int K>
__global__ void __launch_bounds__(MOE_WARPS * 32)
    k_moe_gate_up_q4(const uint8_t* __restrict__ Q, const __half2* __restrict__ SM, const int8_t* __restrict__ shq,
                     const __half* __restrict__ shs, const float* __restrict__ x, const int* __restrict__ idx,
                     float* __restrict__ act, int I, int topk, const float* __restrict__ nw, float eps) {
  __shared__ float4 xs4[K / 4];
  __shared__ float xcs[K / 32];
  stage_x_norm<K>(x, nw, eps, xs4);
  const int k = blockIdx.y;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int j0 = (blockIdx.x * MOE_WARPS + warp) * MOE_RPW;
  if (k == topk) {
#pragma unroll
    for (int r = 0; r < MOE_RPW; ++r) {
      const int j = j0 + r;
      if (j >= I) break;
      const float g = dot_q8<K>(shq, shs, j, xs4, lane);
      const float u = dot_q8<K>(shq, shs, (size_t)I + j, xs4, lane);
      if (lane == 0) act[k * I + j] = silu(g) * u;
    }
    return;
  }
  for (int c = threadIdx.x; c < K / 32; c += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float4 v = xs4[c * 8 + j];
      s += v.x + v.y + v.z + v.w;
    }
    xcs[c] = s;
  }
  __syncthreads();
  const size_t e = idx[k];
  const uint8_t* q = Q + e * (size_t)(2 * I) * (K / 2);
  const __half2* sm = SM + e * (size_t)(2 * I) * (K / 64);
#pragma unroll
  for (int r = 0; r < MOE_RPW; ++r) {
    const int j = j0 + r;
    if (j >= I) break;
    const float g = dot_q4<K>(q, sm, j, xs4, xcs, lane);
    const float u = dot_q4<K>(q, sm, (size_t)I + j, xs4, xcs, lane);
    if (lane == 0) act[k * I + j] = silu(g) * u;
  }
}

// Decode: y[n] = res[n] + sigmoid(sgate) * shared[n] + sum_k w_k * (D_{e_k}[n] . act[k]).
template <int I, int TOPK>
__global__ void __launch_bounds__(MOE_WARPS * 32)
    k_moe_down_q4(const uint8_t* __restrict__ Q, const __half2* __restrict__ SM, const int8_t* __restrict__ shq,
                  const __half* __restrict__ shs, const float* __restrict__ act, const int* __restrict__ idx,
                  const float* __restrict__ wts, const float* __restrict__ sgate, const float* res, float* y, int N,
                  const char* pf, size_t pf_bytes) {
  __shared__ float4 as4[(TOPK + 1) * I / 4];
  __shared__ float acs[TOPK * I / 32];
  stage_x<(TOPK + 1) * I>(act, as4);
  for (int c = threadIdx.x; c < TOPK * I / 32; c += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float4 v = as4[c * 8 + j];
      s += v.x + v.y + v.z + v.w;
    }
    acs[c] = s;
  }
  __syncthreads();
  constexpr int CH = I / 32;            // chunks per expert row
  constexpr int TOT = TOPK * CH;        // chunks per output row over all experts
  constexpr int PL = (TOT + 31) / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n0 = (blockIdx.x * MOE_WARPS + warp) * MOE_RPW;
  const float sg = sigmoid(sgate[0]);
#pragma unroll
  for (int r = 0; r < MOE_RPW; ++r) {
    const int n = n0 + r;
    if (n >= N) break;
    int4 v[PL];
    float2 smv[PL];
#pragma unroll
    for (int i = 0; i < PL; ++i) {
      const int c = lane + 32 * i;
      if (c < TOT) {
        const int k = c / CH, cc = c % CH;
        const size_t rowi = (size_t)idx[k] * N + n;
        v[i] = __ldg(reinterpret_cast<const int4*>(Q + rowi * (I / 2)) + cc);
        smv[i] = __half22float2(SM[rowi * (I / 64) + (cc >> 1)]);
      }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < PL; ++i) {
      const int c = lane + 32 * i;
      if (c < TOT) {
        const int k = c / CH;
        float qx = 0.f;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const int j = (jj + lane) & 7;
          const int bits = pick4(v[i], j >> 1) >> ((j & 1) * 16);
          const float4 xv = as4[c * 8 + j];
          qx += (float)(bits & 15) * xv.x + (float)((bits >> 4) & 15) * xv.y +
                (float)((bits >> 8) & 15) * xv.z + (float)((bits >> 12) & 15) * xv.w;
        }
        acc += wts[k] * (smv[i].x * qx + smv[i].y * acs[c]);
      }
    }
    acc = warp_sum(acc);
    const float sh = dot_q8<I>(shq, shs, n, as4 + TOPK * I / 4, lane);
    if (lane == 0) y[n] = res[n] + sg * sh + acc;
  }
  prefetch_share(pf, pf_bytes);
}

// act must hold (topk + 1) * I floats; the shared expert's intermediate size must equal I.
API int dr_moe_decode(const uint8_t* gu_q, const __half2* gu_sm, const uint8_t* d_q, const __half2* d_sm,
                      const int8_t* sgu_q, const __half* sgu_s, const int8_t* sd_q, const __half* sd_s,
                      const float* x, const int* idx, const float* wts, const float* sgate, const float* res,
                      float* act, float* y, int H, int I, int topk, const float* nw, float eps, const char* pf,
                      size_t pf_bytes, cudaStream_t st) {
  if (H != 2048 || I != 512 || topk != 8) return -1;
  k_moe_gate_up_q4<2048><<<dim3((I + MOE_WARPS * MOE_RPW - 1) / (MOE_WARPS * MOE_RPW), topk + 1),
                           MOE_WARPS * 32, 0, st>>>(gu_q, gu_sm, sgu_q, sgu_s, x, idx, act, I, topk, nw, eps);
  k_moe_down_q4<512, 8><<<(H + MOE_WARPS * MOE_RPW - 1) / (MOE_WARPS * MOE_RPW), MOE_WARPS * 32, 0, st>>>(
      d_q, d_sm, sd_q, sd_s, act, idx, wts, sgate, res, y, H, pf, pf_bytes);
  return 0;
}

// Prefill routing: count rows per expert.
__global__ void k_moe_count(const int* __restrict__ idx, int n, int* __restrict__ counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(counts + idx[i], 1);
}

API int dr_moe_count(const int* idx, int n, int* counts, cudaStream_t st) {
  k_moe_count<<<(n + 255) / 256, 256, 0, st>>>(idx, n, counts);
  return 0;
}

// Prefill routing: scatter rows of x (bf16) into expert-sorted order.
// offsets[e] is the first slot of expert e; fill[e] starts at zero.
__global__ void k_moe_permute(const int* __restrict__ idx, int rows, int topk, const int* __restrict__ offsets,
                              int* __restrict__ fill, int* __restrict__ slot_of, const bf16* __restrict__ x,
                              bf16* __restrict__ xp, int H) {
  const int i = blockIdx.x;  // (row, k) pair
  __shared__ int slot;
  if (threadIdx.x == 0) {
    const int e = idx[i];
    slot = offsets[e] + atomicAdd(fill + e, 1);
    slot_of[i] = slot;
  }
  __syncthreads();
  const int r = i / topk;
  const int4* src = reinterpret_cast<const int4*>(x + (size_t)r * H);
  int4* dst = reinterpret_cast<int4*>(xp + (size_t)slot * H);
  for (int c = threadIdx.x; c < H / 8; c += blockDim.x) dst[c] = src[c];
}

API int dr_moe_permute(const int* idx, int rows, int topk, const int* offsets, int* fill, int* slot_of,
                       const bf16* x, bf16* xp, int H, cudaStream_t st) {
  k_moe_permute<<<rows * topk, 128, 0, st>>>(idx, rows, topk, offsets, fill, slot_of, x, xp, H);
  return 0;
}

// Prefill: x[r] += sigmoid(sgate[r]) * shared[r] + sum_k w[r,k] * Y[slot_of[r,k]].
__global__ void k_moe_combine(float* __restrict__ x, const float* __restrict__ Y, const int* __restrict__ slot_of,
                              const float* __restrict__ wts, const float* __restrict__ sgate,
                              const float* __restrict__ shared, int topk, int H) {
  const int r = blockIdx.x;
  const float sg = sigmoid(sgate[r]);
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    float acc = sg * shared[(size_t)r * H + c];
    for (int k = 0; k < topk; ++k) acc += wts[r * topk + k] * Y[(size_t)slot_of[r * topk + k] * H + c];
    x[(size_t)r * H + c] += acc;
  }
}

API int dr_moe_combine(float* x, const float* Y, const int* slot_of, const float* wts, const float* sgate,
                       const float* shared, int rows, int topk, int H, cudaStream_t st) {
  k_moe_combine<<<rows, 256, 0, st>>>(x, Y, slot_of, wts, sgate, shared, topk, H);
  return 0;
}

// Prefill, grouped by expert: rows of xp are sorted by expert (offsets/counts per expert).
// Each block owns a slice of one expert's weight rows, reads them once per tile of TT
// tokens and applies them to every token routed to that expert.

constexpr int PF_TT = 8;

// Bank and index of expert e for a prefill pass. Without `where`, expert e of the bank (Q, SM). With it (experts in
// RAM): where[e] >= 0 is a slot of the VRAM cache (Q, SM), <= -2 is index -where[e] - 2 of the stage (Q2, SM2), and
// -1 is an expert this pass skips. Returns false to skip.
DEV bool pf_expert(int e, const int* where, const uint8_t*& Q, const __half2*& SM, const uint8_t* Q2,
                   const __half2* SM2, size_t& at) {
  at = e;
  if (!where) return true;
  const int w = where[e];
  if (w == -1) return false;
  if (w < -1) {
    at = (size_t)(-w - 2);
    Q = Q2;
    SM = SM2;
  } else {
    at = (size_t)w;
  }
  return true;
}

// act[slot, j] = silu(G_e[j] . x_slot) * (U_e[j] . x_slot), bf16. K = 2048, I = 512.
__global__ void __launch_bounds__(256)
    k_moe_pf_gate_up(const uint8_t* Q, const __half2* SM, const uint8_t* Q2, const __half2* SM2,
                     const int* __restrict__ where, const bf16* __restrict__ xp, const int* __restrict__ offsets,
                     const int* __restrict__ counts, bf16* __restrict__ act, int I) {
  constexpr int K = 2048, CH = K / 32, PAIRS_PER_WARP = 4;
  __shared__ __align__(16) bf16 xs[PF_TT * K];
  __shared__ float xcs[PF_TT * CH];
  const int e = blockIdx.x;
  const int cnt = counts[e];
  if (cnt == 0) return;
  size_t at;
  if (!pf_expert(e, where, Q, SM, Q2, SM2, at)) return;
  const int off = offsets[e];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint8_t* q = Q + at * 2 * I * (K / 2);
  const __half2* sm = SM + at * 2 * I * (K / 64);
  const int j0 = (blockIdx.y * 8 + warp) * PAIRS_PER_WARP;
  for (int t0 = 0; t0 < cnt; t0 += PF_TT) {
    const int nt = min(PF_TT, cnt - t0);
    __syncthreads();
    for (int i = threadIdx.x; i < PF_TT * K / 8; i += blockDim.x) {
      const int tt = i / (K / 8);
      int4 v = make_int4(0, 0, 0, 0);
      if (tt < nt) v = reinterpret_cast<const int4*>(xp + (size_t)(off + t0 + tt) * K)[i % (K / 8)];
      reinterpret_cast<int4*>(xs)[i] = v;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < PF_TT * CH; i += blockDim.x) {
      const bf16* c = xs + (i / CH) * K + (i % CH) * 32;
      float s = 0.f;
#pragma unroll
      for (int v = 0; v < 32; ++v) s += __bfloat162float(c[v]);
      xcs[i] = s;
    }
    __syncthreads();
    for (int pj = 0; pj < PAIRS_PER_WARP; ++pj) {
      const int j = j0 + pj;
      if (j >= I) break;
      float ag[PF_TT], au[PF_TT];
#pragma unroll
      for (int tt = 0; tt < PF_TT; ++tt) ag[tt] = au[tt] = 0.f;
#pragma unroll
      for (int ci = 0; ci < CH / 32; ++ci) {
        const int c = lane + 32 * ci;
#pragma unroll
        for (int which = 0; which < 2; ++which) {
          const size_t row = which ? (size_t)I + j : (size_t)j;
          const int4 w = __ldg(reinterpret_cast<const int4*>(q + row * (K / 2)) + c);
          const float2 smv = __half22float2(sm[row * (K / 64) + (c >> 1)]);
          float wq[32];
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int word = pick4(w, b);
#pragma unroll
            for (int nib = 0; nib < 8; ++nib) wq[b * 8 + nib] = (float)((word >> (4 * nib)) & 15);
          }
#pragma unroll
          for (int tt = 0; tt < PF_TT; ++tt) {
            if (tt < nt) {
              const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xs + tt * K + c * 32);
              float qx = 0.f;
#pragma unroll
              for (int v = 0; v < 16; ++v) {
                const float2 xv = __bfloat1622float2(x2[v]);
                qx += wq[2 * v] * xv.x + wq[2 * v + 1] * xv.y;
              }
              const float d = smv.x * qx + smv.y * xcs[tt * CH + c];
              if (which) au[tt] += d; else ag[tt] += d;
            }
          }
        }
      }
#pragma unroll
      for (int tt = 0; tt < PF_TT; ++tt) {
        if (tt < nt) {
          const float g = warp_sum(ag[tt]), u = warp_sum(au[tt]);
          if (lane == 0) act[(size_t)(off + t0 + tt) * I + j] = __float2bfloat16(silu(g) * u);
        }
      }
    }
  }
}

// y[slot, n] = D_e[n] . act_slot (f32). K = I = 512, N = 2048. Half-warps own rows.
__global__ void __launch_bounds__(256)
    k_moe_pf_down(const uint8_t* Q, const __half2* SM, const uint8_t* Q2, const __half2* SM2,
                  const int* __restrict__ where, const bf16* __restrict__ act, const int* __restrict__ offsets,
                  const int* __restrict__ counts, float* __restrict__ y, int N) {
  constexpr int K = 512, CH = K / 32, ROWS_PER_HALF = 4;
  __shared__ __align__(16) bf16 xs[PF_TT * K];
  __shared__ float xcs[PF_TT * CH];
  const int e = blockIdx.x;
  const int cnt = counts[e];
  if (cnt == 0) return;
  size_t at;
  if (!pf_expert(e, where, Q, SM, Q2, SM2, at)) return;
  const int off = offsets[e];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int half = lane >> 4, c = lane & 15;
  const uint8_t* q = Q + at * N * (K / 2);
  const __half2* sm = SM + at * N * (K / 64);
  const int n0 = ((blockIdx.y * 8 + warp) * 2 + half) * ROWS_PER_HALF;
  for (int t0 = 0; t0 < cnt; t0 += PF_TT) {
    const int nt = min(PF_TT, cnt - t0);
    __syncthreads();
    for (int i = threadIdx.x; i < PF_TT * K / 8; i += blockDim.x) {
      const int tt = i / (K / 8);
      int4 v = make_int4(0, 0, 0, 0);
      if (tt < nt) v = reinterpret_cast<const int4*>(act + (size_t)(off + t0 + tt) * K)[i % (K / 8)];
      reinterpret_cast<int4*>(xs)[i] = v;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < PF_TT * CH; i += blockDim.x) {
      const bf16* cc = xs + (i / CH) * K + (i % CH) * 32;
      float s = 0.f;
#pragma unroll
      for (int v = 0; v < 32; ++v) s += __bfloat162float(cc[v]);
      xcs[i] = s;
    }
    __syncthreads();
    for (int r = 0; r < ROWS_PER_HALF; ++r) {
      const int n = n0 + r;
      const bool valid = n < N;
      float acc[PF_TT];
#pragma unroll
      for (int tt = 0; tt < PF_TT; ++tt) acc[tt] = 0.f;
      if (valid) {
        const int4 w = __ldg(reinterpret_cast<const int4*>(q + (size_t)n * (K / 2)) + c);
        const float2 smv = __half22float2(sm[(size_t)n * (K / 64) + (c >> 1)]);
        float wq[32];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int word = pick4(w, b);
#pragma unroll
          for (int nib = 0; nib < 8; ++nib) wq[b * 8 + nib] = (float)((word >> (4 * nib)) & 15);
        }
#pragma unroll
        for (int tt = 0; tt < PF_TT; ++tt) {
          if (tt < nt) {
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xs + tt * K + c * 32);
            float qx = 0.f;
#pragma unroll
            for (int v = 0; v < 16; ++v) {
              const float2 xv = __bfloat1622float2(x2[v]);
              qx += wq[2 * v] * xv.x + wq[2 * v + 1] * xv.y;
            }
            acc[tt] = smv.x * qx + smv.y * xcs[tt * CH + c];
          }
        }
      }
#pragma unroll
      for (int tt = 0; tt < PF_TT; ++tt) {
        float v = acc[tt];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (valid && c == 0 && tt < nt) y[(size_t)(off + t0 + tt) * N + n] = v;
      }
    }
  }
}

API int dr_moe_prefill(const uint8_t* gu_q, const __half2* gu_sm, const uint8_t* d_q, const __half2* d_sm,
                       const bf16* xp, const int* offsets, const int* counts, bf16* act, float* y, int E, int H, int I,
                       cudaStream_t st) {
  if (H != 2048 || I != 512) return -1;
  k_moe_pf_gate_up<<<dim3(E, (I + 31) / 32), 256, 0, st>>>(gu_q, gu_sm, nullptr, nullptr, nullptr, xp, offsets, counts,
                                                           act, I);
  k_moe_pf_down<<<dim3(E, (H + 63) / 64), 256, 0, st>>>(d_q, d_sm, nullptr, nullptr, nullptr, act, offsets, counts, y, H);
  return 0;
}

// As dr_moe_prefill for experts in RAM: `where` (see pf_expert) picks each expert from the cache banks (c_*) or the
// stage banks (s_*), or skips it.
API int dr_moe_prefill_where(const uint8_t* c_gq, const __half2* c_gsm, const uint8_t* c_dq, const __half2* c_dsm,
                             const uint8_t* s_gq, const __half2* s_gsm, const uint8_t* s_dq, const __half2* s_dsm,
                             const int* where, const bf16* xp, const int* offsets, const int* counts, bf16* act,
                             float* y, int E, int H, int I, cudaStream_t st) {
  if (H != 2048 || I != 512) return -1;
  k_moe_pf_gate_up<<<dim3(E, (I + 31) / 32), 256, 0, st>>>(c_gq, c_gsm, s_gq, s_gsm, where, xp, offsets, counts, act, I);
  k_moe_pf_down<<<dim3(E, (H + 63) / 64), 256, 0, st>>>(c_dq, c_dsm, s_dq, s_dsm, where, act, offsets, counts, y, H);
  return 0;
}

// ------------------------------------------------------------------------------------------
// Routed experts kept in mapped host memory, with a cache of S expert slots in VRAM shared by all layers.
// The decode kernels read a slot bank exactly like an expert bank, with slot ids in place of expert ids.
//   map[layer * E + e]  slot holding expert e of the layer, or -1
//   owner[s]            layer * E + e held by slot s, or -1
//   ref[s]              CLOCK reference bit;  tag[s] the call that last used slot s (never evicted by that call)
//   state               [0] clock hand, [1] call counter, [2..3] hits (u64), [4..5] misses (u64)

__global__ void __launch_bounds__(32)
    k_expert_slots(const int* __restrict__ idx, int topk, int key0, int S, int* __restrict__ map,
                   int* __restrict__ owner, unsigned* __restrict__ ref, unsigned* __restrict__ tag,
                   unsigned* __restrict__ state, int* __restrict__ sidx, int* __restrict__ fill) {
  const int lane = threadIdx.x;
  __shared__ unsigned tick;
  if (lane == 0) tick = ++state[1];
  __syncwarp();
  bool miss = false;
  if (lane < topk) {
    const int s = map[key0 + idx[lane]];
    if (s >= 0) {
      ref[s] = 1u;
      tag[s] = tick;
      sidx[lane] = s;
      fill[lane] = -1;
    } else {
      miss = true;
    }
  }
  unsigned need = __ballot_sync(0xffffffffu, miss);
  if (lane == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(state + 2), (unsigned long long)(topk - __popc(need)));
    atomicAdd(reinterpret_cast<unsigned long long*>(state + 4), (unsigned long long)__popc(need));
  }
  __syncwarp();
  unsigned hand = state[0];
  while (need) {
    const int k = __ffs(need) - 1;
    need &= need - 1;
    int victim = -1;
    while (victim < 0) {
      const int s = (int)((hand + lane) % S);
      const bool used = tag[s] == tick;
      const unsigned m = __ballot_sync(0xffffffffu, !used && ref[s] == 0u);
      const int first = m ? __ffs(m) - 1 : 32;
      if (lane < first && !used) ref[s] = 0u;
      __syncwarp();
      if (m) {
        victim = (int)((hand + first) % S);
        hand = (unsigned)((victim + 1) % S);
      } else {
        hand = (hand + 32) % S;
      }
    }
    if (lane == 0) {
      const int key = key0 + idx[k];
      const int old = owner[victim];
      if (old >= 0) map[old] = -1;
      owner[victim] = key;
      map[key] = victim;
      ref[victim] = 1u;
      tag[victim] = tick;
      sidx[k] = victim;
      fill[k] = victim;
    }
    __syncwarp();
  }
  if (lane == 0) state[0] = hand;
}

API int dr_expert_slots(const int* idx, int topk, int key0, int S, int* map, int* owner, unsigned* ref, unsigned* tag,
                        unsigned* state, int* sidx, int* fill, cudaStream_t st) {
  if (topk > 32 || S < topk) return -1;
  k_expert_slots<<<1, 32, 0, st>>>(idx, topk, key0, S, map, owner, ref, tag, state, sidx, fill);
  return 0;
}

// One expert part: `n` int4 per expert, read from the source bank at expert `e`, written to the slot bank at `slot`.
DEV void copy_part(const int4* __restrict__ src, int4* __restrict__ dst, size_t n, size_t e, size_t slot) {
  constexpr int U = 4;
  src += e * n;
  dst += slot * n;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * stride < n; i += U * stride) {
    int4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = src[i + u * stride];
#pragma unroll
    for (int u = 0; u < U; ++u) dst[i + u * stride] = v[u];
  }
  for (; i < n; i += stride) dst[i] = src[i];
}

// The experts k with fill[k] >= 0: expert idx[k] of the source banks into slot fill[k]. Grid (blocks, topk).
__global__ void __launch_bounds__(256)
    k_expert_fill(const int* __restrict__ idx, const int* __restrict__ fill, const int4* g_q, const int4* g_sm,
                  const int4* d_q, const int4* d_sm, int4* c_gq, int4* c_gsm, int4* c_dq, int4* c_dsm, size_t n_gq,
                  size_t n_gsm, size_t n_dq, size_t n_dsm) {
  const int slot = fill[blockIdx.y];
  if (slot < 0) return;
  const size_t e = idx[blockIdx.y];
  copy_part(g_q, c_gq, n_gq, e, slot);
  copy_part(g_sm, c_gsm, n_gsm, e, slot);
  copy_part(d_q, c_dq, n_dq, e, slot);
  copy_part(d_sm, c_dsm, n_dsm, e, slot);
}

// Byte sizes are per expert and multiples of 16.
API int dr_expert_fill(const int* idx, const int* fill, int topk, const void* g_q, const void* g_sm, const void* d_q,
                       const void* d_sm, void* c_gq, void* c_gsm, void* c_dq, void* c_dsm, size_t b_gq, size_t b_gsm,
                       size_t b_dq, size_t b_dsm, cudaStream_t st) {
  if ((b_gq | b_gsm | b_dq | b_dsm) & 15) return -1;
  k_expert_fill<<<dim3(32, topk), 256, 0, st>>>(
      idx, fill, (const int4*)g_q, (const int4*)g_sm, (const int4*)d_q, (const int4*)d_sm, (int4*)c_gq, (int4*)c_gsm,
      (int4*)c_dq, (int4*)c_dsm, b_gq / 16, b_gsm / 16, b_dq / 16, b_dsm / 16);
  return 0;
}

// ------------------------------------------------------------------------------------------
// dequantisation for prefill GEMMs

__global__ void k_dequant_q8(const int8_t* __restrict__ W, const __half* __restrict__ S, bf16* __restrict__ out,
                             size_t n_elems, int K) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= n_elems) return;
  const size_t row = i / K, col = i % K;
  const float s = __half2float(S[row * (K / 128) + col / 128]);
  const int2 v = *reinterpret_cast<const int2*>(W + i);
  __align__(16) bf16 o[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    o[j] = __float2bfloat16((float)(int8_t)(v.x >> (8 * j)) * s);
    o[4 + j] = __float2bfloat16((float)(int8_t)(v.y >> (8 * j)) * s);
  }
  *reinterpret_cast<int4*>(out + i) = *reinterpret_cast<int4*>(o);
}

API int dr_dequant_q8(const int8_t* W, const __half* S, bf16* out, int N, int K, cudaStream_t st) {
  const size_t n = (size_t)N * K;
  k_dequant_q8<<<(unsigned)((n / 8 + 255) / 256), 256, 0, st>>>(W, S, out, n, K);
  return 0;
}

__global__ void k_dequant_q4(const uint8_t* __restrict__ Q, const __half2* __restrict__ SM, bf16* __restrict__ out,
                             size_t n_elems, int K) {
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= n_elems) return;
  const size_t row = i / K, col = i % K;
  const float2 sm = __half22float2(SM[row * (K / 64) + col / 64]);
  const int v = *reinterpret_cast<const int*>(Q + i / 2);
  __align__(16) bf16 o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16((float)((v >> (4 * j)) & 15) * sm.x + sm.y);
  *reinterpret_cast<int4*>(out + i) = *reinterpret_cast<int4*>(o);
}

API int dr_dequant_q4(const uint8_t* Q, const __half2* SM, bf16* out, int N, int K, cudaStream_t st) {
  const size_t n = (size_t)N * K;
  k_dequant_q4<<<(unsigned)((n / 8 + 255) / 256), 256, 0, st>>>(Q, SM, out, n, K);
  return 0;
}

// ------------------------------------------------------------------------------------------
// elementwise

__global__ void k_f32_to_bf16(const float* __restrict__ x, bf16* __restrict__ y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(x[i]);
}

API int dr_f32_to_bf16(const float* x, bf16* y, size_t n, cudaStream_t st) {
  k_f32_to_bf16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, y, n);
  return 0;
}

// y[r, j] = silu(h[r, j]) * h[r, I + j] in bf16 (row stride of h is 2I).
__global__ void k_swiglu_rows(const float* __restrict__ h, bf16* __restrict__ y, int rows, int I) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * I) return;
  const size_t r = i / I, j = i % I;
  y[i] = __float2bfloat16(silu(h[r * 2 * I + j]) * h[r * 2 * I + I + j]);
}

API int dr_swiglu_rows(const float* h, bf16* y, int rows, int I, cudaStream_t st) {
  const size_t n = (size_t)rows * I;
  k_swiglu_rows<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(h, y, rows, I);
  return 0;
}

// Row-wise RMSNorm with the weight's "+1" already folded in. Writes f32 and/or bf16.
__global__ void __launch_bounds__(256)
    k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* y, bf16* yb, int D, float eps) {
  __shared__ float red[32];
  const float* xr = x + (size_t)blockIdx.x * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xr[i] * xr[i];
  ss = block_sum(ss, red);
  const float inv = rsqrtf(ss / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = xr[i] * inv * w[i];
    if (y) y[(size_t)blockIdx.x * D + i] = v;
    if (yb) yb[(size_t)blockIdx.x * D + i] = __float2bfloat16(v);
  }
}

API int dr_rmsnorm(const float* x, const float* w, float* y, bf16* yb, int rows, int D, float eps,
                   cudaStream_t st) {
  k_rmsnorm<<<rows, 256, 0, st>>>(x, w, y, yb, D, eps);
  return 0;
}

// ------------------------------------------------------------------------------------------
// embeddings

DEV bool is_sem(int tok, int sem0, int sem1) { return tok >= sem0 && tok <= sem1; }
DEV int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

// Backbone input rows. rows_emb [n, D] holds each row's text embedding (prefill); when it is
// null the embedding comes from cand_table ([eos, semantic...] rows, decode).
// Semantic rows add codes 1..nb from cb_emb and scale by 1/sqrt(1 + nb).
__global__ void __launch_bounds__(256)
    k_embed(const bf16* __restrict__ rows_emb, const bf16* __restrict__ cand_table, const int* __restrict__ frames,
            int fstride, const bf16* __restrict__ cb_emb, float* __restrict__ x, int D, int sem0, int sem1, int eos,
            int cb, int nb) {
  const int r = blockIdx.x;
  const int* f = frames + (size_t)r * fstride;
  const int main_tok = f[0];
  const bf16* e;
  if (rows_emb) {
    e = rows_emb + (size_t)r * D;
  } else {
    const int ci = main_tok == eos ? 0 : clampi(main_tok - sem0, 0, cb - 1) + 1;
    e = cand_table + (size_t)ci * D;
  }
  const bool sem = is_sem(main_tok, sem0, sem1);
  const float scale = rsqrtf(1.f + nb);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v = __bfloat162float(e[i]);
    if (sem) {
      for (int c = 1; c <= nb; ++c) {
        const int code = clampi(f[1 + c], 0, cb - 1);
        v += __bfloat162float(cb_emb[((size_t)c * cb + code) * D + i]);
      }
      v *= scale;
    }
    x[(size_t)r * D + i] = v;
  }
}

API int dr_embed(const bf16* rows_emb, const bf16* cand_table, const int* frames, int fstride, const bf16* cb_emb,
                 float* x, int rows, int D, int sem0, int sem1, int eos, int cb, int nb, cudaStream_t st) {
  k_embed<<<rows, 256, 0, st>>>(rows_emb, cand_table, frames, fstride, cb_emb, x, D, sem0, sem1, eos, cb, nb);
  return 0;
}

// Tower layer input: [enorm(e) | hnorm(h)] with e = label-is-semantic ? tok_emb[code] : 0,
// plus (first shared layer) the summed codebook embeddings of the input frame's codes nb+1...
// Labels come from frames (row stride fstride): main token at column 0, code at code_col.
__global__ void __launch_bounds__(256)
    k_tower_in(const float* __restrict__ h, const int* __restrict__ frames, int fstride, int code_col,
               const int* __restrict__ in_frames, int in_stride, const bf16* __restrict__ tok_emb,
               const bf16* __restrict__ cb_emb, const float* __restrict__ enorm, const float* __restrict__ hnorm,
               const float* __restrict__ pre_norm, float* y, bf16* yb, int sem0, int sem1, int cb, int ncb, int nb,
               float eps) {
  constexpr int D = 2048;
  constexpr int PER = D / 256;
  __shared__ float red[32];
  const int r = blockIdx.x;
  const int* f = frames + (size_t)r * fstride;
  const bool lab = is_sem(f[0], sem0, sem1);
  const int code = clampi(f[code_col], 0, cb - 1);
  bool shared = false;
  const int* fi = nullptr;
  if (in_frames) {
    fi = in_frames + (size_t)r * in_stride;
    shared = is_sem(fi[0], sem0, sem1);
  }
  float e[PER], hv[PER];
  float se = 0.f, sh = 0.f;
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = threadIdx.x + k * 256;
    float v = lab ? __bfloat162float(tok_emb[(size_t)code * D + i]) : 0.f;
    if (shared) {
      for (int c = 1 + nb; c < ncb; ++c) {
        const int cc = clampi(fi[1 + c], 0, cb - 1);
        v += __bfloat162float(cb_emb[((size_t)c * cb + cc) * D + i]);
      }
    }
    e[k] = v;
    hv[k] = h[(size_t)r * D + i];
    se += v * v;
    sh += hv[k] * hv[k];
  }
  if (pre_norm) {
    // h arrives un-normalised: apply the previous layer's final RMSNorm first.
    const float ip = rsqrtf(block_sum(sh, red) / D + eps);
    sh = 0.f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      hv[k] = hv[k] * ip * pre_norm[threadIdx.x + k * 256];
      sh += hv[k] * hv[k];
    }
  }
  se = block_sum(se, red);
  sh = block_sum(sh, red);
  const float ie = rsqrtf(se / D + eps), ih = rsqrtf(sh / D + eps);
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int i = threadIdx.x + k * 256;
    const float a = e[k] * ie * enorm[i];
    const float b = hv[k] * ih * hnorm[i];
    if (y) {
      y[(size_t)r * 2 * D + i] = a;
      y[(size_t)r * 2 * D + D + i] = b;
    }
    if (yb) {
      yb[(size_t)r * 2 * D + i] = __float2bfloat16(a);
      yb[(size_t)r * 2 * D + D + i] = __float2bfloat16(b);
    }
  }
}

API int dr_tower_in(const float* h, const int* frames, int fstride, int code_col, const int* in_frames, int in_stride,
                    const bf16* tok_emb, const bf16* cb_emb, const float* enorm, const float* hnorm,
                    const float* pre_norm, float* y, bf16* yb, int rows, int D, int sem0, int sem1, int cb, int ncb,
                    int nb, float eps, cudaStream_t st) {
  if (D != 2048) return -1;
  k_tower_in<<<rows, 256, 0, st>>>(h, frames, fstride, code_col, in_frames, in_stride, tok_emb, cb_emb, enorm, hnorm,
                                   pre_norm, y, yb, sem0, sem1, cb, ncb, nb, eps);
  return 0;
}

// ------------------------------------------------------------------------------------------
// full attention

// Per (row, head): RMSNorm over head_dim, rotary on the first `rot` dims, then store.
// qkv rows are [query (H*hd) | gate (H*hd) | key (Hkv*hd) | value (Hkv*hd)].
// Query heads go to q_out [rows, H*hd]; key/value heads go to the bf16 caches at position
// pos_base[0] + pos_off + row.
__global__ void k_qk_norm_rope(const float* __restrict__ qkv, int rs, int H, int Hkv, const float* __restrict__ qn,
                               const float* __restrict__ kn, const float* __restrict__ inv_freq, int rot,
                               int interleaved, const int* __restrict__ pos_base, int pos_off, float* __restrict__ q_out,
                               bf16* __restrict__ kc, bf16* __restrict__ vc, float eps) {
  extern __shared__ float buf[];
  __shared__ float red[32];
  const int hd = blockDim.x;
  const int r = blockIdx.x, hh = blockIdx.y, t = threadIdx.x;
  const int pos = pos_base[0] + pos_off + r;
  const float* row = qkv + (size_t)r * rs;
  const bool is_q = hh < H;
  const int kh = hh - H;
  const float* src = is_q ? row + hh * hd : row + 2 * H * hd + kh * hd;
  float v = src[t];
  const float ss = block_sum(v * v, red);
  v = v * rsqrtf(ss / hd + eps) * (is_q ? qn[t] : kn[t]);
  buf[t] = v;
  __syncthreads();
  float o = v;
  if (t < rot) {
    if (interleaved) {
      const int i = t >> 1;
      float s, c;
      sincosf((float)pos * inv_freq[i], &s, &c);
      o = (t & 1) ? buf[t] * c + buf[t - 1] * s : buf[t] * c - buf[t + 1] * s;
    } else {
      const int half = rot >> 1;
      const int i = t < half ? t : t - half;
      float s, c;
      sincosf((float)pos * inv_freq[i], &s, &c);
      o = t < half ? buf[t] * c - buf[t + half] * s : buf[t] * c + buf[t - half] * s;
    }
  }
  if (is_q) {
    q_out[(size_t)r * H * hd + hh * hd + t] = o;
  } else {
    const size_t at = ((size_t)pos * Hkv + kh) * hd + t;
    kc[at] = __float2bfloat16(o);
    vc[at] = __float2bfloat16(row[2 * H * hd + Hkv * hd + kh * hd + t]);
  }
}

API int dr_qk_norm_rope(const float* qkv, int rs, int rows, int H, int Hkv, int hd, const float* qn, const float* kn,
                        const float* inv_freq, int rot, int interleaved, const int* pos_base, int pos_off,
                        float* q_out, bf16* kc, bf16* vc, float eps, cudaStream_t st) {
  k_qk_norm_rope<<<dim3(rows, H + Hkv), hd, hd * sizeof(float), st>>>(qkv, rs, H, Hkv, qn, kn, inv_freq, rot,
                                                                      interleaved, pos_base, pos_off, q_out, kc, vc,
                                                                      eps);
  return 0;
}

// One warp's online-softmax pass over keys [k0, k1) with stride `step`, head_dim 256:
// each lane owns 8 consecutive dims.
struct AttnAcc {
  float m, l, acc[8];
};

DEV void attn_warp(const float* q8, const bf16* __restrict__ kc, const bf16* __restrict__ vc, int Hkv, int kh,
                   int k0, int k1, int step, int lane, AttnAcc& a) {
  a.m = -INFINITY;
  a.l = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) a.acc[i] = 0.f;
  for (int j = k0; j < k1; j += step) {
    const size_t at = ((size_t)j * Hkv + kh) * 256 + lane * 8;
    const int4 kv = *reinterpret_cast<const int4*>(kc + at);
    float s = q8[0] * bf_lo(kv.x) + q8[1] * bf_hi(kv.x) + q8[2] * bf_lo(kv.y) + q8[3] * bf_hi(kv.y) +
              q8[4] * bf_lo(kv.z) + q8[5] * bf_hi(kv.z) + q8[6] * bf_lo(kv.w) + q8[7] * bf_hi(kv.w);
    s = warp_sum(s);
    const int4 vv = *reinterpret_cast<const int4*>(vc + at);
    const float mn = fmaxf(a.m, s);
    const float corr = expf(a.m - mn);
    const float p = expf(s - mn);
    a.l = a.l * corr + p;
    const float v8[8] = {bf_lo(vv.x), bf_hi(vv.x), bf_lo(vv.y), bf_hi(vv.y),
                         bf_lo(vv.z), bf_hi(vv.z), bf_lo(vv.w), bf_hi(vv.w)};
#pragma unroll
    for (int i = 0; i < 8; ++i) a.acc[i] = a.acc[i] * corr + p * v8[i];
    a.m = mn;
  }
}

constexpr int ATTN_WARPS = 4;

// Merge the per-warp partials of a block into (m, l, acc[256]) in shared memory.
DEV void attn_block_merge(const AttnAcc& a, float* sm_m, float* sm_l, float* sm_acc, float& M, float& Lsum,
                          float* out_acc) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) sm_m[w] = a.m, sm_l[w] = a.l;
#pragma unroll
  for (int i = 0; i < 8; ++i) sm_acc[w * 256 + lane * 8 + i] = a.acc[i];
  __syncthreads();
  M = -INFINITY;
  for (int i = 0; i < ATTN_WARPS; ++i) M = fmaxf(M, sm_m[i]);
  Lsum = 0.f;
  for (int i = 0; i < ATTN_WARPS; ++i) Lsum += sm_m[i] == -INFINITY ? 0.f : sm_l[i] * expf(sm_m[i] - M);
  for (int d = threadIdx.x; d < 256; d += blockDim.x) {
    float o = 0.f;
    for (int i = 0; i < ATTN_WARPS; ++i)
      o += sm_m[i] == -INFINITY ? 0.f : sm_acc[i * 256 + d] * expf(sm_m[i] - M);
    out_acc[d] = o;
  }
}

// Prefill: one block per (row, head); row r sees keys 0..pos_base+pos_off+r.
// Output is multiplied by sigmoid(gate) and written to o [rows, H*256].
__global__ void __launch_bounds__(ATTN_WARPS * 32)
    k_attn_prefill(const float* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                   const float* __restrict__ gate, int gate_stride, int H, int Hkv, const int* __restrict__ pos_base,
                   int pos_off, float* __restrict__ o) {
  __shared__ float sm_m[ATTN_WARPS], sm_l[ATTN_WARPS], sm_acc[ATTN_WARPS * 256], out_acc[256];
  const int r = blockIdx.x, h = blockIdx.y;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int kh = h / (H / Hkv);
  const int nkeys = pos_base[0] + pos_off + r + 1;
  float q8[8];
  const float scale = rsqrtf(256.f);
#pragma unroll
  for (int i = 0; i < 8; ++i) q8[i] = q[(size_t)r * H * 256 + h * 256 + lane * 8 + i] * scale;
  AttnAcc a;
  attn_warp(q8, kc, vc, Hkv, kh, w, nkeys, ATTN_WARPS, lane, a);
  float M, Lsum;
  attn_block_merge(a, sm_m, sm_l, sm_acc, M, Lsum, out_acc);
  __syncthreads();
  for (int d = threadIdx.x; d < 256; d += blockDim.x) {
    const float g = gate[(size_t)r * gate_stride + h * 256 + d];
    o[(size_t)r * H * 256 + h * 256 + d] = out_acc[d] / Lsum * sigmoid(g);
  }
}

API int dr_attn_prefill(const float* q, const bf16* kc, const bf16* vc, const float* gate, int gate_stride, int rows,
                        int H, int Hkv, int hd, const int* pos_base, int pos_off, float* o, cudaStream_t st) {
  if (hd != 256) return -1;
  k_attn_prefill<<<dim3(rows, H), ATTN_WARPS * 32, 0, st>>>(q, kc, vc, gate, gate_stride, H, Hkv, pos_base, pos_off,
                                                             o);
  return 0;
}

constexpr int ATTN_SPLITS = 16;

// Decode, split over keys: block (head, split) writes its partial (m, l, acc).
__global__ void __launch_bounds__(ATTN_WARPS * 32)
    k_attn_decode_part(const float* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc, int H,
                       int Hkv, const int* __restrict__ pos_base, int pos_off, float* __restrict__ part) {
  __shared__ float sm_m[ATTN_WARPS], sm_l[ATTN_WARPS], sm_acc[ATTN_WARPS * 256], out_acc[256];
  const int h = blockIdx.x, s = blockIdx.y;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int kh = h / (H / Hkv);
  const int nkeys = pos_base[0] + pos_off + 1;
  const int chunk = (nkeys + ATTN_SPLITS - 1) / ATTN_SPLITS;
  const int k0 = s * chunk, k1 = min(nkeys, k0 + chunk);
  float q8[8];
  const float scale = rsqrtf(256.f);
#pragma unroll
  for (int i = 0; i < 8; ++i) q8[i] = q[h * 256 + lane * 8 + i] * scale;
  AttnAcc a;
  attn_warp(q8, kc, vc, Hkv, kh, k0 + w, k1, ATTN_WARPS, lane, a);
  float M, Lsum;
  attn_block_merge(a, sm_m, sm_l, sm_acc, M, Lsum, out_acc);
  __syncthreads();
  float* p = part + ((size_t)h * ATTN_SPLITS + s) * 258;
  if (threadIdx.x == 0) p[0] = M, p[1] = Lsum;
  for (int d = threadIdx.x; d < 256; d += blockDim.x) p[2 + d] = out_acc[d];
}

__global__ void k_attn_decode_combine(const float* __restrict__ part, const float* __restrict__ gate,
                                      float* __restrict__ o) {
  const int h = blockIdx.x, d = threadIdx.x;
  const float* p = part + (size_t)h * ATTN_SPLITS * 258;
  float M = -INFINITY;
  for (int s = 0; s < ATTN_SPLITS; ++s) M = fmaxf(M, p[s * 258]);
  float L = 0.f, acc = 0.f;
  for (int s = 0; s < ATTN_SPLITS; ++s) {
    const float m = p[s * 258];
    if (m == -INFINITY) continue;
    const float f = expf(m - M);
    L += p[s * 258 + 1] * f;
    acc += p[s * 258 + 2 + d] * f;
  }
  o[h * 256 + d] = acc / L * sigmoid(gate[h * 256 + d]);
}

API int dr_attn_decode(const float* q, const bf16* kc, const bf16* vc, const float* gate, int H, int Hkv, int hd,
                       const int* pos_base, int pos_off, float* part, float* o, cudaStream_t st) {
  if (hd != 256) return -1;
  k_attn_decode_part<<<dim3(H, ATTN_SPLITS), ATTN_WARPS * 32, 0, st>>>(q, kc, vc, H, Hkv, pos_base, pos_off, part);
  k_attn_decode_combine<<<H, 256, 0, st>>>(part, gate, o);
  return 0;
}

API int dr_attn_part_floats(void) { return ATTN_SPLITS * 258; }

// ------------------------------------------------------------------------------------------
// linear attention (Gated DeltaNet)

// Depthwise causal conv (kernel 4) + SiLU over rows; state keeps the last 3 inputs per channel.
__global__ void k_conv(const float* __restrict__ in, int in_stride, const float* __restrict__ w,
                       float* __restrict__ state, float* __restrict__ out, int n, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float h0 = state[c * 3], h1 = state[c * 3 + 1], h2 = state[c * 3 + 2];
  const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  for (int t = 0; t < n; ++t) {
    const float xt = in[(size_t)t * in_stride + c];
    out[(size_t)t * C + c] = silu(w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt);
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }
  state[c * 3] = h0;
  state[c * 3 + 1] = h1;
  state[c * 3 + 2] = h2;
}

API int dr_conv(const float* in, int in_stride, const float* w, float* state, float* out, int n, int C,
                cudaStream_t st) {
  k_conv<<<(C + 255) / 256, 256, 0, st>>>(in, in_stride, w, state, out, n, C);
  return 0;
}

// One block per value head (4 warps, one thread per value column). Each thread keeps its
// 128-row state column in registers for the whole sequence. The output gets the gated
// RMSNorm (weight norm_w, gate silu(z)) applied in place.
// conv rows: [q (nk*128) | k (nk*128) | v (nv*128)]; proj rows hold z, b, a at the offsets.
__global__ void __launch_bounds__(128)
    k_gdn(const float* __restrict__ conv, const float* __restrict__ proj, int ps, int z_off, int b_off, int a_off,
          const float* __restrict__ A_log, const float* __restrict__ dt_bias, const float* __restrict__ norm_w,
          float* __restrict__ state, float* __restrict__ o, int n, int nk, int nv, float eps) {
  constexpr int DK = 128;
  __shared__ float qs[DK], ks[DK];
  __shared__ float red[4];
  const int h = blockIdx.x, v = threadIdx.x, lane = v & 31, w = v >> 5;
  const int kh = h / (nv / nk);
  const int cs = 2 * nk * DK + nv * DK;
  float S[DK];
  float* sp = state + (size_t)h * DK * DK;
#pragma unroll
  for (int k = 0; k < DK; ++k) S[k] = sp[k * DK + v];
  const float A = -expf(A_log[h]);
  const float dtb = dt_bias[h];
  const float qscale = rsqrtf((float)DK);
  const float nwv = norm_w[v];
  for (int t = 0; t < n; ++t) {
    const float* cr = conv + (size_t)t * cs;
    const float* pr = proj + (size_t)t * ps;
    if (w == 0) {
      float qv[4], kv[4], qss = 0.f, kss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        qv[i] = cr[kh * DK + lane + 32 * i];
        kv[i] = cr[nk * DK + kh * DK + lane + 32 * i];
        qss += qv[i] * qv[i];
        kss += kv[i] * kv[i];
      }
      const float qi = rsqrtf(warp_sum(qss) + 1e-6f) * qscale, ki = rsqrtf(warp_sum(kss) + 1e-6f);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        qs[lane + 32 * i] = qv[i] * qi;
        ks[lane + 32 * i] = kv[i] * ki;
      }
    }
    const float vv = cr[2 * nk * DK + h * DK + v];
    const float z = pr[z_off + h * DK + v];
    const float decay = expf(A * softplus(pr[a_off + h] + dtb));
    const float beta = sigmoid(pr[b_off + h]);
    __syncthreads();
    float kvm = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
      S[k] *= decay;
      kvm += S[k] * ks[k];
    }
    const float delta = (vv - kvm) * beta;
    float out = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
      S[k] += ks[k] * delta;
      out += S[k] * qs[k];
    }
    const float ws = warp_sum(out * out);
    if (lane == 0) red[w] = ws;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / DK + eps);
    o[(size_t)t * nv * DK + h * DK + v] = out * inv * nwv * silu(z);
  }
#pragma unroll
  for (int k = 0; k < DK; ++k) sp[k * DK + v] = S[k];
}

API int dr_gdn(const float* conv, const float* proj, int ps, int z_off, int b_off, int a_off, const float* A_log,
               const float* dt_bias, const float* norm_w, float* state, float* o, int n, int nk, int nv, int dk, int dv,
               float eps, cudaStream_t st) {
  if (dk != 128 || dv != 128) return -1;
  k_gdn<<<nv, 128, 0, st>>>(conv, proj, ps, z_off, b_off, a_off, A_log, dt_bias, norm_w, state, o, n, nk, nv, eps);
  return 0;
}

// ------------------------------------------------------------------------------------------
// sampling (one block of 1024 threads; logits sorted in shared memory)

struct SampleParams {
  float temperature;
  int top_k;
  float top_p;
  int pad;
  unsigned long long seed;
};

constexpr int SAMPLE_THREADS = 1024;

template <int NP>
DEV void bitonic_desc(float* key, int* idx) {
  for (int k = 2; k <= NP; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < NP; i += SAMPLE_THREADS) {
        const int p = i ^ j;
        if (p > i) {
          const bool desc = (i & k) == 0;
          const float a = key[i], b = key[p];
          const int ia = idx[i], ib = idx[p];
          // "a before b" in descending order with index tie-break.
          const bool a_first = a > b || (a == b && ia < ib);
          if (desc != a_first) {
            key[i] = b, key[p] = a;
            idx[i] = ib, idx[p] = ia;
          }
        }
      }
      __syncthreads();
    }
  }
}

// Draw a rank from sorted (descending) keys[0..n) with sglang-omni's top-k / top-p rule:
// drop ranks >= top_k and ranks whose inclusive cumulative probability exceeds top_p,
// always keeping rank 0; then sample proportionally among the kept ranks.
template <int NP>
DEV int draw_rank(const float* key, int n, float temperature, int top_k, float top_p, float u, float* red,
                  float* scan) {
  constexpr int C = NP / SAMPLE_THREADS;
  const float inv_t = 1.f / fmaxf(temperature, 1e-5f);
  const float k0 = key[0];
  const int base = threadIdx.x * C;
  float pre[C];
  float run = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int i = base + c;
    run += i < n ? expf((key[i] - k0) * inv_t) : 0.f;
    pre[c] = run;
  }
  // Exclusive scan of per-thread totals.
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  float incl = run;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float t = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += t;
  }
  __syncthreads();
  if (lane == 31) scan[w] = incl;
  __syncthreads();
  if (w == 0) {
    float v = scan[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float t = __shfl_up_sync(0xffffffffu, v, o);
      if (lane >= o) v += t;
    }
    scan[32 + lane] = v;  // inclusive warp totals
  }
  __syncthreads();
  const float offset = (w ? scan[32 + w - 1] : 0.f) + incl - run;
  const float Z = scan[32 + 31];
  float cnt = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    pre[c] += offset;
    const int i = base + c;
    if (i < n && pre[c] <= top_p * Z) cnt += 1.f;
  }
  cnt = block_sum(cnt, red);
  int K = (int)cnt;
  if (K > top_k) K = top_k;
  if (K < 1) K = 1;
  __syncthreads();
#pragma unroll
  for (int c = 0; c < C; ++c)
    if (base + c == K - 1) scan[64] = pre[c];
  __syncthreads();
  const float target = u * scan[64];
  int best = 1 << 30;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int i = base + c;
    if (i < K && pre[c] >= target && i < best) best = i;
  }
  // block min
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) best = min(best, __shfl_xor_sync(0xffffffffu, best, o));
  __syncthreads();
  if (lane == 0) scan[w] = (float)best;
  __syncthreads();
  float b = lane < (SAMPLE_THREADS / 32) ? scan[lane] : 1e30f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) b = fminf(b, __shfl_xor_sync(0xffffffffu, b, o));
  int rank = (int)b;
  if (rank >= K) rank = K - 1;  // rounding guard
  __syncthreads();
  return rank;
}

DEV float philox_uniform(unsigned long long seed, unsigned long long stream, unsigned long long step) {
  curandStatePhilox4_32_10_t s;
  curand_init(seed, stream, step * 4ull, &s);
  return curand_uniform(&s);
}

template <int NP>
DEV void load_sorted(const float* __restrict__ logits, int n, float* key, int* idx) {
  for (int i = threadIdx.x; i < NP; i += SAMPLE_THREADS) {
    key[i] = i < n ? logits[i] : -INFINITY;
    idx[i] = i;
  }
  __syncthreads();
  bitonic_desc<NP>(key, idx);
}

// ---- fast path: exact top-32 by warp-local selection + a one-warp merge

constexpr int TOPC = 32;
constexpr int PER_T = 5;  // elements per thread (n <= 5 * 1024)

struct TopShared {
  float wv[32 * TOPC];
  int wi[32 * TOPC];
  float tv[TOPC];
  int ti[TOPC];
  float red[32];
  int flag;
};

DEV bool better(float av, int ai, float bv, int bi) { return av > bv || (av == bv && ai < bi); }

DEV void warp_argmax(float& v, int& i, int& l) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    const int ol = __shfl_xor_sync(0xffffffffu, l, o);
    if (better(ov, oi, v, i)) v = ov, i = oi, l = ol;
  }
}

// Leaves the 32 largest logits (descending, index tie-break) in sh.tv / sh.ti and each
// thread's own elements in v / id.
DEV void top32(const float* __restrict__ logits, int n, TopShared& sh, float* v, int* id) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int e = 0; e < PER_T; ++e) {
    const int i = threadIdx.x + SAMPLE_THREADS * e;
    v[e] = i < n ? logits[i] : -INFINITY;
    id[e] = i < n ? i : 0x7fffffff;
  }
  float rv[PER_T];
#pragma unroll
  for (int e = 0; e < PER_T; ++e) rv[e] = v[e];
  for (int r = 0; r < TOPC; ++r) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, be = 0;
#pragma unroll
    for (int e = 0; e < PER_T; ++e)
      if (better(rv[e], id[e], bv, bi)) bv = rv[e], bi = id[e], be = e;
    float mv = bv;
    int mi = bi, ml = lane;
    warp_argmax(mv, mi, ml);
    if (lane == ml) {
#pragma unroll
      for (int e = 0; e < PER_T; ++e)
        if (e == be) rv[e] = -INFINITY;
    }
    if (lane == 0) sh.wv[w * TOPC + r] = mv, sh.wi[w * TOPC + r] = mi;
  }
  __syncthreads();
  if (w == 0) {
    int ptr = 0;
    for (int r = 0; r < TOPC; ++r) {
      float hv = ptr < TOPC ? sh.wv[lane * TOPC + ptr] : -INFINITY;
      int hi = ptr < TOPC ? sh.wi[lane * TOPC + ptr] : 0x7fffffff;
      int hl = lane;
      warp_argmax(hv, hi, hl);
      if (lane == hl) ++ptr;
      if (lane == 0) sh.tv[r] = hv, sh.ti[r] = hi;
    }
  }
  __syncthreads();
}

// Z = sum over all logits of exp((l - max) / T), every thread gets it.
DEV float softmax_z(const float* v, float mx, float inv_t, float* red) {
  float z = 0.f;
#pragma unroll
  for (int e = 0; e < PER_T; ++e) z += v[e] == -INFINITY ? 0.f : expf((v[e] - mx) * inv_t);
  return block_sum(z, red);
}

// Warp 0: draw a rank among the top 32; returns -1 if the kept set may extend past 32.
DEV int draw_top32(const TopShared& sh, float Z, float temperature, int top_k, float top_p, float u) {
  const int lane = threadIdx.x & 31;
  const float inv_t = 1.f / fmaxf(temperature, 1e-5f);
  float P = sh.tv[lane] == -INFINITY ? 0.f : expf((sh.tv[lane] - sh.tv[0]) * inv_t);
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float t = __shfl_up_sync(0xffffffffu, P, o);
    if (lane >= o) P += t;
  }
  const int cnt = __popc(__ballot_sync(0xffffffffu, P <= top_p * Z));
  if (cnt == TOPC && top_k > TOPC) return -1;
  int K = min(top_k, cnt);
  if (K < 1) K = 1;
  const float target = u * __shfl_sync(0xffffffffu, P, K - 1);
  const unsigned hit = __ballot_sync(0xffffffffu, lane < K && P >= target);
  return hit ? __ffs(hit) - 1 : K - 1;
}

// Full-sort fallback over all n logits; returns the sampled logit index.
template <int NP>
DEV int draw_full(const float* __restrict__ logits, int n, float temperature, int top_k, float top_p, float u,
                  float* key, int* idx, float* red, float* scan) {
  load_sorted<NP>(logits, n, key, idx);
  const int r = draw_rank<NP>(key, n, temperature, top_k, top_p, u, red, scan);
  return idx[r];
}

// Main token: logits over [eos, semantic 0..cb-1]. First draw with the request's params,
// a RAS draw with (1.0, 30, 0.95); if the first draw is semantic and repeats one of the last
// 16 main tokens, the RAS draw wins. Writes frame[0] = main token, frame[1] = code 0.
__global__ void __launch_bounds__(SAMPLE_THREADS)
    k_sample_main(const float* __restrict__ logits, int n, const SampleParams* __restrict__ prm,
                  const int* __restrict__ ras, int ras_len, const unsigned long long* __restrict__ step, int eos, int sem0, int cb, int* __restrict__ frame) {
  constexpr int NP = 8192;
  extern __shared__ float smem[];
  __shared__ TopShared sh;
  __shared__ float scan[96];
  __shared__ int choice[2];
  float v[PER_T];
  int id[PER_T];
  top32(logits, n, sh, v, id);
  const SampleParams p = *prm;
  const unsigned long long s = *step;
  const float u1 = philox_uniform(p.seed, 0, s), u2 = philox_uniform(p.seed, 1, s);
  const float mx = sh.tv[0];
  const float z1 = softmax_z(v, mx, 1.f / fmaxf(p.temperature, 1e-5f), sh.red);
  const float z2 = softmax_z(v, mx, 1.f, sh.red);
  if (threadIdx.x < 32) {
    const int r1 = draw_top32(sh, z1, p.temperature, p.top_k, p.top_p, u1);
    const int r2 = draw_top32(sh, z2, 1.0f, 30, 0.95f, u2);
    if (threadIdx.x == 0) {
      choice[0] = r1 < 0 ? -1 : sh.ti[r1];
      choice[1] = sh.ti[r2];
    }
  }
  __syncthreads();
  if (choice[0] < 0) {  // uniform across the block
    const int c = draw_full<NP>(logits, n, p.temperature, p.top_k, p.top_p, u1, smem,
                                reinterpret_cast<int*>(smem + NP), sh.red, scan);
    if (threadIdx.x == 0) choice[0] = c;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int c1 = choice[0], c2 = choice[1];
    const int t1 = c1 == 0 ? eos : sem0 + c1 - 1;
    const int t2 = c2 == 0 ? eos : sem0 + c2 - 1;
    bool repeat = false;
    for (int i = 0; i < ras_len; ++i) repeat |= ras[i] == t1;
    const int main_tok = (repeat && c1 != 0) ? t2 : t1;
    frame[0] = main_tok;
    frame[1] = clampi(main_tok - sem0, 0, cb - 1);
  }
}

// Codebook head: writes frame[col] = sampled code.
__global__ void __launch_bounds__(SAMPLE_THREADS)
    k_sample_head(const float* __restrict__ logits, int n, const SampleParams* __restrict__ prm,
                  int stream, const unsigned long long* __restrict__ step,
                  int* __restrict__ frame, int col) {
  constexpr int NP = 4096;
  extern __shared__ float smem[];
  __shared__ TopShared sh;
  __shared__ float scan[96];
  __shared__ int choice;
  float v[PER_T];
  int id[PER_T];
  top32(logits, n, sh, v, id);
  const SampleParams p = *prm;
  const float u = philox_uniform(p.seed, stream, *step);
  const float z = softmax_z(v, sh.tv[0], 1.f / fmaxf(p.temperature, 1e-5f), sh.red);
  if (threadIdx.x < 32) {
    const int r = draw_top32(sh, z, p.temperature, p.top_k, p.top_p, u);
    if (threadIdx.x == 0) choice = r < 0 ? -1 : sh.ti[r];
  }
  __syncthreads();
  if (choice < 0) {
    const int c = draw_full<NP>(logits, n, p.temperature, p.top_k, p.top_p, u, smem,
                                reinterpret_cast<int*>(smem + NP), sh.red, scan);
    if (threadIdx.x == 0) choice = c;
    __syncthreads();
  }
  if (threadIdx.x == 0) frame[col] = choice;
}

API int dr_sample_init(void) {
  cudaError_t e = cudaFuncSetAttribute(k_sample_main, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192 * 8);
  if (e != cudaSuccess) return (int)e;
  e = cudaFuncSetAttribute(k_sample_head, cudaFuncAttributeMaxDynamicSharedMemorySize, 4096 * 8);
  return (int)e;
}

API int dr_sample_main(const float* logits, int n, const void* prm, const int* ras, int ras_len,
                       const unsigned long long* step, int eos, int sem0, int cb, int* frame, cudaStream_t st) {
  if (n > 8192) return -1;
  k_sample_main<<<1, SAMPLE_THREADS, 8192 * 8, st>>>(logits, n, (const SampleParams*)prm, ras, ras_len, step,
                                                     eos, sem0, cb, frame);
  return 0;
}

API int dr_sample_head(const float* logits, int n, const void* prm, int stream, const unsigned long long* step,
                       int* frame, int col, cudaStream_t st) {
  if (n > 4096) return -1;
  k_sample_head<<<1, SAMPLE_THREADS, 4096 * 8, st>>>(logits, n, (const SampleParams*)prm, stream, step, frame,
                                                     col);
  return 0;
}

// End of a decode step: the sampled frame becomes the next input, it is appended to the
// history and the RAS window, and the position and RNG step advance.
__global__ void k_frame_end(const int* __restrict__ next, int* __restrict__ in_frame, int* __restrict__ ras,
                            int ras_len, int* __restrict__ pos, unsigned long long* __restrict__ step,
                            int* __restrict__ hist, int* __restrict__ hist_len, int hist_cap, int width) {
  if (threadIdx.x != 0) return;
  for (int i = 0; i < width; ++i) in_frame[i] = next[i];
  for (int i = 0; i + 1 < ras_len; ++i) ras[i] = ras[i + 1];
  ras[ras_len - 1] = next[0];
  pos[0] += 1;
  step[0] += 1;
  const int h = hist_len[0];
  if (h < hist_cap) {
    for (int i = 0; i < width; ++i) hist[h * width + i] = next[i];
    hist_len[0] = h + 1;
  }
}

API int dr_frame_end(const int* next, int* in_frame, int* ras, int ras_len, int* pos, unsigned long long* step,
                     int* hist, int* hist_len, int hist_cap, int width, cudaStream_t st) {
  k_frame_end<<<1, 32, 0, st>>>(next, in_frame, ras, ras_len, pos, step, hist, hist_len, hist_cap, width);
  return 0;
}

API int dr_last_error(void) { return (int)cudaGetLastError(); }

// ------------------------------------------------------------------------------------------
// runtime + cuBLAS wrappers (so the Rust side never depends on CUDA header macros)

#include <cublas_v2.h>

API const char* dr_error_string(int e) { return cudaGetErrorString((cudaError_t)e); }
API int dr_malloc(void** p, size_t n) { return (int)cudaMalloc(p, n); }
API int dr_free(void* p) { return (int)cudaFree(p); }
API int dr_host_alloc(void** p, size_t n) { return (int)cudaMallocHost(p, n); }
API int dr_host_free(void* p) { return (int)cudaFreeHost(p); }
// Written once by the CPU and read by the GPU only: write-combined, mapped so kernels read it over PCIe.
API int dr_host_alloc_mapped(void** p, void** dev, size_t n) {
  cudaError_t e = cudaHostAlloc(p, n, cudaHostAllocMapped | cudaHostAllocPortable | cudaHostAllocWriteCombined);
  if (e != cudaSuccess) return (int)e;
  return (int)cudaHostGetDevicePointer(dev, *p, 0);
}
API int dr_h2d(void* d, const void* h, size_t n, cudaStream_t st) {
  return (int)cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, st);
}
API int dr_d2h(void* h, const void* d, size_t n, cudaStream_t st) {
  return (int)cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, st);
}
API int dr_d2d(void* d, const void* s, size_t n, cudaStream_t st) {
  return (int)cudaMemcpyAsync(d, s, n, cudaMemcpyDeviceToDevice, st);
}
API int dr_memset(void* d, int v, size_t n, cudaStream_t st) { return (int)cudaMemsetAsync(d, v, n, st); }
API int dr_stream_create(cudaStream_t* s) { return (int)cudaStreamCreateWithFlags(s, cudaStreamNonBlocking); }
API int dr_stream_sync(cudaStream_t s) { return (int)cudaStreamSynchronize(s); }
API int dr_device_sync(void) { return (int)cudaDeviceSynchronize(); }
API int dr_mem_info(size_t* free_b, size_t* total_b) { return (int)cudaMemGetInfo(free_b, total_b); }

API int dr_graph_begin(cudaStream_t s) { return (int)cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); }
API int dr_graph_end(cudaStream_t s, cudaGraphExec_t* exec) {
  cudaGraph_t g;
  cudaError_t e = cudaStreamEndCapture(s, &g);
  if (e != cudaSuccess) return (int)e;
  e = cudaGraphInstantiate(exec, g, 0);
  cudaGraphDestroy(g);
  return (int)e;
}
API int dr_graph_launch(cudaGraphExec_t exec, cudaStream_t s) { return (int)cudaGraphLaunch(exec, s); }
API int dr_graph_destroy(cudaGraphExec_t exec) { return (int)cudaGraphExecDestroy(exec); }

API int dr_cublas_create(void** h, cudaStream_t s) {
  cublasHandle_t handle;
  cublasStatus_t r = cublasCreate(&handle);
  if (r != CUBLAS_STATUS_SUCCESS) return (int)r;
  r = cublasSetStream(handle, s);
  *h = handle;
  return (int)r;
}

// C[M, N] (row-major, f32) = A[M, K] (bf16) . W[N, K]^T (bf16) + beta * C
API int dr_gemm(void* h, const bf16* A, const bf16* W, float* C, int M, int N, int K, float beta) {
  const float alpha = 1.f;
  return (int)cublasGemmEx((cublasHandle_t)h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16BF, K, A,
                           CUDA_R_16BF, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
