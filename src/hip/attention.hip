#pragma once
#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>

// Enable/disable non-temporal loads (can be overridden at compile time)
#ifndef FA_USE_NT
#define FA_USE_NT 0
#endif

// ---- NT policy: only stream when no L2 reuse is expected ----
__device__ __forceinline__ bool use_nt_for_kv(int kv_mul, int B, int H, int head_dim, int T) {
    // Heuristics:
    // - Disable NT if multiple heads share K/V (kv_mul>1) => let L2 share.
    // - Disable NT for small batch*heads (not enough outstanding misses to hide latency).
    // - Enable NT for long sequences where we truly stream.
    return (FA_USE_NT != 0) && (kv_mul == 1) && (B * H >= 8) && (T >= 256) && (head_dim >= 64);
}

template<typename T>
__device__ __forceinline__ T ld_global_cond(const T* p, bool nt) {
#if defined(__HIP_DEVICE_COMPILE__)
    // Only use NT loads for supported types (integer, float, pointer, not bf16 or vector types)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                  std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        if (nt) return __builtin_nontemporal_load(p);
    }
#endif
    return *p;
}

// BF16 packed (two bf16 in a u32)
__device__ __forceinline__ uint32_t ld_global_u32_cond(const uint32_t* p, bool nt) {
#if defined(__HIP_DEVICE_COMPILE__)
    if (nt) return __builtin_nontemporal_load(p);
#endif
    return *p;
}

static __device__ __forceinline__ float wave_reduce_sum(float v) {
    #pragma unroll
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off, warpSize);
    return v;
}

static __device__ __forceinline__ void wave_sync() {
#if defined(__CUDA_ARCH__)
    __syncwarp();
#elif defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
#endif
}

// ======================= Ring index helpers =======================
static __device__ __forceinline__ bool is_pow2_int(int x){ return (x & (x-1)) == 0; }
template<bool POW2>
static __device__ __forceinline__ int ring_index(int t, int cap) {
    if constexpr (POW2) return t & (cap - 1);
    else                return (t >= cap) ? (t % cap) : t;
}

// ======================= K/V vectorized access =======================
static inline __device__ float bf16_to_f32_intr(__hip_bfloat16 x) {
    return __bfloat162float(x);
}

static __device__ __forceinline__ __hip_bfloat16 raw_to_bf16(uint16_t bits) {
    union {
        uint16_t u;
        __hip_bfloat16 b;
    } conv;
    conv.u = bits;
    return conv.b;
}

// Unpack two bf16 from u32
__device__ __forceinline__ void unpack_bf16x2(uint32_t packed, float& lo, float& hi) {
    __hip_bfloat16 bf_lo = raw_to_bf16((uint16_t)(packed & 0xFFFF));
    __hip_bfloat16 bf_hi = raw_to_bf16((uint16_t)(packed >> 16));
    lo = bf16_to_f32_intr(bf_lo);
    hi = bf16_to_f32_intr(bf_hi);
}

// ======================= LDS bank conflict mitigation =======================
// one pad every 32 floats (per 128B) to break 32-bank conflicts
#define PAD32_INDEX(j) ((j) + ((j) >> 5))  // j + j/32

__device__ __forceinline__ bool is_aligned_16(const void* p) {
    return ((uintptr_t)p & 0xF) == 0;
}

__device__ __forceinline__ bool is_aligned_4(const void* p) {
    return ((uintptr_t)p & 3) == 0;
}

// Coalesced store from padded shared O to global contiguous O
__device__ void store_O_from_padded(float* __restrict__ dst,
                                    const float* __restrict__ o_sh,
                                    int head_dim)
{
    // Fast path: aligned v4
    if (is_aligned_16(dst)) {
        int vec = head_dim >> 2;      // /4
        // cooperate: each thread writes many float4s spaced by blockDim.x
        float4* d4 = reinterpret_cast<float4*>(dst);
        for (int i = threadIdx.x; i < vec; i += blockDim.x) {
            int j = i << 2;
            float4 v;
            v.x = o_sh[PAD32_INDEX(j+0)];
            v.y = o_sh[PAD32_INDEX(j+1)];
            v.z = o_sh[PAD32_INDEX(j+2)];
            v.w = o_sh[PAD32_INDEX(j+3)];
            d4[i] = v;
        }
        int tail = head_dim & 3;
        int base = vec << 2;
        if (tail && threadIdx.x == 0) {
            for (int t = 0; t < tail; ++t) dst[base + t] = o_sh[PAD32_INDEX(base + t)];
        }
    } else {
        // Scalar fallback (still coalesced across lanes)
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            dst[j] = o_sh[PAD32_INDEX(j)];
        }
    }
}

// ======================================================================
// FA2 tiled decode kernel (Br=1) for head_dim==64
// - Tile only across KV columns (Bc>1), per-tile renormalization.
// - No score matrix materialization; two-pass per tile: (S,max) then V mix.
// - Supports GQA, sliding-window, optional mask, BF16/FP32 KV.
// ======================================================================
template<typename KV_T, bool USE_MASK, int BcMax=64>
__global__ void fa2_decode_br1_head64(
    const float* __restrict__ q_batch,      // (B, H*64)
    const void*  __restrict__ k_cache_base, // base (byte addr)
    const void*  __restrict__ v_cache_base, // base (byte addr)
    const float* __restrict__ mask,         // (T,T) if USE_MASK
    const __hip_bfloat16* __restrict__ attn_sinks_half, // (H) or nullptr
    float* __restrict__ tb_batch,           // (B, H*64)  [OUTPUT]
    // sizes & layout
    int B, int seq_len, int /*head_dim==64*/, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    // positions, batch mapping, kv layer info
    const int* __restrict__ d_pos_per_token,  // (B)
    const int* __restrict__ d_batch_indices,  // (B)
    long long /*B_stride*/,                   // unused with layer offsets
    const long long* __restrict__ d_layer_kv_off,
    const int*       __restrict__ d_layer_kv_cap,
    const int*       __restrict__ d_layer_is_local,
    int bc_tile_hint // runtime Bc to try (<=BcMax)
){
    const int h    = blockIdx.x;
    const int b    = blockIdx.y;
    const int lane = threadIdx.x;            // wave64

    if (h >= n_attn_heads || b >= B) return;

    // ---- Head grouping (GQA) ----
    const int g = h / kv_mul;

    // ---- Token position and window ----
    const int pos_b = d_pos_per_token[b];
    const int Lb    = pos_b + 1;

    // Local sliding-window on even layers: either clamp range (no mask) or use mask
    const int local = d_layer_is_local[layer_idx];
    const bool local_nomask = (local != 0) && !USE_MASK && (sliding_window > 0) && ((layer_idx & 1) == 0);
    const int t_window_start = local_nomask ? max(0, pos_b - sliding_window + 1) : 0;
    const int t_start = local_nomask ? t_window_start : 0;
    const int t_end   = Lb;   // causal upper bound

    // ---- KV addressing (ring buffer per batch/layer) ----
    const long long kv_base = d_layer_kv_off[layer_idx];
    const int       cap     = d_layer_kv_cap[layer_idx];
    const bool      cap_pow2= is_pow2_int(cap);

    const int  gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + (long long)gb * (long long)cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = (const char*)k_cache_base + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = (const char*)v_cache_base + batch_off_elems * elem_size;

    const long long head_off_bytes = ((long long)g * 64) * elem_size;
    const long long step_bytes     = ((long long)kv_dim) * elem_size;

    // ---- NT streaming heuristic ----
    const bool NT = use_nt_for_kv(kv_mul, B, n_attn_heads, 64, Lb);

    // ---- Shared memory layout ----
    extern __shared__ float smem[];
    float* __restrict__ q_sh = smem;                                 // padded
    float* __restrict__ O_sh = q_sh + (64 + (64 >> 5) + 1);          // padded
    float* __restrict__ s_sh = O_sh + (64 + (64 >> 5) + 1);          // up to BcMax scalars

    // ---- Load & scale Q, zero O ----
    const float* __restrict__ q_b    = q_batch + (long long)b * (n_attn_heads * 64);
    const float* __restrict__ q_head = q_b + (long long)h * 64;
    float* __restrict__ o_out        = tb_batch + (long long)b * (n_attn_heads * 64) + (long long)h * 64;

    const float inv_sqrt_d = rsqrtf(64.0f);
    for (int j = lane; j < 64; j += warpSize) {
        q_sh[PAD32_INDEX(j)] = q_head[j] * inv_sqrt_d;
        O_sh[PAD32_INDEX(j)] = 0.0f;
    }
    wave_sync();

    // ---- Running LSE state (m, l) ----
    __shared__ float m_shared, l_shared;
    if (lane == 0) { m_shared = -std::numeric_limits<float>::infinity(); l_shared = 0.0f; }
    wave_sync();

    // ---- Tile loop over time ----
    const int BcTry = (bc_tile_hint > 0 && bc_tile_hint <= BcMax) ? bc_tile_hint : BcMax;
    for (int t0 = t_start; t0 < t_end; t0 += BcTry) {
        const int bc = min(BcTry, t_end - t0);

        // -------- Pass 1: compute scores s[t] and the tile rowmax --------
        float m_tile = -std::numeric_limits<float>::infinity();

        #pragma unroll 1
        for (int r = 0; r < bc; ++r) {
            const int t = t0 + r;
            const int rt = (local != 0) ? (cap_pow2 ? (t & (cap - 1)) : (t % cap)) : t;

            const char* __restrict__ kp_bytes = k_layer_base + (long long)rt * step_bytes + head_off_bytes;

            float partial = 0.0f;
            if constexpr (std::is_same<KV_T,float>::value) {
                const float* __restrict__ k_head = reinterpret_cast<const float*>(kp_bytes);
                if (lane < 64) {
                    float kv = ld_global_cond(k_head + lane, NT);
                    partial = q_sh[PAD32_INDEX(lane)] * kv;
                }
            } else { // BF16 path: 32 lanes load (2 elems each)
                if (lane < 32) {
                    if (is_aligned_4(kp_bytes)) {
                        const uint32_t* __restrict__ k2 = reinterpret_cast<const uint32_t*>(kp_bytes);
                        uint32_t w = ld_global_u32_cond(k2 + lane, NT);
                        float k0, k1; unpack_bf16x2(w, k0, k1);
                        int j = lane << 1;
                        partial = q_sh[PAD32_INDEX(j+0)] * k0 + q_sh[PAD32_INDEX(j+1)] * k1;
                    } else {
                        const __hip_bfloat16* __restrict__ k16 = reinterpret_cast<const __hip_bfloat16*>(kp_bytes);
                        int j = lane << 1;
                        float k0 = bf16_to_f32_intr(k16[j+0]);
                        float k1 = bf16_to_f32_intr(k16[j+1]);
                        partial = q_sh[PAD32_INDEX(j+0)] * k0 + q_sh[PAD32_INDEX(j+1)] * k1;
                    }
                }
            }

            // Reduce dot(q, k_t) across the wave
            float s = wave_reduce_sum(partial);
            if (lane == 0) {
                if constexpr (USE_MASK) s += mask[(long long)pos_b * seq_len + t];
                s_sh[r] = s;
                m_tile = fmaxf(m_tile, s);
            }
            wave_sync();
        }

        // -------- Per-tile (m,l) update --------
        __shared__ float scale_old, inv_l_new, m_new_shared;
        if (lane == 0) {
            const float m_prev = m_shared;
            const float l_prev = l_shared;

            const float m_new  = fmaxf(m_prev, m_tile);
            const float alpha  = __expf(m_prev - m_new);

            float sumexp = 0.0f;
            for (int r = 0; r < bc; ++r) sumexp += __expf(s_sh[r] - m_new);

            const float l_new = alpha * l_prev + sumexp;

            scale_old    = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            inv_l_new    = (l_new > 0.0f) ? 1.0f / l_new : 0.0f;
            m_new_shared = m_new;
            m_shared     = m_new;
            l_shared     = l_new;
        }
        wave_sync();

        // Pre-scale old O once
        for (int j = lane; j < 64; j += warpSize)
            O_sh[PAD32_INDEX(j)] *= scale_old;
        wave_sync();

        // -------- Pass 2: mix V with weights exp(s - m_new)/l_new --------
        #pragma unroll 1
        for (int r = 0; r < bc; ++r) {
            const int t  = t0 + r;
            const int rt = (local != 0) ? (cap_pow2 ? (t & (cap - 1)) : (t % cap)) : t;

            const char* __restrict__ vp_bytes = v_layer_base + (long long)rt * step_bytes + head_off_bytes;

            // Reuse s_sh[r] to broadcast weight to all lanes
            if (lane == 0) s_sh[r] = __expf(s_sh[r] - m_new_shared) * inv_l_new;
            wave_sync();
            const float w = s_sh[r];

            if constexpr (std::is_same<KV_T,float>::value) {
                const float* __restrict__ v_head = reinterpret_cast<const float*>(vp_bytes);
                if (lane < 64) {
                    float vv = ld_global_cond(v_head + lane, NT);
                    O_sh[PAD32_INDEX(lane)] += w * vv;
                }
            } else { // BF16
                if (lane < 32) {
                    if (is_aligned_4(vp_bytes)) {
                        const uint32_t* __restrict__ v2 = reinterpret_cast<const uint32_t*>(vp_bytes);
                        uint32_t w2 = ld_global_u32_cond(v2 + lane, NT);
                        float v0, v1; unpack_bf16x2(w2, v0, v1);
                        int j = lane << 1;
                        O_sh[PAD32_INDEX(j+0)] += w * v0;
                        O_sh[PAD32_INDEX(j+1)] += w * v1;
                    } else {
                        const __hip_bfloat16* __restrict__ v16 = reinterpret_cast<const __hip_bfloat16*>(vp_bytes);
                        int j = lane << 1;
                        float v0 = bf16_to_f32_intr(v16[j+0]);
                        float v1 = bf16_to_f32_intr(v16[j+1]);
                        O_sh[PAD32_INDEX(j+0)] += w * v0;
                        O_sh[PAD32_INDEX(j+1)] += w * v1;
                    }
                }
            }
            wave_sync();
        }
    }

    // ---- Optional attention sink post-renorm (kept same as your decode) ----
    if (attn_sinks_half != nullptr) {
        float sink_s = __bfloat162float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m_shared; const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, sink_s);
            const float a = __expf(m_prev - m_new) * l_prev;
            const float b = __expf(sink_s - m_new);
            const float denom = a + b;
            const float renorm= (denom > 0.0f) ? (a / denom) : 1.0f;
            for (int j = lane; j < 64; j += warpSize)
                O_sh[PAD32_INDEX(j)] *= renorm;
            wave_sync();
            m_shared = m_new; l_shared = denom; // optional: keep LSE consistent
        }
    }

    // ---- Store O ----
    store_O_from_padded(o_out, O_sh, 64);
}

static inline int fa_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }

void fa(
    const float* q_batch,      // (B, H*D)
    const void*  k_cache,      // base pointer
    const void*  v_cache,      // base pointer
    const float* mask,         // (T, T)
    const __hip_bfloat16* attn_sinks,  // (L, H) — pass attn_sinks + layer_idx*H
    float* tb_batch,           // (B, H*D)
    int B,
    int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    int kv_cache_is_fp16,
    const int* d_pos_per_token,      // (B)
    const int* d_batch_indices,      // (B)
    long long B_stride,              // elements per (B*T*kv_dim) per layer
    int /*max_pos_in_batch*/,        // unused in new path
    float* /*workspace_partial_O*/,  // unused
    float* /*workspace_partial_m*/,  // unused
    float* /*workspace_partial_l*/,  // unused
    hipStream_t stream,
    const long long* d_layer_kv_off,
    const int* d_layer_kv_cap,
    const int* d_layer_is_local)
{
    if (B <= 0 || n_attn_heads <= 0 || head_dim != 64) return; // this path specializes D=64

    // ---- shared memory: q_sh + o_sh + s_sh(BcMax) ----
    auto padded = [](int n){ return n + (n >> 5) + 1; };
    constexpr int BcMax = 64;                 // safe default on MI250
    size_t shmem = (padded(64) + padded(64) + BcMax) * sizeof(float);

    const __hip_bfloat16* attn_sinks_head =
        (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0) && (mask != nullptr);

    dim3 grid(n_attn_heads, B);
    dim3 block(64); // one wave per (b,h)

    // Heuristic Bc to try: big enough to amortize LSE, small enough for latency
    // You can expose this as a runtime knob; 64 works well for D=64 on MI250.
    const int bc_tile_hint = min(64, seq_len); // kernel clamps per-token

    if (kv_cache_is_fp16) {
        if (use_mask) {
            hipLaunchKernelGGL(
                (fa2_decode_br1_head64<__hip_bfloat16, true, BcMax>),
                grid, block, shmem, stream,
                q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint
            );
        } else {
            hipLaunchKernelGGL(
                (fa2_decode_br1_head64<__hip_bfloat16, false, BcMax>),
                grid, block, shmem, stream,
                q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint
            );
        }
    } else {
        if (use_mask) {
            hipLaunchKernelGGL(
                (fa2_decode_br1_head64<float, true, BcMax>),
                grid, block, shmem, stream,
                q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint
            );
        } else {
            hipLaunchKernelGGL(
                (fa2_decode_br1_head64<float, false, BcMax>),
                grid, block, shmem, stream,
                q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint
            );
        }
    }
    CHECK_HIP(hipGetLastError());
}
