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

#if defined(__HIP_DEVICE_COMPILE__)
template<typename T>
__device__ __forceinline__ T nt_load(const T* ptr) {
    using base_t = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<base_t, float>  || std::is_same_v<base_t, double> ||
                  std::is_same_v<base_t, int32_t> || std::is_same_v<base_t, uint32_t> ||
                  std::is_same_v<base_t, int64_t> || std::is_same_v<base_t, uint64_t>) {
        return __builtin_nontemporal_load(ptr);
    } else {
        return *ptr;
    }
}

template<typename T>
__device__ __forceinline__ void nt_store(T* ptr, T value) {
    using base_t = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<base_t, float>  || std::is_same_v<base_t, double> ||
                  std::is_same_v<base_t, int32_t> || std::is_same_v<base_t, uint32_t> ||
                  std::is_same_v<base_t, int64_t> || std::is_same_v<base_t, uint64_t>) {
        __builtin_nontemporal_store(value, ptr);
    } else {
        *ptr = value;
    }
}
#else
template<typename T>
__device__ __forceinline__ T nt_load(const T* ptr) { return *ptr; }

template<typename T>
__device__ __forceinline__ void nt_store(T* ptr, T value) { *ptr = value; }
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

// Specialized kernel for head_dim==64 with increased ILP and prefetch
template<typename KV_T, bool USE_MASK>
__global__ void fa_decode_full_typed_head64(
    const float* __restrict__ q_batch,      // (B, H*D)
    const void*  __restrict__ k_cache_base, // byte addressable
    const void*  __restrict__ v_cache_base, // byte addressable
    const float* __restrict__ mask,         // (T,T) if USE_MASK
    const __hip_bfloat16* __restrict__ attn_sinks_half, // (H) or nullptr
    float* __restrict__ tb_batch,           // (B, H*D)  [OUTPUT]
    // sizes & layout
    int B, int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    // positions, batch mapping, kv layer info
    const int* __restrict__ d_pos_per_token,  // (B)
    const int* __restrict__ d_batch_indices,  // (B)
    long long B_stride,
    const long long* __restrict__ d_layer_kv_off,
    const int*       __restrict__ d_layer_kv_cap,
    const int*       __restrict__ d_layer_is_local
){
    // Same setup as regular kernel but with head_dim==64 assumption
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    if (h >= n_attn_heads || b >= B) return;

    const int g = h / kv_mul;
    const int pos_b = d_pos_per_token[b];
    const int Lb = pos_b + 1;
    const bool NT = use_nt_for_kv(kv_mul, B, n_attn_heads, 64, Lb);

    const float* __restrict__ q_b    = q_batch + (long long)b * (n_attn_heads * 64);
    const float* __restrict__ q_head = q_b + (long long)h * 64;
    float* __restrict__ o_out        = tb_batch + (long long)b * (n_attn_heads * 64) + (long long)h * 64;

    extern __shared__ float smem[];
    float* __restrict__ q_sh = smem;
    float* __restrict__ O_sh = q_sh + (64 + (64 >> 5) + 1);

    const float inv_sqrt_d = rsqrtf(64.0f);
    for (int j = lane; j < 64; j += warpSize) {
        q_sh[PAD32_INDEX(j)] = q_head[j] * inv_sqrt_d;
        O_sh[PAD32_INDEX(j)] = 0.0f;
    }
    wave_sync();

    __shared__ float m_shared, l_shared, scale_old, scale_new, s_curr;
    if (lane == 0) { m_shared = -std::numeric_limits<float>::infinity(); l_shared = 0.0f; }
    wave_sync();

    // KV addressing (same as regular kernel)
    const long long kv_base = d_layer_kv_off[layer_idx];
    const int       cap     = d_layer_kv_cap[layer_idx];
    const int       local   = d_layer_is_local[layer_idx];
    const bool      cap_pow2= is_pow2_int(cap);

    const int  gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + (long long)gb * (long long)cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = (const char*)k_cache_base + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = (const char*)v_cache_base + batch_off_elems * elem_size;

    const bool local_and_nomask = (local != 0) && !USE_MASK && (sliding_window > 0) && ((layer_idx & 1) == 0);
    const int t_window_start = local_and_nomask ? max(0, pos_b - sliding_window + 1) : 0;
    const int t_start        = local_and_nomask ? t_window_start : 0;
    const int t_end          = Lb;

    const long long head_off_bytes = ((long long)g * 64) * elem_size;
    const long long step_bytes     = ((long long)kv_dim) * elem_size;

    // With head_dim==64, each lane handles 1 element with 4x unroll for ILP
    for (int t = t_start; t < t_end; ++t) {
        int rt = (local != 0) ? (cap_pow2 ? (t & (cap - 1)) : (t % cap)) : t;

        const char* __restrict__ kp_bytes = k_layer_base + (long long)rt * step_bytes + head_off_bytes;
        const char* __restrict__ vp_bytes = v_layer_base + (long long)rt * step_bytes + head_off_bytes;

        float partial = 0.0f;

        if constexpr (std::is_same<KV_T,float>::value) {
            const float* __restrict__ k_head = reinterpret_cast<const float*>(kp_bytes);
            // Each lane processes 1 element with simple prefetch pattern
            if (lane < 64) {
                float k_val = ld_global_cond(k_head + lane, NT);
                partial = q_sh[PAD32_INDEX(lane)] * k_val;
            }
        } else {
            // BF16 path: head_dim=64, pairs=32 u32 words, one pair per lane
            if (lane < 32) {
                if (is_aligned_4(kp_bytes)) {
                    const uint32_t* __restrict__ k2 = reinterpret_cast<const uint32_t*>(kp_bytes);
                    uint32_t w = ld_global_u32_cond(k2 + lane, NT);
                    float k0, k1;
                    unpack_bf16x2(w, k0, k1);
                    int j = lane << 1;
                    partial += q_sh[PAD32_INDEX(j+0)] * k0 + q_sh[PAD32_INDEX(j+1)] * k1;
                } else {
                    const __hip_bfloat16* __restrict__ k16 = reinterpret_cast<const __hip_bfloat16*>(kp_bytes);
                    int j = lane << 1;
                    __hip_bfloat16 lo = k16[j+0], hi = k16[j+1];
                    partial += q_sh[PAD32_INDEX(j+0)] * bf16_to_f32_intr(lo) + q_sh[PAD32_INDEX(j+1)] * bf16_to_f32_intr(hi);
                }
            }
        }

        partial = wave_reduce_sum(partial);
        if (lane == 0) {
            float s = partial;
            if constexpr (USE_MASK) {
                s += mask[(long long)pos_b * seq_len + t];
            }
            s_curr = s;
        }
        wave_sync();

        if (!isfinite(s_curr)) continue;

        if (lane == 0) {
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, s_curr);
            const float alpha  = __expf(m_prev - m_new);
            const float p      = __expf(s_curr - m_new);
            const float l_new  = alpha * l_prev + p;

            scale_old = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new = (l_new > 0.0f) ? p / l_new : 0.0f;

            m_shared = m_new;
            l_shared = l_new;
        }
        wave_sync();

        // V update with same ILP pattern
        if constexpr (std::is_same<KV_T,float>::value) {
            const float* __restrict__ v_head = reinterpret_cast<const float*>(vp_bytes);
            if (lane < 64) {
                float v_val = ld_global_cond(v_head + lane, NT);
                O_sh[PAD32_INDEX(lane)] = scale_old * O_sh[PAD32_INDEX(lane)] + scale_new * v_val;
            }
        } else {
            // BF16 path: head_dim=64, pairs=32 u32 words, one pair per lane
            if (lane < 32) {
                if (is_aligned_4(vp_bytes)) {
                    const uint32_t* __restrict__ v2 = reinterpret_cast<const uint32_t*>(vp_bytes);
                    uint32_t w = ld_global_u32_cond(v2 + lane, NT);
                    float v0, v1;
                    unpack_bf16x2(w, v0, v1);
                    int j = lane << 1;
                    O_sh[PAD32_INDEX(j+0)] = scale_old * O_sh[PAD32_INDEX(j+0)] + scale_new * v0;
                    O_sh[PAD32_INDEX(j+1)] = scale_old * O_sh[PAD32_INDEX(j+1)] + scale_new * v1;
                } else {
                    const __hip_bfloat16* __restrict__ v16 = reinterpret_cast<const __hip_bfloat16*>(vp_bytes);
                    int j = lane << 1;
                    float v0 = bf16_to_f32_intr(v16[j+0]);
                    float v1 = bf16_to_f32_intr(v16[j+1]);
                    O_sh[PAD32_INDEX(j+0)] = scale_old * O_sh[PAD32_INDEX(j+0)] + scale_new * v0;
                    O_sh[PAD32_INDEX(j+1)] = scale_old * O_sh[PAD32_INDEX(j+1)] + scale_new * v1;
                }
            }
        }
        wave_sync();
    }

    // Sink processing (same as regular kernel)
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
        }
    }

    store_O_from_padded(o_out, O_sh, 64);
}

template<typename KV_T, bool USE_MASK>
__global__ void fa_decode_full_typed(
    const float* __restrict__ q_batch,      // (B, H*D)
    const void*  __restrict__ k_cache_base, // byte addressable
    const void*  __restrict__ v_cache_base, // byte addressable
    const float* __restrict__ mask,         // (T,T) if USE_MASK
    const __hip_bfloat16* __restrict__ attn_sinks_half, // (H) or nullptr
    float* __restrict__ tb_batch,           // (B, H*D)  [OUTPUT]
    // sizes & layout
    int B, int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    // positions, batch mapping, kv layer info
    const int* __restrict__ d_pos_per_token,  // (B)
    const int* __restrict__ d_batch_indices,  // (B)
    long long B_stride,
    const long long* __restrict__ d_layer_kv_off,
    const int*       __restrict__ d_layer_kv_cap,
    const int*       __restrict__ d_layer_is_local
){
    // grid: (h over x, b over y). One Wave64 per block.
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;           // 0..63 (we launch 64 threads)

    if (h >= n_attn_heads || b >= B) return;

    // Head group index (GQA/MQA)
    const int g = h / kv_mul;

    // Token position for this request
    const int pos_b = d_pos_per_token[b];
    const int Lb = pos_b + 1;

    // Compute NT policy once for this kernel instance
    const bool NT = use_nt_for_kv(kv_mul, B, n_attn_heads, head_dim, Lb);

    // ---- pointers for Q and output ----
    const float* __restrict__ q_b    = q_batch + (long long)b * (n_attn_heads * head_dim);
    const float* __restrict__ q_head = q_b + (long long)h * head_dim;
    float* __restrict__ o_out        = tb_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;

    // ---- shared memory for q and O with padding ----
    extern __shared__ float smem[];
    float* __restrict__ q_sh = smem;                                      // [padded_size]
    float* __restrict__ O_sh = q_sh + (head_dim + (head_dim >> 5) + 1);   // [padded_size]

    // Scale Q by 1/sqrt(d) and zero O with padded addressing
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int j = lane; j < head_dim; j += warpSize) {
        q_sh[PAD32_INDEX(j)] = q_head[j] * inv_sqrt_d;
        O_sh[PAD32_INDEX(j)] = 0.0f;
    }
    wave_sync();

    // ---- shared scalars for online softmax ----
    __shared__ float m_shared, l_shared, scale_old, scale_new, s_curr;
    if (lane == 0) { m_shared = -std::numeric_limits<float>::infinity(); l_shared = 0.0f; }
    wave_sync();

    // ---- KV layer addressing ----
    const long long kv_base = d_layer_kv_off[layer_idx];
    const int       cap     = d_layer_kv_cap[layer_idx];
    const int       local   = d_layer_is_local[layer_idx];
    const bool      cap_pow2= is_pow2_int(cap);

    const int  gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + (long long)gb * (long long)cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = (const char*)k_cache_base + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = (const char*)v_cache_base + batch_off_elems * elem_size;

    // Local sliding window bounds (for even layers if you use local KV)
    const bool local_and_nomask = (local != 0) && !USE_MASK && (sliding_window > 0) && ((layer_idx & 1) == 0);
    const int t_window_start = local_and_nomask ? max(0, pos_b - sliding_window + 1) : 0;
    const int t_start        = local_and_nomask ? t_window_start : 0;
    const int t_end          = Lb;

    // Per-head base offsets and step in BYTES
    const long long head_off_bytes = ((long long)g * head_dim) * elem_size;
    const long long step_bytes     = ((long long)kv_dim)      * elem_size;

    // One pass (FlashAttention online softmax).
    for (int t = t_start; t < t_end; ++t) {
        int rt;
        if (local != 0) {
            rt = cap_pow2 ? (t & (cap - 1)) : (t % cap);
        } else {
            rt = t;
        }

        const char* __restrict__ kp_bytes = k_layer_base + (long long)rt * step_bytes + head_off_bytes;
        const char* __restrict__ vp_bytes = v_layer_base + (long long)rt * step_bytes + head_off_bytes;

        const KV_T* __restrict__ k_head = reinterpret_cast<const KV_T*>(kp_bytes);
        const KV_T* __restrict__ v_head = reinterpret_cast<const KV_T*>(vp_bytes);

        // ---- dot(q, k_t) with vectorized streaming loads ----
        float partial = 0.0f;

        if constexpr (std::is_same<KV_T,float>::value) {
            // FP32 path: float4 vectorization with conditional NT and alignment safety
            int vecN = head_dim >> 2, rem = head_dim & 3;

            if (is_aligned_16(k_head)) {
                // Aligned fast path - load float4 as individual floats for NT support
                const float* __restrict__ k_f = reinterpret_cast<const float*>(k_head);
                for (int i4 = lane; i4 < vecN; i4 += warpSize) {
                    int j = (i4 << 2);
                    float k0 = ld_global_cond(k_f + j + 0, NT);
                    float k1 = ld_global_cond(k_f + j + 1, NT);
                    float k2 = ld_global_cond(k_f + j + 2, NT);
                    float k3 = ld_global_cond(k_f + j + 3, NT);
                    partial += q_sh[PAD32_INDEX(j+0)]*k0 + q_sh[PAD32_INDEX(j+1)]*k1 +
                              q_sh[PAD32_INDEX(j+2)]*k2 + q_sh[PAD32_INDEX(j+3)]*k3;
                }
            } else {
                // Unaligned fallback - scalar loads but still vectorized compute
                for (int i4 = lane; i4 < vecN; i4 += warpSize) {
                    int j = (i4 << 2);
                    float k0 = ld_global_cond(reinterpret_cast<const float*>(k_head) + j + 0, NT);
                    float k1 = ld_global_cond(reinterpret_cast<const float*>(k_head) + j + 1, NT);
                    float k2 = ld_global_cond(reinterpret_cast<const float*>(k_head) + j + 2, NT);
                    float k3 = ld_global_cond(reinterpret_cast<const float*>(k_head) + j + 3, NT);
                    partial += q_sh[PAD32_INDEX(j+0)]*k0 + q_sh[PAD32_INDEX(j+1)]*k1 +
                              q_sh[PAD32_INDEX(j+2)]*k2 + q_sh[PAD32_INDEX(j+3)]*k3;
                }
            }
            for (int j = (vecN<<2) + lane; j < (vecN<<2)+rem; j += warpSize) {
                float kj = ld_global_cond(reinterpret_cast<const float*>(k_head) + j, NT);
                partial += q_sh[PAD32_INDEX(j)] * kj;
            }
        } else {
            // BF16 path: load 2×bf16 as u32 (packed) with conditional NT
            const uint32_t* __restrict__ k2 = reinterpret_cast<const uint32_t*>(k_head);
            int pairs = head_dim >> 1, rem = head_dim & 1;

            for (int p = lane; p < pairs; p += warpSize) {
                uint32_t w = ld_global_u32_cond(k2 + p, NT);
                __hip_bfloat16 lo = raw_to_bf16((uint16_t)(w & 0xFFFF));
                __hip_bfloat16 hi = raw_to_bf16((uint16_t)(w >> 16));
                int j = (p << 1);
                partial += q_sh[PAD32_INDEX(j+0)]*bf16_to_f32_intr(lo) + q_sh[PAD32_INDEX(j+1)]*bf16_to_f32_intr(hi);
            }
            if (rem) {
                int j = (pairs<<1) + lane;
                if (j < head_dim) {
                    __hip_bfloat16 kb = ld_global_cond(reinterpret_cast<const __hip_bfloat16*>(k_head) + j, NT);
                    partial += q_sh[PAD32_INDEX(j)] * bf16_to_f32_intr(kb);
                }
            }
        }

        partial = wave_reduce_sum(partial);
        if (lane == 0) {
            float s = partial;
            if constexpr (USE_MASK) {
                s += mask[(long long)pos_b * seq_len + t];
            }
            s_curr = s;
        }
        wave_sync();

        // Skip update if masked to -inf
        if (!isfinite(s_curr)) continue;

        if (lane == 0) {
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, s_curr);
            const float alpha  = __expf(m_prev - m_new);
            const float p      = __expf(s_curr - m_new);
            const float l_new  = alpha * l_prev + p;

            scale_old = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new = (l_new > 0.0f) ? p / l_new : 0.0f;

            m_shared = m_new;
            l_shared = l_new;
        }
        wave_sync();

        // ---- O update with vectorized V loads (streaming) ----
        if constexpr (std::is_same<KV_T,float>::value) {
            int vecN = head_dim >> 2, rem = head_dim & 3;

            if (is_aligned_16(v_head)) {
                // Aligned fast path - load float4 as individual floats for NT support
                const float* __restrict__ v_f = reinterpret_cast<const float*>(v_head);
                for (int i4 = lane; i4 < vecN; i4 += warpSize) {
                    int j = (i4 << 2);
                    float v0 = ld_global_cond(v_f + j + 0, NT);
                    float v1 = ld_global_cond(v_f + j + 1, NT);
                    float v2 = ld_global_cond(v_f + j + 2, NT);
                    float v3 = ld_global_cond(v_f + j + 3, NT);
                    float o0 = scale_old * O_sh[PAD32_INDEX(j+0)] + scale_new * v0;
                    float o1 = scale_old * O_sh[PAD32_INDEX(j+1)] + scale_new * v1;
                    float o2 = scale_old * O_sh[PAD32_INDEX(j+2)] + scale_new * v2;
                    float o3 = scale_old * O_sh[PAD32_INDEX(j+3)] + scale_new * v3;
                    O_sh[PAD32_INDEX(j+0)]=o0; O_sh[PAD32_INDEX(j+1)]=o1;
                    O_sh[PAD32_INDEX(j+2)]=o2; O_sh[PAD32_INDEX(j+3)]=o3;
                }
            } else {
                // Unaligned fallback
                for (int i4 = lane; i4 < vecN; i4 += warpSize) {
                    int j = (i4 << 2);
                    float v0 = ld_global_cond(reinterpret_cast<const float*>(v_head) + j + 0, NT);
                    float v1 = ld_global_cond(reinterpret_cast<const float*>(v_head) + j + 1, NT);
                    float v2 = ld_global_cond(reinterpret_cast<const float*>(v_head) + j + 2, NT);
                    float v3 = ld_global_cond(reinterpret_cast<const float*>(v_head) + j + 3, NT);
                    O_sh[PAD32_INDEX(j+0)] = scale_old * O_sh[PAD32_INDEX(j+0)] + scale_new * v0;
                    O_sh[PAD32_INDEX(j+1)] = scale_old * O_sh[PAD32_INDEX(j+1)] + scale_new * v1;
                    O_sh[PAD32_INDEX(j+2)] = scale_old * O_sh[PAD32_INDEX(j+2)] + scale_new * v2;
                    O_sh[PAD32_INDEX(j+3)] = scale_old * O_sh[PAD32_INDEX(j+3)] + scale_new * v3;
                }
            }
            for (int j = (vecN<<2) + lane; j < (vecN<<2)+rem; j += warpSize) {
                float vj = ld_global_cond(reinterpret_cast<const float*>(v_head) + j, NT);
                O_sh[PAD32_INDEX(j)] = scale_old * O_sh[PAD32_INDEX(j)] + scale_new * vj;
            }
        } else {
            const uint32_t* __restrict__ v2 = reinterpret_cast<const uint32_t*>(v_head);
            int pairs = head_dim >> 1, rem = head_dim & 1;

            for (int p = lane; p < pairs; p += warpSize) {
                uint32_t w = ld_global_u32_cond(v2 + p, NT);
                __hip_bfloat16 lo = raw_to_bf16((uint16_t)(w & 0xFFFF));
                __hip_bfloat16 hi = raw_to_bf16((uint16_t)(w >> 16));
                int j = (p << 1);
                float v0 = bf16_to_f32_intr(lo);
                float v1 = bf16_to_f32_intr(hi);
                O_sh[PAD32_INDEX(j+0)] = scale_old * O_sh[PAD32_INDEX(j+0)] + scale_new * v0;
                O_sh[PAD32_INDEX(j+1)] = scale_old * O_sh[PAD32_INDEX(j+1)] + scale_new * v1;
            }
            if (rem) {
                int j = (pairs<<1) + lane;
                if (j < head_dim) {
                    __hip_bfloat16 vb = ld_global_cond(reinterpret_cast<const __hip_bfloat16*>(v_head) + j, NT);
                    float vj = bf16_to_f32_intr(vb);
                    O_sh[PAD32_INDEX(j)] = scale_old * O_sh[PAD32_INDEX(j)] + scale_new * vj;
                }
            }
        }
        wave_sync();
    } // end t

    if (attn_sinks_half != nullptr) {
        float sink_s = __bfloat162float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m_shared; const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, sink_s);
            const float a = __expf(m_prev - m_new) * l_prev;
            const float b = __expf(sink_s - m_new);
            const float denom = a + b;
            const float renorm= (denom > 0.0f) ? (a / denom) : 1.0f;

            for (int j = lane; j < head_dim; j += warpSize)
                O_sh[PAD32_INDEX(j)] *= renorm;
            wave_sync();
        }
    }

    // Use alignment-safe padded store
    store_O_from_padded(o_out, O_sh, head_dim);
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
    if (B <= 0 || n_attn_heads <= 0 || head_dim <= 0) return;

    const int tpb = 64; // Wave64 on MI250

    // bytes needed for q_sh and o_sh as float32 with padding
    auto padded_floats = [](int n){ return n + (n >> 5) + 1; }; // +1 safety
    size_t q_bytes = padded_floats(head_dim) * sizeof(float);
    size_t o_bytes = padded_floats(head_dim) * sizeof(float);
    size_t shmem = q_bytes + o_bytes;

    const __hip_bfloat16* attn_sinks_head =
        (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;

    // Decide compile-time mask specialization for this layer (even layers with SWA)
    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0) && (mask != nullptr);

    dim3 grid(n_attn_heads, B), block(tpb);

    // Use specialized kernel for head_dim==64
    if (head_dim == 64) {
        if (kv_cache_is_fp16) {
            if (use_mask) {
                hipLaunchKernelGGL((fa_decode_full_typed_head64<__hip_bfloat16, true>),  grid, block, shmem, stream,
                    q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                    B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                    d_pos_per_token, d_batch_indices, (long long)B_stride,
                    d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
            } else {
                hipLaunchKernelGGL((fa_decode_full_typed_head64<__hip_bfloat16, false>), grid, block, shmem, stream,
                    q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                    B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                    d_pos_per_token, d_batch_indices, (long long)B_stride,
                    d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
            }
        } else {
            if (use_mask) {
                hipLaunchKernelGGL((fa_decode_full_typed_head64<float, true>),  grid, block, shmem, stream,
                    q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                    B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                    d_pos_per_token, d_batch_indices, (long long)B_stride,
                    d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
            } else {
                hipLaunchKernelGGL((fa_decode_full_typed_head64<float, false>), grid, block, shmem, stream,
                    q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                    B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                    d_pos_per_token, d_batch_indices, (long long)B_stride,
                    d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
            }
        }
    } else if (kv_cache_is_fp16) {
        if (use_mask) {
            hipLaunchKernelGGL((fa_decode_full_typed<__hip_bfloat16, true>),  grid, block, shmem, stream,
                q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        } else {
            hipLaunchKernelGGL((fa_decode_full_typed<__hip_bfloat16, false>), grid, block, shmem, stream,
                q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        }
    } else {
        if (use_mask) {
            hipLaunchKernelGGL((fa_decode_full_typed<float, true>),  grid, block, shmem, stream,
                q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        } else {
            hipLaunchKernelGGL((fa_decode_full_typed<float, false>), grid, block, shmem, stream,
                q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, (long long)B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        }
    }

    CHECK_HIP(hipGetLastError());
}
