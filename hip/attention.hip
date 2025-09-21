#pragma once
#include "BLAS.hip"
#include <cmath>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdint.h>
#include <type_traits>
#include <vector>

#if defined(__HIP_DEVICE_COMPILE__)
template <typename T>
__device__ __forceinline__ T nt_load(const T* ptr) {
    using base_t = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<base_t, float> || std::is_same_v<base_t, double> ||
                  std::is_same_v<base_t, int32_t> || std::is_same_v<base_t, uint32_t> ||
                  std::is_same_v<base_t, int64_t> || std::is_same_v<base_t, uint64_t>) {
        return __builtin_nontemporal_load(ptr);
    } else {
        return *ptr;
    }
}
template <typename T>
__device__ __forceinline__ void nt_store(T* ptr, T value) {
    using base_t = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<base_t, float> || std::is_same_v<base_t, double> ||
                  std::is_same_v<base_t, int32_t> || std::is_same_v<base_t, uint32_t> ||
                  std::is_same_v<base_t, int64_t> || std::is_same_v<base_t, uint64_t>) {
        __builtin_nontemporal_store(value, ptr);
    } else {
        *ptr = value;
    }
}
#else
template <typename T>
__device__ __forceinline__ T nt_load(const T* ptr) {
    return *ptr;
}
template <typename T>
__device__ __forceinline__ void nt_store(T* ptr, T v) {
    *ptr = v;
}
#endif

static __device__ __forceinline__ float wave_reduce_sum(float v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}
template <typename T>
static __device__ __forceinline__ T wave_bcast(T x, int srcLane = 0) {
    return __shfl(x, srcLane);
}

static __device__ __forceinline__ bool is_pow2_int(int x) { return (x & (x - 1)) == 0; }

static inline __device__ float bf16_to_f32_intr(__hip_bfloat16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ __hip_bfloat16 raw_to_bf16(uint16_t bits) {
    union {
        uint16_t u;
        __hip_bfloat16 b;
    } conv;
    conv.u = bits;
    return conv.b;
}

#ifndef MAX_TPOS
#define MAX_TPOS 8
#endif
#ifndef T_TILE
#define T_TILE 256
#endif

template <typename KV_T, bool USE_MASK>
__global__ void
fa_decode_reg_tiled_single(const float* __restrict__ q_batch,                  // (B, H*D)
                           const void* __restrict__ k_cache_base,              // byte addressable
                           const void* __restrict__ v_cache_base,              // byte addressable
                           const float* __restrict__ mask,                     // (T,T) if USE_MASK
                           const __hip_bfloat16* __restrict__ attn_sinks_half, // (H) or nullptr
                           float* __restrict__ tb_batch,                       // (B, H*D)  [OUTPUT]
                           int B, int seq_len, int head_dim, int kv_dim, int kv_mul,
                           int sliding_window, int layer_idx, int n_attn_heads,
                           const int* __restrict__ d_pos_per_token, // (B)
                           const int* __restrict__ d_batch_indices, // (B)
                           const long long* __restrict__ d_layer_kv_off,
                           const int* __restrict__ d_layer_kv_cap,
                           const int* __restrict__ d_layer_is_local) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;

    if (h >= n_attn_heads || b >= B)
        return;

    const int g = h / kv_mul;
    const int pos_b = d_pos_per_token[b];
    const int Lb = pos_b + 1;
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    const float* __restrict__ q_head =
        q_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;
    float* __restrict__ o_out =
        tb_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;

    const int W = warpSize;
    const int POS = (head_dim + W - 1) / W;
    float q_reg[MAX_TPOS];
    float o_reg[MAX_TPOS];

#pragma unroll
    for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
        int j = tpos * W + lane;
        float qj = (tpos < POS && j < head_dim) ? q_head[j] * inv_sqrt_d : 0.0f;
        q_reg[tpos] = qj;
        o_reg[tpos] = 0.0f;
    }

    const long long kv_base = d_layer_kv_off[layer_idx];
    const int cap = d_layer_kv_cap[layer_idx];
    const int local = d_layer_is_local[layer_idx];
    const bool cap_pow2 = is_pow2_int(cap);

    const int gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + (long long)gb * (long long)cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = (const char*)k_cache_base + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = (const char*)v_cache_base + batch_off_elems * elem_size;

    const bool local_and_nomask =
        (local != 0) && !USE_MASK && (sliding_window > 0) && ((layer_idx & 1) == 0);
    const int t_window_start = local_and_nomask ? max(0, pos_b - sliding_window + 1) : 0;
    const int t_start = local_and_nomask ? t_window_start : 0;
    const int t_end = Lb;

    const long long head_off_bytes = ((long long)g * head_dim) * elem_size;
    const long long step_bytes = ((long long)kv_dim) * elem_size;

    float m_shared = -INFINITY;
    float l_shared = 0.0f;

    for (int t = t_start; t < t_end; ++t) {
        const int rt = (local != 0) ? (cap_pow2 ? (t & (cap - 1)) : (t % cap)) : t;
        const char* __restrict__ kp_bytes =
            k_layer_base + (long long)rt * step_bytes + head_off_bytes;
        const char* __restrict__ vp_bytes =
            v_layer_base + (long long)rt * step_bytes + head_off_bytes;
        const KV_T* __restrict__ k_head = reinterpret_cast<const KV_T*>(kp_bytes);
        const KV_T* __restrict__ v_head = reinterpret_cast<const KV_T*>(vp_bytes);

        float partial = 0.0f;
#pragma unroll
        for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
            int j = tpos * W + lane;
            if (tpos < POS && j < head_dim) {
                float kj = std::is_same<KV_T, float>::value
                               ? nt_load(reinterpret_cast<const float*>(k_head) + j)
                               : bf16_to_f32_intr(
                                     nt_load(reinterpret_cast<const __hip_bfloat16*>(k_head) + j));
                partial += q_reg[tpos] * kj;
            }
        }
        float s = wave_reduce_sum(partial);
        if constexpr (USE_MASK) {
            if (lane == 0)
                s += mask[(long long)pos_b * seq_len + t];
        }
        s = wave_bcast(s, 0);
        if (!isfinite(s))
            continue;

        float scale_old, scale_new;
        if (lane == 0) {
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new = fmaxf(m_prev, s);
            const float alpha = __expf(m_prev - m_new);
            const float p = __expf(s - m_new);
            const float l_new = alpha * l_prev + p;
            scale_old = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new = (l_new > 0.0f) ? p / l_new : 0.0f;
            m_shared = m_new;
            l_shared = l_new;
        }
        scale_old = wave_bcast(scale_old, 0);
        scale_new = wave_bcast(scale_new, 0);

#pragma unroll
        for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
            int j = tpos * W + lane;
            if (tpos < POS && j < head_dim) {
                float vj = std::is_same<KV_T, float>::value
                               ? nt_load(reinterpret_cast<const float*>(v_head) + j)
                               : bf16_to_f32_intr(
                                     nt_load(reinterpret_cast<const __hip_bfloat16*>(v_head) + j));
                o_reg[tpos] = scale_old * o_reg[tpos] + scale_new * vj;
            }
        }
    }

    if (attn_sinks_half != nullptr) {
        float sink_s = __bfloat162float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m_shared, l_prev = l_shared;
            const float m_new = fmaxf(m_prev, sink_s);
            const float a = __expf(m_prev - m_new) * l_prev;
            const float b = __expf(sink_s - m_new);
            const float denom = a + b;
            const float r = (denom > 0.0f) ? (a / denom) : 1.0f;
            float ren = wave_bcast((lane == 0) ? r : 0.0f, 0);
#pragma unroll
            for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
                int j = tpos * warpSize + lane;
                if (tpos < POS && j < head_dim)
                    o_reg[tpos] *= ren;
            }
        }
    }

#pragma unroll
    for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
        int j = tpos * warpSize + lane;
        if (tpos < POS && j < head_dim)
            nt_store(o_out + j, o_reg[tpos]);
    }
}

template <typename KV_T, bool USE_MASK>
__global__ void fa_decode_reg_tiled_split_stage1(
    const float* __restrict__ q_batch, const void* __restrict__ k_cache_base,
    const void* __restrict__ v_cache_base, const float* __restrict__ mask,
    float* __restrict__ O_partials, // [B*H*Smax, head_dim]
    float* __restrict__ m_partials, // [B*H*Smax]
    float* __restrict__ l_partials, // [B*H*Smax]
    int B, int seq_len, int head_dim, int kv_dim, int kv_mul, int sliding_window, int layer_idx,
    int n_attn_heads,
    int Smax, // ceil(seq_len / T_TILE)
    const int* __restrict__ d_pos_per_token, const int* __restrict__ d_batch_indices,
    const long long* __restrict__ d_layer_kv_off, const int* __restrict__ d_layer_kv_cap,
    const int* __restrict__ d_layer_is_local) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int sidx = blockIdx.z; // split index
    const int lane = threadIdx.x;
    if (h >= n_attn_heads || b >= B || sidx >= Smax)
        return;

    const int g = h / kv_mul;
    const int pos_b = d_pos_per_token[b];
    const int Lb = pos_b + 1;
    const float inv_sqrt_d = rsqrtf((float)head_dim);

    const int W = warpSize;
    const int POS = (head_dim + W - 1) / W;

    const float* __restrict__ q_head =
        q_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;

    const long long kv_base = d_layer_kv_off[layer_idx];
    const int cap = d_layer_kv_cap[layer_idx];
    const int local = d_layer_is_local[layer_idx];
    const bool cap_pow2 = is_pow2_int(cap);

    const int gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + (long long)gb * (long long)cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = (const char*)k_cache_base + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = (const char*)v_cache_base + batch_off_elems * elem_size;

    const bool local_and_nomask =
        (local != 0) && !USE_MASK && (sliding_window > 0) && ((layer_idx & 1) == 0);
    const int t_window_start = local_and_nomask ? max(0, pos_b - sliding_window + 1) : 0;
    const int t_start_full = local_and_nomask ? t_window_start : 0;
    const int t_end_full = Lb;

    const int t0 = t_start_full + sidx * T_TILE;
    if (t0 >= t_end_full)
        return;
    const int t1 = min(t0 + T_TILE, t_end_full);

    float q_reg[MAX_TPOS], o_reg[MAX_TPOS];
#pragma unroll
    for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
        int j = tpos * W + lane;
        float qj = (tpos < POS && j < head_dim) ? q_head[j] * inv_sqrt_d : 0.0f;
        q_reg[tpos] = qj;
        o_reg[tpos] = 0.0f;
    }

    const long long head_off_bytes = ((long long)g * head_dim) * elem_size;
    const long long step_bytes = ((long long)kv_dim) * elem_size;

    float m_local = -INFINITY;
    float l_local = 0.0f;

    for (int t = t0; t < t1; ++t) {
        const int rt = (local != 0) ? (cap_pow2 ? (t & (cap - 1)) : (t % cap)) : t;
        const char* __restrict__ kp_bytes =
            k_layer_base + (long long)rt * step_bytes + head_off_bytes;
        const char* __restrict__ vp_bytes =
            v_layer_base + (long long)rt * step_bytes + head_off_bytes;
        const KV_T* __restrict__ k_head = reinterpret_cast<const KV_T*>(kp_bytes);
        const KV_T* __restrict__ v_head = reinterpret_cast<const KV_T*>(vp_bytes);

        float partial = 0.0f;
#pragma unroll
        for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
            int j = tpos * W + lane;
            if (tpos < POS && j < head_dim) {
                float kj = std::is_same<KV_T, float>::value
                               ? nt_load(reinterpret_cast<const float*>(k_head) + j)
                               : bf16_to_f32_intr(
                                     nt_load(reinterpret_cast<const __hip_bfloat16*>(k_head) + j));
                partial += q_reg[tpos] * kj;
            }
        }
        float s = wave_reduce_sum(partial);
        if constexpr (USE_MASK) {
            if (lane == 0)
                s += mask[(long long)pos_b * seq_len + t];
        }
        s = wave_bcast(s, 0);
        if (!isfinite(s))
            continue;

        float scale_old, scale_new;
        if (lane == 0) {
            const float m_prev = m_local;
            const float l_prev = l_local;
            const float m_new = fmaxf(m_prev, s);
            const float alpha = __expf(m_prev - m_new);
            const float p = __expf(s - m_new);
            const float l_new = alpha * l_prev + p;
            scale_old = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new = (l_new > 0.0f) ? p / l_new : 0.0f;
            m_local = m_new;
            l_local = l_new;
        }
        scale_old = wave_bcast(scale_old, 0);
        scale_new = wave_bcast(scale_new, 0);

#pragma unroll
        for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
            int j = tpos * W + lane;
            if (tpos < POS && j < head_dim) {
                float vj = std::is_same<KV_T, float>::value
                               ? nt_load(reinterpret_cast<const float*>(v_head) + j)
                               : bf16_to_f32_intr(
                                     nt_load(reinterpret_cast<const __hip_bfloat16*>(v_head) + j));
                o_reg[tpos] = scale_old * o_reg[tpos] + scale_new * vj;
            }
        }
    }

    const long long base_idx = ((long long)b * n_attn_heads + h) * Smax + sidx;
    float* O_dst = O_partials + base_idx * (long long)head_dim;
    if (lane == 0) {
        m_partials[base_idx] = m_local;
        l_partials[base_idx] = l_local;
    }
#pragma unroll
    for (int tpos = 0; tpos < MAX_TPOS; ++tpos) {
        int j = tpos * W + lane;
        if (tpos < POS && j < head_dim)
            nt_store(O_dst + j, o_reg[tpos]);
    }
}

__global__ void
fa_decode_split_merge(const float* __restrict__ O_partials,               // [B*H*Smax, D]
                      const float* __restrict__ m_partials,               // [B*H*Smax]
                      const float* __restrict__ l_partials,               // [B*H*Smax]
                      const __hip_bfloat16* __restrict__ attn_sinks_half, // (H) or nullptr
                      float* __restrict__ tb_batch,                       // (B, H*D)
                      int B, int head_dim, int n_attn_heads, int Smax, int b,
                      int layer_idx) // pass b/layer for sink indexing compatibility
{
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    if (h >= n_attn_heads)
        return;

    const int W = warpSize;
    const int POS = (head_dim + W - 1) / W;

    float m_global = -INFINITY;
    for (int s = lane; s < Smax; s += W) {
        const long long idx = ((long long)b * n_attn_heads + h) * Smax + s;
        float m_i = m_partials[idx];
        m_global = fmaxf(m_global, m_i);
    }
    m_global = wave_reduce_sum(
        m_global);
    float mg = m_global;
    for (int off = W >> 1; off > 0; off >>= 1) {
        float other = __shfl_down(mg, off);
        mg = fmaxf(mg, other);
    }
    m_global = wave_bcast(mg, 0);

    float L_partial = 0.0f;
    for (int s = lane; s < Smax; s += W) {
        const long long idx = ((long long)b * n_attn_heads + h) * Smax + s;
        float m_i = m_partials[idx];
        float l_i = l_partials[idx];
        if (isfinite(m_i) && l_i > 0.0f) {
            L_partial += __expf(m_i - m_global) * l_i;
        }
    }
    for (int off = W >> 1; off > 0; off >>= 1)
        L_partial += __shfl_down(L_partial, off);
    float L = wave_bcast(L_partial, 0);
    L = L > 0.0f ? L : 1.0f;

    float* __restrict__ o_out =
        tb_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;

    for (int tpos = 0; tpos < POS; ++tpos) {
        int j = tpos * W + lane;
        float Oj = 0.0f;
        if (j < head_dim) {
            for (int s = 0; s < Smax; ++s) {
                const long long base = ((long long)b * n_attn_heads + h) * Smax + s;
                const float m_i = m_partials[base];
                const float l_i = l_partials[base];
                if (!isfinite(m_i) || l_i <= 0.0f)
                    continue;
                const float w = __expf(m_i - m_global) * l_i;
                Oj += w * O_partials[base * (long long)head_dim + j];
            }
            Oj /= L;
            nt_store(o_out + j, Oj);
        }
    }

    if (attn_sinks_half != nullptr) {
        float sink_s = __bfloat162float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_new = fmaxf(m_global, sink_s);
            const float a = __expf(m_global - m_new) * L;
            const float b2 = __expf(sink_s - m_new);
            const float denom = a + b2;
            const float r = (denom > 0.0f) ? (a / denom) : 1.0f;
            float ren = wave_bcast((lane == 0) ? r : 0.0f, 0);

            for (int tpos = 0; tpos < POS; ++tpos) {
                int j = tpos * W + lane;
                if (j < head_dim)
                    o_out[j] *= ren;
            }
        }
    }
}

void fa(const float* q_batch,             // (B, H*D)
        const void* k_cache,              // base pointer
        const void* v_cache,              // base pointer
        const float* mask,                // (T, T)
        const __hip_bfloat16* attn_sinks, // (L, H) — pass attn_sinks + layer_idx*H
        float* tb_batch,                  // (B, H*D)
        int B, int seq_len, int head_dim, int kv_dim, int kv_mul, int sliding_window, int layer_idx,
        int n_attn_heads, int kv_cache_is_fp16,
        const int* d_pos_per_token, // (B)
        const int* d_batch_indices, // (B)
        long long /*B_stride*/,     // not needed
        int /*max_pos_in_batch*/,   // unused
        float* workspace_partial_O,
        float* workspace_partial_m, float* workspace_partial_l, hipStream_t stream,
        const long long* d_layer_kv_off, const int* d_layer_kv_cap, const int* d_layer_is_local) {
    if (B <= 0 || n_attn_heads <= 0 || head_dim <= 0)
        return;

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0) && (mask != nullptr);
    const __hip_bfloat16* attn_sinks_head =
        (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;

    const bool want_split = (seq_len >= T_TILE * 2);
    const bool have_ws = (workspace_partial_O && workspace_partial_m && workspace_partial_l);
    const bool do_split = want_split && have_ws;

    const int tpb = 64;

    if (do_split) {
        const int Smax = (seq_len + T_TILE - 1) / T_TILE;

        dim3 grid1(n_attn_heads, B, Smax), block1(tpb);

        if (kv_cache_is_fp16) {
            if (use_mask) {
                hipLaunchKernelGGL((fa_decode_reg_tiled_split_stage1<__hip_bfloat16, true>), grid1,
                                   block1, 0, stream, q_batch, k_cache, v_cache, mask,
                                   workspace_partial_O, workspace_partial_m, workspace_partial_l, B,
                                   seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                                   n_attn_heads, Smax, d_pos_per_token, d_batch_indices,
                                   d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
            } else {
                hipLaunchKernelGGL((fa_decode_reg_tiled_split_stage1<__hip_bfloat16, false>), grid1,
                                   block1, 0, stream, q_batch, k_cache, v_cache, nullptr,
                                   workspace_partial_O, workspace_partial_m, workspace_partial_l, B,
                                   seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                                   n_attn_heads, Smax, d_pos_per_token, d_batch_indices,
                                   d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
            }
        } else {
            if (use_mask) {
                hipLaunchKernelGGL((fa_decode_reg_tiled_split_stage1<float, true>), grid1, block1,
                                   0, stream, q_batch, k_cache, v_cache, mask, workspace_partial_O,
                                   workspace_partial_m, workspace_partial_l, B, seq_len, head_dim,
                                   kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads, Smax,
                                   d_pos_per_token, d_batch_indices, d_layer_kv_off, d_layer_kv_cap,
                                   d_layer_is_local);
            } else {
                hipLaunchKernelGGL((fa_decode_reg_tiled_split_stage1<float, false>), grid1, block1,
                                   0, stream, q_batch, k_cache, v_cache, nullptr,
                                   workspace_partial_O, workspace_partial_m, workspace_partial_l, B,
                                   seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                                   n_attn_heads, Smax, d_pos_per_token, d_batch_indices,
                                   d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
            }
        }
        CHECK_HIP(hipGetLastError());

        for (int b = 0; b < B; ++b) {
            dim3 grid2(n_attn_heads, 1, 1), block2(tpb);
            hipLaunchKernelGGL((fa_decode_split_merge), grid2, block2, 0, stream,
                               workspace_partial_O, workspace_partial_m, workspace_partial_l,
                               attn_sinks_head, tb_batch, B, head_dim, n_attn_heads, Smax, b,
                               layer_idx);
            CHECK_HIP(hipGetLastError());
        }
        return;
    }

    dim3 grid(n_attn_heads, B), block(tpb);
    if (kv_cache_is_fp16) {
        if (use_mask) {
            hipLaunchKernelGGL((fa_decode_reg_tiled_single<__hip_bfloat16, true>), grid, block, 0,
                               stream, q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                               B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                               n_attn_heads, d_pos_per_token, d_batch_indices, d_layer_kv_off,
                               d_layer_kv_cap, d_layer_is_local);
        } else {
            hipLaunchKernelGGL((fa_decode_reg_tiled_single<__hip_bfloat16, false>), grid, block, 0,
                               stream, q_batch, k_cache, v_cache, nullptr, attn_sinks_head,
                               tb_batch, B, seq_len, head_dim, kv_dim, kv_mul, sliding_window,
                               layer_idx, n_attn_heads, d_pos_per_token, d_batch_indices,
                               d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        }
    } else {
        if (use_mask) {
            hipLaunchKernelGGL((fa_decode_reg_tiled_single<float, true>), grid, block, 0, stream,
                               q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch, B,
                               seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                               n_attn_heads, d_pos_per_token, d_batch_indices, d_layer_kv_off,
                               d_layer_kv_cap, d_layer_is_local);
        } else {
            hipLaunchKernelGGL((fa_decode_reg_tiled_single<float, false>), grid, block, 0, stream,
                               q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch, B,
                               seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                               n_attn_heads, d_pos_per_token, d_batch_indices, d_layer_kv_off,
                               d_layer_kv_cap, d_layer_is_local);
        }
    }
    CHECK_HIP(hipGetLastError());
}
