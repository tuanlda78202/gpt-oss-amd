#pragma once
#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>

template<typename T>
__device__ __forceinline__ T ld_global_cond(const T* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                  std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
    }
#endif
    return *p;
}

__device__ __forceinline__ uint32_t ld_global_u32_cond(const uint32_t* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_nontemporal_load(p);
#endif
    return *p;
}

template<int WIDTH>
__device__ __forceinline__ float wave_reduce_sum_width(float v) {
    #pragma unroll
    for (int off = WIDTH >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off, WIDTH);
    return v;
}

static __device__ __forceinline__ void wave_sync() {
#if defined(__CUDA_ARCH__)
    __syncwarp();
#elif defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
#endif
}

static __device__ __forceinline__ bool is_pow2_int(int x){ return (x & (x-1)) == 0; }
template<bool POW2>
static __device__ __forceinline__ int ring_index(int t, int cap) {
    if constexpr (POW2) return t & (cap - 1);
    else                return (t >= cap) ? (t % cap) : t;
}

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

__device__ __forceinline__ void unpack_bf16x2(uint32_t packed, float& lo, float& hi) {
    __hip_bfloat16 bf_lo = raw_to_bf16((uint16_t)(packed & 0xFFFF));
    __hip_bfloat16 bf_hi = raw_to_bf16((uint16_t)(packed >> 16));
    lo = bf16_to_f32_intr(bf_lo);
    hi = bf16_to_f32_intr(bf_hi);
}

__device__ __forceinline__ bool is_aligned_4(const void* p) {
    return ((uintptr_t)p & 3) == 0;
}

// ======================================================================
// FA2 tiled decode kernel (Br=4) for head_dim==64
// - Processes four query heads (h0..h3) that share the same KV group.
// - Reuses K/V loads across all 4 rows; separate LSE per row.
// ======================================================================
template<typename KV_T, bool USE_MASK, int BcMax=64>
__global__ void fa2_decode_br4_head64(
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
    const int hp   = blockIdx.x;             // head-quad index
    const int b    = blockIdx.y;
    const int lane = threadIdx.x;            // wave64

    const int h0 = hp * 4;
    const int h1 = h0 + 1;
    const int h2 = h0 + 2;
    const int h3 = h0 + 3;
    if (b >= B || h3 >= n_attn_heads) return;

    const int g0 = h0 / kv_mul;
    const int g1 = h1 / kv_mul;
    const int g2 = h2 / kv_mul;
    const int g3 = h3 / kv_mul;
    if (!(g0 == g1 && g1 == g2 && g2 == g3)) return;

    const int pos_b = d_pos_per_token[b];
    const int Lb    = pos_b + 1;

    const int  local = d_layer_is_local[layer_idx];
    const bool local_nomask = (local != 0) && !USE_MASK && (sliding_window > 0) && ((layer_idx & 1) == 0);
    const int  t_window_start = local_nomask ? max(0, pos_b - sliding_window + 1) : 0;
    const int  t_start = local_nomask ? t_window_start : 0;
    const int  t_end   = Lb;

    const long long kv_base = d_layer_kv_off[layer_idx];
    const int       cap     = d_layer_kv_cap[layer_idx];
    const bool      cap_pow2= is_pow2_int(cap);

    const int  gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + (long long)gb * (long long)cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = (const char*)k_cache_base + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = (const char*)v_cache_base + batch_off_elems * elem_size;

    const long long head_off_bytes = ((long long)g0 * 64) * elem_size;
    const long long step_bytes     = ((long long)kv_dim) * elem_size;

    extern __shared__ float smem[];
    float* __restrict__ s0 = smem + 0 * BcMax;
    float* __restrict__ s1 = smem + 1 * BcMax;
    float* __restrict__ s2 = smem + 2 * BcMax;
    float* __restrict__ s3 = smem + 3 * BcMax;

    const float* __restrict__ q_b  = q_batch + (long long)b * (n_attn_heads * 64);
    const float* __restrict__ q0_p = q_b + (long long)h0 * 64;
    const float* __restrict__ q1_p = q_b + (long long)h1 * 64;
    const float* __restrict__ q2_p = q_b + (long long)h2 * 64;
    const float* __restrict__ q3_p = q_b + (long long)h3 * 64;
    float* __restrict__ o0_out = tb_batch + (long long)b * (n_attn_heads * 64) + (long long)h0 * 64;
    float* __restrict__ o1_out = tb_batch + (long long)b * (n_attn_heads * 64) + (long long)h1 * 64;
    float* __restrict__ o2_out = tb_batch + (long long)b * (n_attn_heads * 64) + (long long)h2 * 64;
    float* __restrict__ o3_out = tb_batch + (long long)b * (n_attn_heads * 64) + (long long)h3 * 64;

    const float inv_sqrt_d = rsqrtf(64.0f);
    constexpr int REDW = std::is_same<KV_T,__hip_bfloat16>::value ? 32 : 64;

    float q0r=0, q1r=0, q2r=0, q3r=0, o0r=0, o1r=0, o2r=0, o3r=0;
    float q0_0=0, q0_1=0, q1_0=0, q1_1=0, q2_0=0, q2_1=0, q3_0=0, q3_1=0;
    float oo0_0=0, oo0_1=0, oo1_0=0, oo1_1=0, oo2_0=0, oo2_1=0, oo3_0=0, oo3_1=0;
    if constexpr (std::is_same<KV_T,float>::value) {
        if (lane < 64) {
            q0r = q0_p[lane] * inv_sqrt_d;
            q1r = q1_p[lane] * inv_sqrt_d;
            q2r = q2_p[lane] * inv_sqrt_d;
            q3r = q3_p[lane] * inv_sqrt_d;
            o0r = o1r = o2r = o3r = 0.0f;
        }
    } else {
        if (lane < 32) {
            q0_0 = q0_p[(lane<<1)+0] * inv_sqrt_d; q0_1 = q0_p[(lane<<1)+1] * inv_sqrt_d;
            q1_0 = q1_p[(lane<<1)+0] * inv_sqrt_d; q1_1 = q1_p[(lane<<1)+1] * inv_sqrt_d;
            q2_0 = q2_p[(lane<<1)+0] * inv_sqrt_d; q2_1 = q2_p[(lane<<1)+1] * inv_sqrt_d;
            q3_0 = q3_p[(lane<<1)+0] * inv_sqrt_d; q3_1 = q3_p[(lane<<1)+1] * inv_sqrt_d;
            oo0_0 = oo0_1 = oo1_0 = oo1_1 = 0.0f;
            oo2_0 = oo2_1 = oo3_0 = oo3_1 = 0.0f;
        }
    }

    __shared__ float m0_sh, l0_sh, m1_sh, l1_sh, m2_sh, l2_sh, m3_sh, l3_sh;
    if (lane == 0) {
        m0_sh = m1_sh = m2_sh = m3_sh = -std::numeric_limits<float>::infinity();
        l0_sh = l1_sh = l2_sh = l3_sh = 0.0f;
    }
    wave_sync();

    const int BcTry = (bc_tile_hint > 0 && bc_tile_hint <= BcMax) ? bc_tile_hint : BcMax;
    for (int t0 = t_start; t0 < t_end; t0 += BcTry) {
        const int bc = min(BcTry, t_end - t0);

        float m_tile0 = -std::numeric_limits<float>::infinity();
        float m_tile1 = -std::numeric_limits<float>::infinity();
        float m_tile2 = -std::numeric_limits<float>::infinity();
        float m_tile3 = -std::numeric_limits<float>::infinity();

        #pragma unroll 1
        for (int r = 0; r < bc; ++r) {
            const int t  = t0 + r;
            const int rt = (local != 0) ? (cap_pow2 ? (t & (cap - 1)) : (t % cap)) : t;
            const char* __restrict__ kp_bytes = k_layer_base + (long long)rt * step_bytes + head_off_bytes;

            float p0=0, p1=0, p2=0, p3=0;
            if constexpr (std::is_same<KV_T,float>::value) {
                const float* __restrict__ k_head = reinterpret_cast<const float*>(kp_bytes);
                if (lane < 64) {
                    float kv = ld_global_cond(k_head + lane);
                    p0 = q0r * kv; p1 = q1r * kv; p2 = q2r * kv; p3 = q3r * kv;
                }
            } else {
                if (lane < 32) {
                    if (is_aligned_4(kp_bytes)) {
                        const uint32_t* __restrict__ k2 = reinterpret_cast<const uint32_t*>(kp_bytes);
                        uint32_t w = ld_global_u32_cond(k2 + lane);
                        float k0, k1; unpack_bf16x2(w, k0, k1);
                        p0 = q0_0 * k0 + q0_1 * k1;
                        p1 = q1_0 * k0 + q1_1 * k1;
                        p2 = q2_0 * k0 + q2_1 * k1;
                        p3 = q3_0 * k0 + q3_1 * k1;
                    } else {
                        const __hip_bfloat16* __restrict__ k16 = reinterpret_cast<const __hip_bfloat16*>(kp_bytes);
                        float k0 = bf16_to_f32_intr(k16[(lane<<1)+0]);
                        float k1 = bf16_to_f32_intr(k16[(lane<<1)+1]);
                        p0 = q0_0 * k0 + q0_1 * k1;
                        p1 = q1_0 * k0 + q1_1 * k1;
                        p2 = q2_0 * k0 + q2_1 * k1;
                        p3 = q3_0 * k0 + q3_1 * k1;
                    }
                }
            }

            float s0r = wave_reduce_sum_width<REDW>(p0);
            float s1r = wave_reduce_sum_width<REDW>(p1);
            float s2r = wave_reduce_sum_width<REDW>(p2);
            float s3r = wave_reduce_sum_width<REDW>(p3);
            if (lane == 0) {
                if constexpr (USE_MASK) {
                    const float add = mask[(long long)pos_b * seq_len + t];
                    s0r += add; s1r += add; s2r += add; s3r += add;
                }
                s0[r] = s0r; m_tile0 = fmaxf(m_tile0, s0r);
                s1[r] = s1r; m_tile1 = fmaxf(m_tile1, s1r);
                s2[r] = s2r; m_tile2 = fmaxf(m_tile2, s2r);
                s3[r] = s3r; m_tile3 = fmaxf(m_tile3, s3r);
            }
        }
        wave_sync();

        __shared__ float sc0, il0, mn0_sh;
        if (lane == 0) { float mp=m0_sh; float mn=fmaxf(mp, m_tile0); mn0_sh=mn; }
        wave_sync();
        float mn0 = __shfl(mn0_sh, 0, REDW);
        float part0 = 0.0f; for (int r = lane; r < bc; r += REDW) part0 += __expf(s0[r] - mn0);
        float sum0 = wave_reduce_sum_width<REDW>(part0);
        if (lane == 0) {
            float mp=m0_sh, lp=l0_sh; float a=__expf(mp - mn0); float ln=a*lp + sum0;
            sc0 = (ln>0.0f)? (a*lp)/ln : 0.0f; il0 = (ln>0.0f)? 1.0f/ln : 0.0f; m0_sh=mn0; l0_sh=ln;
        }
        wave_sync();

        __shared__ float sc1, il1, mn1_sh;
        if (lane == 0) { float mp=m1_sh; float mn=fmaxf(mp, m_tile1); mn1_sh=mn; }
        wave_sync();
        float mn1 = __shfl(mn1_sh, 0, REDW);
        float part1 = 0.0f; for (int r = lane; r < bc; r += REDW) part1 += __expf(s1[r] - mn1);
        float sum1 = wave_reduce_sum_width<REDW>(part1);
        if (lane == 0) {
            float mp=m1_sh, lp=l1_sh; float a=__expf(mp - mn1); float ln=a*lp + sum1;
            sc1 = (ln>0.0f)? (a*lp)/ln : 0.0f; il1 = (ln>0.0f)? 1.0f/ln : 0.0f; m1_sh=mn1; l1_sh=ln;
        }
        wave_sync();

        __shared__ float sc2, il2, mn2_sh;
        if (lane == 0) { float mp=m2_sh; float mn=fmaxf(mp, m_tile2); mn2_sh=mn; }
        wave_sync();
        float mn2 = __shfl(mn2_sh, 0, REDW);
        float part2 = 0.0f; for (int r = lane; r < bc; r += REDW) part2 += __expf(s2[r] - mn2);
        float sum2 = wave_reduce_sum_width<REDW>(part2);
        if (lane == 0) {
            float mp=m2_sh, lp=l2_sh; float a=__expf(mp - mn2); float ln=a*lp + sum2;
            sc2 = (ln>0.0f)? (a*lp)/ln : 0.0f; il2 = (ln>0.0f)? 1.0f/ln : 0.0f; m2_sh=mn2; l2_sh=ln;
        }
        wave_sync();

        __shared__ float sc3, il3, mn3_sh;
        if (lane == 0) { float mp=m3_sh; float mn=fmaxf(mp, m_tile3); mn3_sh=mn; }
        wave_sync();
        float mn3 = __shfl(mn3_sh, 0, REDW);
        float part3 = 0.0f; for (int r = lane; r < bc; r += REDW) part3 += __expf(s3[r] - mn3);
        float sum3 = wave_reduce_sum_width<REDW>(part3);
        if (lane == 0) {
            float mp=m3_sh, lp=l3_sh; float a=__expf(mp - mn3); float ln=a*lp + sum3;
            sc3 = (ln>0.0f)? (a*lp)/ln : 0.0f; il3 = (ln>0.0f)? 1.0f/ln : 0.0f; m3_sh=mn3; l3_sh=ln;
        }
        wave_sync();

        const float s_sc0 = __shfl(sc0, 0, REDW);
        const float s_sc1 = __shfl(sc1, 0, REDW);
        const float s_sc2 = __shfl(sc2, 0, REDW);
        const float s_sc3 = __shfl(sc3, 0, REDW);
        if constexpr (std::is_same<KV_T,float>::value) {
            if (lane < 64) { o0r *= s_sc0; o1r *= s_sc1; o2r *= s_sc2; o3r *= s_sc3; }
        } else {
            if (lane < 32) {
                oo0_0 *= s_sc0; oo0_1 *= s_sc0; oo1_0 *= s_sc1; oo1_1 *= s_sc1;
                oo2_0 *= s_sc2; oo2_1 *= s_sc2; oo3_0 *= s_sc3; oo3_1 *= s_sc3;
            }
        }

        const float s_il0 = __shfl(il0, 0, REDW);
        const float s_il1 = __shfl(il1, 0, REDW);
        const float s_il2 = __shfl(il2, 0, REDW);
        const float s_il3 = __shfl(il3, 0, REDW);
        const float s_mn0 = __shfl(mn0_sh, 0, REDW);
        const float s_mn1 = __shfl(mn1_sh, 0, REDW);
        const float s_mn2 = __shfl(mn2_sh, 0, REDW);
        const float s_mn3 = __shfl(mn3_sh, 0, REDW);

        #pragma unroll 1
        for (int r = 0; r < bc; ++r) {
            const int t  = t0 + r;
            const int rt = (local != 0) ? (cap_pow2 ? (t & (cap - 1)) : (t % cap)) : t;
            const char* __restrict__ vp_bytes = v_layer_base + (long long)rt * step_bytes + head_off_bytes;

            float w0f, w1f, w2f, w3f;
            if (lane == 0) {
                w0f = __expf(s0[r] - s_mn0) * s_il0;
                w1f = __expf(s1[r] - s_mn1) * s_il1;
                w2f = __expf(s2[r] - s_mn2) * s_il2;
                w3f = __expf(s3[r] - s_mn3) * s_il3;
            }
            w0f = __shfl(w0f, 0, REDW);
            w1f = __shfl(w1f, 0, REDW);
            w2f = __shfl(w2f, 0, REDW);
            w3f = __shfl(w3f, 0, REDW);

            if constexpr (std::is_same<KV_T,float>::value) {
                const float* __restrict__ v_head = reinterpret_cast<const float*>(vp_bytes);
                if (lane < 64) {
                    float vv = ld_global_cond(v_head + lane);
                    o0r += w0f * vv; o1r += w1f * vv; o2r += w2f * vv; o3r += w3f * vv;
                }
            } else {
                if (lane < 32) {
                    if (is_aligned_4(vp_bytes)) {
                        const uint32_t* __restrict__ v2 = reinterpret_cast<const uint32_t*>(vp_bytes);
                        uint32_t w = ld_global_u32_cond(v2 + lane);
                        float v0, v1; unpack_bf16x2(w, v0, v1);
                        oo0_0 += w0f * v0; oo0_1 += w0f * v1;
                        oo1_0 += w1f * v0; oo1_1 += w1f * v1;
                        oo2_0 += w2f * v0; oo2_1 += w2f * v1;
                        oo3_0 += w3f * v0; oo3_1 += w3f * v1;
                    } else {
                        const __hip_bfloat16* __restrict__ v16 = reinterpret_cast<const __hip_bfloat16*>(vp_bytes);
                        float v0 = bf16_to_f32_intr(v16[(lane<<1)+0]);
                        float v1 = bf16_to_f32_intr(v16[(lane<<1)+1]);
                        oo0_0 += w0f * v0; oo0_1 += w0f * v1;
                        oo1_0 += w1f * v0; oo1_1 += w1f * v1;
                        oo2_0 += w2f * v0; oo2_1 += w2f * v1;
                        oo3_0 += w3f * v0; oo3_1 += w3f * v1;
                    }
                }
            }
        }
    }

    if (attn_sinks_half != nullptr) {
        float sink0 = __bfloat162float(attn_sinks_half[h0]);
        float sink1 = __bfloat162float(attn_sinks_half[h1]);
        float sink2 = __bfloat162float(attn_sinks_half[h2]);
        float sink3 = __bfloat162float(attn_sinks_half[h3]);
        if (isfinite(sink0)) { float r=1.0f; if (lane==0){float mp=m0_sh, lp=l0_sh; float mn=fmaxf(mp,sink0); float a=__expf(mp-mn)*lp; float b=__expf(sink0-mn); float d=a+b; r=(d>0.0f)?(a/d):1.0f; m0_sh=mn; l0_sh=d;} r=__shfl(r,0,REDW); if constexpr (std::is_same<KV_T,float>::value){ if(lane<64) o0r*=r; } else { if(lane<32){ oo0_0*=r; oo0_1*=r; } } }
        if (isfinite(sink1)) { float r=1.0f; if (lane==0){float mp=m1_sh, lp=l1_sh; float mn=fmaxf(mp,sink1); float a=__expf(mp-mn)*lp; float b=__expf(sink1-mn); float d=a+b; r=(d>0.0f)?(a/d):1.0f; m1_sh=mn; l1_sh=d;} r=__shfl(r,0,REDW); if constexpr (std::is_same<KV_T,float>::value){ if(lane<64) o1r*=r; } else { if(lane<32){ oo1_0*=r; oo1_1*=r; } } }
        if (isfinite(sink2)) { float r=1.0f; if (lane==0){float mp=m2_sh, lp=l2_sh; float mn=fmaxf(mp,sink2); float a=__expf(mp-mn)*lp; float b=__expf(sink2-mn); float d=a+b; r=(d>0.0f)?(a/d):1.0f; m2_sh=mn; l2_sh=d;} r=__shfl(r,0,REDW); if constexpr (std::is_same<KV_T,float>::value){ if(lane<64) o2r*=r; } else { if(lane<32){ oo2_0*=r; oo2_1*=r; } } }
        if (isfinite(sink3)) { float r=1.0f; if (lane==0){float mp=m3_sh, lp=l3_sh; float mn=fmaxf(mp,sink3); float a=__expf(mp-mn)*lp; float b=__expf(sink3-mn); float d=a+b; r=(d>0.0f)?(a/d):1.0f; m3_sh=mn; l3_sh=d;} r=__shfl(r,0,REDW); if constexpr (std::is_same<KV_T,float>::value){ if(lane<64) o3r*=r; } else { if(lane<32){ oo3_0*=r; oo3_1*=r; } } }
    }

    if constexpr (std::is_same<KV_T,float>::value) {
        if (lane < 64) { o0_out[lane]=o0r; o1_out[lane]=o1r; o2_out[lane]=o2r; o3_out[lane]=o3r; }
    } else {
        if (lane < 32) {
            o0_out[(lane<<1)+0]=oo0_0; o0_out[(lane<<1)+1]=oo0_1;
            o1_out[(lane<<1)+0]=oo1_0; o1_out[(lane<<1)+1]=oo1_1;
            o2_out[(lane<<1)+0]=oo2_0; o2_out[(lane<<1)+1]=oo2_1;
            o3_out[(lane<<1)+0]=oo3_0; o3_out[(lane<<1)+1]=oo3_1;
        }
    }
}

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

    // ---- shared memory size hint ----
    constexpr int BcMax = 64;
    size_t shmem_br4 = 4 * BcMax * sizeof(float);

    const __hip_bfloat16* attn_sinks_head =
        (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;
    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0) && (mask != nullptr);

    dim3 block(64); // one wave per CTA
    const int bc_tile_hint = min(64, seq_len);

    const bool can_br4 = (n_attn_heads % 4 == 0) && (kv_mul % 4 == 0);
    if (can_br4) {
        dim3 grid4(n_attn_heads / 4, B);
        if (kv_cache_is_fp16) {
            if (use_mask) {
                hipLaunchKernelGGL(
                    (fa2_decode_br4_head64<__hip_bfloat16, true, BcMax>),
                    grid4, block, shmem_br4, stream,
                    q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                    B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                    d_pos_per_token, d_batch_indices, (long long)B_stride,
                    d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                    bc_tile_hint
                );
            } else {
                hipLaunchKernelGGL(
                    (fa2_decode_br4_head64<__hip_bfloat16, false, BcMax>),
                    grid4, block, shmem_br4, stream,
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
                    (fa2_decode_br4_head64<float, true, BcMax>),
                    grid4, block, shmem_br4, stream,
                    q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                    B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                    d_pos_per_token, d_batch_indices, (long long)B_stride,
                    d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                    bc_tile_hint
                );
            } else {
                hipLaunchKernelGGL(
                    (fa2_decode_br4_head64<float, false, BcMax>),
                    grid4, block, shmem_br4, stream,
                    q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                    B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                    d_pos_per_token, d_batch_indices, (long long)B_stride,
                    d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                    bc_tile_hint
                );
            }
        }
    }
    CHECK_HIP(hipGetLastError());
}
