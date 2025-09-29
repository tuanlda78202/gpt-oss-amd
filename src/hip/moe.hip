#pragma once
#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include "BLAS.hip"


// MLP1 kernel (updated): [Ne, H] x [H, 2I] -> fused SwiGLU [Ne, I]
__global__ void moe_mlp1_swiglu_mfma_kernel(const int* __restrict__ work_queue, int work_start,
                                            int work_count, const float* __restrict__ x_by_expert,
                                            float* __restrict__ gate_by_expert,
                                            const __hip_bfloat16* __restrict__ weights_base,
                                            const __hip_bfloat16* __restrict__ bias_base, int H,
                                            int I, long long expert_weight_stride,
                                            long long expert_bias_stride, float alpha,
                                            float swiglu_limit) {
    constexpr int MT_M = 256; // rows (M = 2*I)
    constexpr int MT_N = 64;  // cols (N = Ne)
    constexpr int BK = 16;    // K-slice
    constexpr int WR = 16;    // waves along M
    constexpr int WC = 1;     // waves along N
    constexpr int WAVES = WR * WC;
    constexpr int TPB = 64 * WAVES; // 1024

    const int work_idx = blockIdx.z;
    if (work_idx >= work_count)
        return;

    const int q = (work_start + work_idx) * 3;
    const int e = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne = work_queue[q + 2];

    const int M = 2 * I;
    const int N = Ne;
    const int K = H;

    if (M == 0 || N == 0 || K == 0)
        return;

    const __hip_bfloat16* __restrict__ A =
        weights_base + static_cast<long long>(e) * expert_weight_stride; // [M,K] bf16
    const __hip_bfloat16* __restrict__ bias =
        bias_base ? (bias_base + static_cast<long long>(e) * expert_bias_stride) : nullptr; // [M]
    const float* __restrict__ B_batch = x_by_expert + static_cast<long long>(off) * H;      // [N,K]
    float* __restrict__ O_batch = gate_by_expert + static_cast<long long>(off) * I;         // [N,I]

    // CTA tile origins
    const int m0_cta = blockIdx.y * MT_M;
    const int n0_cta = blockIdx.x * MT_N;
    if (m0_cta >= M || n0_cta >= N)
        return;

    // Thread/wave ids
    const int tid = threadIdx.x;
    const int wave_id = tid / 64;
    const int lane_id = tid % 64;
    const int lane_x = lane_id % 16;
    const int lane_y = lane_id / 16;

    // Wave tile origins
    const int m0_wave = (wave_id % WR) * 16;
    const int n0_wave = (wave_id / WR) * 64;

    // Shared memory tiles (double-buffered)
    __shared__ union {
        struct {
            __align__(16) __hip_bfloat16 sA[2][MT_M * BK];
            __align__(16) short sB_T[2][MT_N * BK]; // N-major in LDS
        } tiles;
    } smem;

    // Initialize accumulators with bias
    const int m_bias_base = m0_cta + m0_wave + lane_y * 4;
    f4 bias4 = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m_bias_base + i;
        if (bias && m < M)
            bias4[i] = __bfloat162float(bias[m]);
    }
    f4 c0 = bias4, c1 = bias4, c2 = bias4, c3 = bias4;

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_tile_g2s = [&](int n0_tile, int k_start, int buf) {
        // Load A (MT_M x BK) vectorized as i16x4
        const int total_vecsA = (MT_M * BK) / 4;
        for (int v = tid; v < total_vecsA; v += TPB) {
            int m_loc = v / (BK / 4);
            int k_loc_base = (v % (BK / 4)) * 4;

            __hip_bfloat16* sA_base = &smem.tiles.sA[buf][m_loc * BK + k_loc_base];

            if (m0_cta + m_loc < M && k_start + k_loc_base < K) {
                const i16x4* gA = reinterpret_cast<const i16x4*>(A + (size_t)(m0_cta + m_loc) * K +
                                                                 k_start + k_loc_base);
                *reinterpret_cast<i16x4*>(sA_base) = *gA; // 8B load + 8B LDS store
            } else {
                *reinterpret_cast<i16x4*>(sA_base) = i16x4{0, 0, 0, 0};
            }
        }
        // Load B (BK x MT_N), from global B[N,K] f32 => bf16 and transpose into LDS
        const int total_vecsB = (BK * MT_N) / 4;
        for (int vec = tid; vec < total_vecsB; vec += TPB) {
            int n_loc = vec / (BK / 4);
            int k_loc4 = (vec % (BK / 4)) * 4;
            const int n_gl = n0_tile + n_loc;
            i16x4 b = {0, 0, 0, 0};
            if (n_gl < N && k_start + k_loc4 < K) {
                const float* gB = B_batch + (size_t)n_gl * K + k_start + k_loc4;
                b = load_and_convert_f32_to_bf16x4(gB);
            }
            i16x4* sBv = reinterpret_cast<i16x4*>(&smem.tiles.sB_T[buf][n_loc * BK + k_loc4]);
            *sBv = b;
        }
    };

    // Prefetch first tile
    // Loop across N tiles to honor grid_cap
    const int total_tiles_n = (N + MT_N - 1) / MT_N;
    for (int tile_x = blockIdx.x; tile_x < total_tiles_n; tile_x += gridDim.x) {
        const int n0_tile = tile_x * MT_N;

        // Reset accumulators for this N tile
        f4 c0 = bias4, c1 = bias4, c2 = bias4, c3 = bias4;

        load_tile_g2s(n0_tile, 0, 0);
        __syncthreads();

        for (int kt = 0; kt < num_k_tiles; ++kt) {
            const int ping = kt & 1;
            const int pong = ping ^ 1;
            const int k0 = kt * BK;

            if (kt + 1 < num_k_tiles) {
                load_tile_g2s(n0_tile, k0 + BK, pong);
            }

            const __hip_bfloat16* sA_ping = &smem.tiles.sA[ping][0];
            const short* sB_ping = &smem.tiles.sB_T[ping][0];

            // Compute on current tile
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                const __hip_bfloat16* a_sptr = sA_ping + (m0_wave + lane_x) * BK + (kk + lane_y * 4);
                i16x4 avec = *reinterpret_cast<const i16x4*>(a_sptr);

                const int k_row_base = kk + lane_y * 4;

                const int n_col0 = n0_wave + 0 * 16 + lane_x;
                const int n_col1 = n0_wave + 1 * 16 + lane_x;
                const int n_col2 = n0_wave + 2 * 16 + lane_x;
                const int n_col3 = n0_wave + 3 * 16 + lane_x;

                i16x4 b0 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col0 * BK + k_row_base]);
                i16x4 b1 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col1 * BK + k_row_base]);
                i16x4 b2 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col2 * BK + k_row_base]);
                i16x4 b3 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col3 * BK + k_row_base]);

                c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b0, c0, 0, 0, 0);
                c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b1, c1, 0, 0, 0);
                c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b2, c2, 0, 0, 0);
                c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b3, c3, 0, 0, 0);
            }
            __syncthreads();
        }

        // Epilogue: fused SwiGLU and store [N,I]
        const int m_out_base = m0_cta + m0_wave + lane_y * 4; // 4 rows per thread
        const int n_out0 = n0_tile + n0_wave + 0 * 16 + lane_x;
        const int n_out1 = n0_tile + n0_wave + 1 * 16 + lane_x;
        const int n_out2 = n0_tile + n0_wave + 2 * 16 + lane_x;
        const int n_out3 = n0_tile + n0_wave + 3 * 16 + lane_x;

        auto swiglu = [&](float gate_v, float up_v) {
            float g = fminf(gate_v, swiglu_limit);
            float u = fminf(fmaxf(up_v, -swiglu_limit), swiglu_limit);
            float sig = 1.0f / (1.0f + __expf(-alpha * g));
            return g * sig * (u + 1.0f);
        };

#pragma unroll
        for (int i = 0; i < 4; i += 2) {
            const int m_gate = m_out_base + i;
            const int m_up = m_gate + 1;
            if (m_gate >= M || m_up >= M)
                continue;

            const int i_idx = m_gate >> 1; // [0..I)
            if (i_idx >= I)
                continue;

            if (n_out0 < N)
                O_batch[(size_t)n_out0 * I + i_idx] = swiglu(c0[i], c0[i + 1]);
            if (n_out1 < N)
                O_batch[(size_t)n_out1 * I + i_idx] = swiglu(c1[i], c1[i + 1]);
            if (n_out2 < N)
                O_batch[(size_t)n_out2 * I + i_idx] = swiglu(c2[i], c2[i + 1]);
            if (n_out3 < N)
                O_batch[(size_t)n_out3 * I + i_idx] = swiglu(c3[i], c3[i + 1]);
        }
        __syncthreads();
    }
}

inline void moe_mlp1_swiglu(const int* work_queue, int work_start, int work_count,
                            const float* x_by_expert, float* gate_by_expert,
                            const __hip_bfloat16* weights_base, const __hip_bfloat16* bias_base,
                            int H, int I, long long expert_weight_stride,
                            long long expert_bias_stride, float alpha, float swiglu_limit,
                            int maxNe, int grid_cap, hipStream_t stream) {
    // Match the updated kernel tile sizes (from gemm_o flow)
    constexpr int MT_M = 256;
    constexpr int MT_N = 64;
    constexpr int BK = 16;
    constexpr int WR = 16;
    constexpr int WC = 1;
    constexpr int WAVES = WR * WC;

    const int grid_x = max(1, min(grid_cap, CEIL_DIV(maxNe, MT_N)));
    const int grid_y = max(1, CEIL_DIV(2 * I, MT_M));

    dim3 block(64 * WAVES, 1, 1); // 1024 threads
    dim3 grid(grid_x, grid_y, work_count);

    moe_mlp1_swiglu_mfma_kernel<<<grid, block>>>(
        work_queue, work_start, work_count, x_by_expert, gate_by_expert, weights_base, bias_base, H,
        I, expert_weight_stride, expert_bias_stride, alpha, swiglu_limit);
}

// MLP2 kernel (updated): [Ne, I] x [I, H] -> scatter to e_agg using bf16 MFMA flow
__global__ void moe_mlp2_scatter_mfma_kernel(
    const int* __restrict__ work_queue, int work_start, int work_count,
    const float* __restrict__ gate_by_expert, const int* __restrict__ tokens_flat,
    const float* __restrict__ weights_flat, float* __restrict__ e_agg,
    const __hip_bfloat16* __restrict__ weights_base, const __hip_bfloat16* __restrict__ bias_base,
    int I, int H, long long expert_weight_stride, long long expert_bias_stride) {
    constexpr int MT_M = 256; // rows (M = H)
    constexpr int MT_N = 64;  // cols (N = Ne)
    constexpr int BK = 16;    // K-slice
    constexpr int WR = 16;    // waves along M
    constexpr int WC = 1;     // waves along N
    constexpr int WAVES = WR * WC;
    constexpr int TPB = 64 * WAVES;

    const int work_idx = blockIdx.z;
    if (work_idx >= work_count)
        return;

    const int q = (work_start + work_idx) * 3;
    const int e = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne = work_queue[q + 2];

    const int M = H;
    const int N = Ne;
    const int K = I;

    if (M == 0 || N == 0 || K == 0)
        return;

    const __hip_bfloat16* __restrict__ A =
        weights_base + static_cast<long long>(e) * expert_weight_stride; // [M,K] bf16
    const __hip_bfloat16* __restrict__ bias =
        bias_base ? (bias_base + static_cast<long long>(e) * expert_bias_stride) : nullptr; // [M]
    const float* __restrict__ B_batch = gate_by_expert + static_cast<long long>(off) * I;   // [N,K]
    const int* __restrict__ tokens = tokens_flat + off;
    const float* __restrict__ scales = weights_flat + off;

    const int m0_cta = blockIdx.y * MT_M;
    const int n0_cta = blockIdx.x * MT_N;
    if (m0_cta >= M || n0_cta >= N)
        return;

    const int tid = threadIdx.x;
    const int wave_id = tid / 64;
    const int lane_id = tid % 64;
    const int lane_x = lane_id % 16;
    const int lane_y = lane_id / 16;

    const int m0_wave = (wave_id % WR) * 16;
    const int n0_wave = (wave_id / WR) * 64;

    __shared__ union {
        struct {
            __align__(16) __hip_bfloat16 sA[2][MT_M * BK];
            __align__(16) short sB_T[2][MT_N * BK]; // N-major in LDS
        } tiles;
    } smem;

    const int m_bias_base = m0_cta + m0_wave + lane_y * 4;
    f4 bias4 = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m_bias_base + i;
        if (bias && m < M)
            bias4[i] = __bfloat162float(bias[m]);
    }

    const int num_k_tiles = (K + BK - 1) / BK;
    const int total_tiles_n = (N + MT_N - 1) / MT_N;
    if (total_tiles_n == 0)
        return;

    auto load_tile_g2s = [&](int n0_tile, int k_start, int buf) {
        // Load A (MT_M x BK) vectorized as i16x4
        const int total_vecsA = (MT_M * BK) / 4;
        for (int v = tid; v < total_vecsA; v += TPB) {
            int m_loc = v / (BK / 4);
            int k_loc_base = (v % (BK / 4)) * 4;

            __hip_bfloat16* sA_base = &smem.tiles.sA[buf][m_loc * BK + k_loc_base];

            if (m0_cta + m_loc < M && k_start + k_loc_base < K) {
                const i16x4* gA = reinterpret_cast<const i16x4*>(A + (size_t)(m0_cta + m_loc) * K +
                                                                 k_start + k_loc_base);
                *reinterpret_cast<i16x4*>(sA_base) = *gA; // 8B load + 8B LDS store
            } else {
                *reinterpret_cast<i16x4*>(sA_base) = i16x4{0, 0, 0, 0};
            }
        }
        // Load B (BK x MT_N), from global B[N,K] f32 => bf16 and transpose into LDS
        const int total_vecsB = (BK * MT_N) / 4;
        for (int vec = tid; vec < total_vecsB; vec += TPB) {
            int n_loc = vec / (BK / 4);
            int k_loc4 = (vec % (BK / 4)) * 4;
            const int n_gl = n0_tile + n_loc;
            i16x4 b = {0, 0, 0, 0};
            if (n_gl < N && k_start + k_loc4 < K) {
                const float* gB = B_batch + (size_t)n_gl * K + k_start + k_loc4;
                b = load_and_convert_f32_to_bf16x4(gB);
            }
            i16x4* sBv = reinterpret_cast<i16x4*>(&smem.tiles.sB_T[buf][n_loc * BK + k_loc4]);
            *sBv = b;
        }
    };

    for (int tile_x = blockIdx.x; tile_x < total_tiles_n; tile_x += gridDim.x) {
        const int n0_tile = tile_x * MT_N;

        f4 c0 = bias4, c1 = bias4, c2 = bias4, c3 = bias4;

        load_tile_g2s(n0_tile, 0, 0);
        __syncthreads();

        for (int kt = 0; kt < num_k_tiles; ++kt) {
            const int ping = kt & 1;
            const int pong = ping ^ 1;
            const int k0 = kt * BK;

            if (kt + 1 < num_k_tiles) {
                load_tile_g2s(n0_tile, k0 + BK, pong);
            }

            const __hip_bfloat16* sA_ping = &smem.tiles.sA[ping][0];
            const short* sB_ping = &smem.tiles.sB_T[ping][0];

#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                const __hip_bfloat16* a_sptr =
                    sA_ping + (m0_wave + lane_x) * BK + (kk + lane_y * 4);
                i16x4 avec = *reinterpret_cast<const i16x4*>(a_sptr);

                const int k_row_base = kk + lane_y * 4;

                const int n_col0 = n0_wave + 0 * 16 + lane_x;
                const int n_col1 = n0_wave + 1 * 16 + lane_x;
                const int n_col2 = n0_wave + 2 * 16 + lane_x;
                const int n_col3 = n0_wave + 3 * 16 + lane_x;

                i16x4 b0 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col0 * BK + k_row_base]);
                i16x4 b1 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col1 * BK + k_row_base]);
                i16x4 b2 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col2 * BK + k_row_base]);
                i16x4 b3 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col3 * BK + k_row_base]);

                c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b0, c0, 0, 0, 0);
                c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b1, c1, 0, 0, 0);
                c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b2, c2, 0, 0, 0);
                c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b3, c3, 0, 0, 0);
            }
            __syncthreads();
        }

        const int m_out_base = m0_cta + m0_wave + lane_y * 4;
        const int n_out0 = n0_tile + n0_wave + lane_x;
        const int n_out1 = n_out0 + 16;
        const int n_out2 = n_out0 + 32;
        const int n_out3 = n_out0 + 48;

        float* out0 = nullptr;
        float* out1 = nullptr;
        float* out2 = nullptr;
        float* out3 = nullptr;
        float scale0 = 0.f, scale1 = 0.f, scale2 = 0.f, scale3 = 0.f;

        if (n_out0 < N) {
            out0 = e_agg + static_cast<size_t>(tokens[n_out0]) * H;
            scale0 = scales[n_out0];
        }
        if (n_out1 < N) {
            out1 = e_agg + static_cast<size_t>(tokens[n_out1]) * H;
            scale1 = scales[n_out1];
        }
        if (n_out2 < N) {
            out2 = e_agg + static_cast<size_t>(tokens[n_out2]) * H;
            scale2 = scales[n_out2];
        }
        if (n_out3 < N) {
            out3 = e_agg + static_cast<size_t>(tokens[n_out3]) * H;
            scale3 = scales[n_out3];
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m_out = m_out_base + i;
            if (m_out >= M)
                continue;

            const float v0 = c0[i];
            const float v1 = c1[i];
            const float v2 = c2[i];
            const float v3 = c3[i];

            if (out0)
                atomicAdd(out0 + m_out, v0 * scale0);
            if (out1)
                atomicAdd(out1 + m_out, v1 * scale1);
            if (out2)
                atomicAdd(out2 + m_out, v2 * scale2);
            if (out3)
                atomicAdd(out3 + m_out, v3 * scale3);
        }
        __syncthreads();
    }
}

inline void moe_mlp2_scatter(const int* work_queue, int work_start, int work_count,
                             const float* gate_by_expert, const int* tokens_flat,
                             const float* weights_flat, float* e_agg,
                             const __hip_bfloat16* weights_base, const __hip_bfloat16* bias_base,
                             int I, int H, long long expert_weight_stride,
                             long long expert_bias_stride, int maxNe, int grid_cap,
                             hipStream_t stream) {
    constexpr int MT_M = 256;
    constexpr int MT_N = 64;
    constexpr int BK = 16;
    constexpr int WR = 16;
    constexpr int WC = 1;
    constexpr int WAVES = WR * WC;

    const int grid_x = max(1, min(grid_cap, CEIL_DIV(maxNe, MT_N)));
    const int grid_y = max(1, CEIL_DIV(H, MT_M));

    dim3 block(64 * WAVES, 1, 1);
    dim3 grid(grid_x, grid_y, work_count);

    moe_mlp2_scatter_mfma_kernel<<<grid, block>>>(
        work_queue, work_start, work_count, gate_by_expert, tokens_flat, weights_flat, e_agg,
        weights_base, bias_base, I, H, expert_weight_stride, expert_bias_stride);
}
