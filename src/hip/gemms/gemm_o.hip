#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

// GEMM: A[M,K] bf16, B[N,K] fp32, C[N,M] fp32 (row-major by N)
__global__ void gemm_bf16_f32_mfma_o(const __hip_bfloat16* __restrict__ A,    // [M,K] bf16
                                     const float* __restrict__ B,             // [N,K] f32
                                     float* __restrict__ C,                   // [N,M] f32
                                     const __hip_bfloat16* __restrict__ bias, // [M]
                                     int M, int N, int K) {
    // Tile geometry
    constexpr int MT_M = 256; // CTA tile height (M)
    constexpr int MT_N = 64;  // CTA tile width  (N)
    constexpr int BK = 64;    // K-slice (multiple of 16)
    constexpr int PAD = 4;
    constexpr int LDS_STRIDE_B = BK + PAD;

    constexpr int WR = 16; // waves along M
    constexpr int WC = 1;  // waves along N
    constexpr int WAVES_PER_CTA = WR * WC;
    constexpr int THREADS_PER_CTA = 64 * WAVES_PER_CTA;

    const int tid = threadIdx.x;
    const int wave_id = tid / 64;
    const int lane_id = tid % 64;
    const int lane_x = lane_id % 16;
    const int lane_y = lane_id / 16;

    const int m0_cta = blockIdx.y * MT_M;
    const int n0_cta = blockIdx.x * MT_N;

    const int m0_wave = (wave_id % WR) * 16;
    const int n0_wave = (wave_id / WR) * 64;

    __shared__ union {
        short sB_T[2][MT_N * LDS_STRIDE_B];
    } smem;

    const int m_bias_base = m0_cta + m0_wave + lane_y * 4;

    f4 bias4 = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m_bias_base + i;
        if (m < M)
            bias4[i] = __bfloat162float(bias[m]);
    }

    f4 c0 = bias4;
    f4 c1 = bias4;
    f4 c2 = bias4;
    f4 c3 = bias4;

    const int num_k_tiles = CEIL_DIV(K, BK);

    // --- Cooperative gmem->smem loader (double-buffered) ---
    auto load_tile_g2s = [&](int k_start, int buf) {
        const int total_vecsB = (BK * MT_N) / 4;
        for (int v = tid; v < total_vecsB; v += THREADS_PER_CTA) {
            int n_loc = v / (BK / 4);
            int k_loc_base = (v % (BK / 4)) * 4;

            int n_gl = n0_cta + n_loc;
            i16x4 bvals = {0, 0, 0, 0};
            if (n_gl < N && k_start + k_loc_base < K) {
                const float* gB = B + (size_t)n_gl * K + k_start + k_loc_base;
                bvals = load_and_convert_f32_to_bf16x4(gB);
            }

            i16x4* sBv =
                reinterpret_cast<i16x4*>(&smem.sB_T[buf][n_loc * LDS_STRIDE_B + k_loc_base]);
            *sBv = bvals;
        }
    };

    load_tile_g2s(0, 0);
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int ping = kt & 1;
        int pong = ping ^ 1;
        int k0 = kt * BK;

        if (kt + 1 < num_k_tiles) {
            load_tile_g2s(k0 + BK, pong);
        }

        const short* sB_ping = &smem.sB_T[ping][0];
        const int m_gl = m0_cta + m0_wave + lane_x;

        auto load_avec = [&](int kk_offset) {
            i16x4 avec_local = {0, 0, 0, 0};
            if (m_gl < M) {
                int k_gl_base = k0 + kk_offset + lane_y * 4;
                const __hip_bfloat16* gA_ptr = A + (size_t)m_gl * K + k_gl_base;
                if (k_gl_base + 3 < K) {
                    avec_local = *reinterpret_cast<const i16x4*>(gA_ptr);
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int k_gl = k_gl_base + i;
                        if (k_gl < K) {
                            avec_local[i] = bf16_bits(gA_ptr[i]);
                        }
                    }
                }
            }
            return avec_local;
        };

        i16x4 avec_curr = load_avec(0);

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            i16x4 avec_next = {0, 0, 0, 0};
            if (kk + 16 < BK)
                avec_next = load_avec(kk + 16);

            const int k_row_base = kk + lane_y * 4;
            const int n_col0 = n0_wave + 0 * 16 + lane_x;
            const int n_col1 = n0_wave + 1 * 16 + lane_x;
            const int n_col2 = n0_wave + 2 * 16 + lane_x;
            const int n_col3 = n0_wave + 3 * 16 + lane_x;

            i16x4 b0 = {0, 0, 0, 0};
            i16x4 b1 = {0, 0, 0, 0};
            i16x4 b2 = {0, 0, 0, 0};
            i16x4 b3 = {0, 0, 0, 0};
            if (n_col0 < MT_N)
                b0 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col0 * LDS_STRIDE_B + k_row_base]);
            if (n_col1 < MT_N)
                b1 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col1 * LDS_STRIDE_B + k_row_base]);
            if (n_col2 < MT_N)
                b2 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col2 * LDS_STRIDE_B + k_row_base]);
            if (n_col3 < MT_N)
                b3 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col3 * LDS_STRIDE_B + k_row_base]);

            // 4 MFMA ops
            c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b0, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b1, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b2, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec_curr, b3, c3, 0, 0, 0);

            if (kk + 16 < BK)
                avec_curr = avec_next;
        }
        __syncthreads();
    }

    // --- Vectorized writeback to C (N-major: N × M with M contiguous) ---
    const int m_out_base = m0_cta + m0_wave + lane_y * 4;
    const int n_out0 = n0_cta + n0_wave + 0 * 16 + lane_x;
    const int n_out1 = n0_cta + n0_wave + 1 * 16 + lane_x;
    const int n_out2 = n0_cta + n0_wave + 2 * 16 + lane_x;
    const int n_out3 = n0_cta + n0_wave + 3 * 16 + lane_x;

    if (m_out_base + 3 < M) {
        if (n_out0 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out0 * M + m_out_base]) = c0;
        if (n_out1 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out1 * M + m_out_base]) = c1;
        if (n_out2 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out2 * M + m_out_base]) = c2;
        if (n_out3 < N)
            *reinterpret_cast<f4*>(&C[(size_t)n_out3 * M + m_out_base]) = c3;
    } else {

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int m_out = m_out_base + i;
            if (m_out < M) {
                if (n_out0 < N)
                    nt_store(&C[(size_t)n_out0 * M + m_out], c0[i]);
                if (n_out1 < N)
                    nt_store(&C[(size_t)n_out1 * M + m_out], c1[i]);
                if (n_out2 < N)
                    nt_store(&C[(size_t)n_out2 * M + m_out], c2[i]);
                if (n_out3 < N)
                    nt_store(&C[(size_t)n_out3 * M + m_out], c3[i]);
            }
        }
    }
}

void gemm_o(float* xout, const float* x_batch, const __hip_bfloat16* w, const __hip_bfloat16* bias,
            int N, int K, int M, hipStream_t stream) {
    dim3 grid(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
    dim3 block(1024, 1, 1);

    gemm_bf16_f32_mfma_o<<<grid, block, 0, stream>>>(w, x_batch, xout, bias, M, N, K);
}
