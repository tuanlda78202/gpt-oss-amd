#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

// GEMM: A[M,K] bf16, B[N,K] fp32, C[N,M] fp32 (row-major by N)
__global__ void gemm_bf16_f32_mfma_logits(const __hip_bfloat16* __restrict__ A, // [M,K] bf16
                                          const float* __restrict__ B,          // [N,K] f32
                                          float* __restrict__ C,                // [N,M] f32
                                          int M, int N, int K) {
    // Tile geometry
    constexpr int MT_M = 256; // CTA tile height (M)
    constexpr int MT_N = 64;  // CTA tile width  (N)
    constexpr int BK = 16;    // K-slice (multiple of 16)

    // 16 waves × 64 threads = 1024 threads per CTA
    constexpr int WR = 16;                              // waves along M (8 * 16 rows = 128)
    constexpr int WC = 1;                               // waves along N (1 * 64 cols = 64)
    constexpr int WAVES_PER_CTA = WR * WC;              // 16
    constexpr int THREADS_PER_CTA = 64 * WAVES_PER_CTA; // 1024

    const int tid = threadIdx.x;     // [0..1023]
    const int wave_id = tid / 64;    // [0..15]
    const int lane_id = tid % 64;    // [0..63]
    const int lane_x = lane_id % 16; // [0..15]
    const int lane_y = lane_id / 16; // [0..3]

    const int m0_cta = blockIdx.y * MT_M;
    const int n0_cta = blockIdx.x * MT_N;

    const int m0_wave = (wave_id % WR) * 16;
    const int n0_wave = (wave_id / WR) * 64;

    // Shared memory: keep A as [MT_M,BK], store B as N-major [MT_N,BK] for vector LDS ops.
    __shared__ union {
        struct {
            __align__(16) __hip_bfloat16 sA[2][MT_M * BK];
            __align__(16) short sB_T[2][MT_N * BK]; // N-major in LDS
        } tiles;
    } smem;

    // Accumulators (each thread computes 4 rows × 4 groups of 16 cols)
    f4 c0 = {0.f}, c1 = {0.f}, c2 = {0.f}, c3 = {0.f};

    const int num_k_tiles = (K + BK - 1) / BK;

    // --- Cooperative gmem->smem loader (double-buffered) ---
    auto load_tile_g2s = [&](int k_start, int buf) {
        // A tile: [MT_M, BK] — vectorized i16x4 copies (contiguous)
        const int total_vecsA = (MT_M * BK) / 4;
        for (int v = tid; v < total_vecsA; v += THREADS_PER_CTA) {
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

        // B tile: load 4 contiguous K values as float4 → convert → store as i16x4
        // Store as N-major in LDS: sB_T[n_loc * BK + k0..k0+3] are contiguous.
        const int total_vecsB = (BK * MT_N) / 4; // each vec covers 4 along K
        for (int v = tid; v < total_vecsB; v += THREADS_PER_CTA) {
            int n_loc = v / (BK / 4);            // [0..MT_N-1]
            int k_loc_base = (v % (BK / 4)) * 4; // 0,4,8,12

            int n_gl = n0_cta + n_loc;
            i16x4 bvals = {0, 0, 0, 0};
            if (n_gl < N && k_start + k_loc_base < K) {
                const float* gB = B + (size_t)n_gl * K + k_start + k_loc_base;
                bvals = load_and_convert_f32_to_bf16x4(gB);
            }

            // Contiguous 8B write in LDS
            i16x4* sBv = reinterpret_cast<i16x4*>(&smem.tiles.sB_T[buf][n_loc * BK + k_loc_base]);
            *sBv = bvals;
        }
    };

    // Preload first tiles
    load_tile_g2s(0, 0);
    __syncthreads();

    // --- Main loop over K tiles (double buffered) ---
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int ping = kt & 1;
        int pong = ping ^ 1;
        int k0 = kt * BK;

        if (kt + 1 < num_k_tiles) {
            load_tile_g2s(k0 + BK, pong);
        }

        // Compute on 'ping'
        // Hoist LDS base pointers
        const __hip_bfloat16* sA_ping = &smem.tiles.sA[ping][0];
        const short* sB_ping = &smem.tiles.sB_T[ping][0];

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // A fragment: 4 bf16 along K for this lane
            const __hip_bfloat16* a_sptr = sA_ping + (m0_wave + lane_x) * BK + (kk + lane_y * 4);
            i16x4 avec = *reinterpret_cast<const i16x4*>(a_sptr);

            // B fragments: each is 4 bf16 contiguous in LDS (thanks to N-major layout)
            const int k_row_base = kk + lane_y * 4;

            const int n_col0 = n0_wave + 0 * 16 + lane_x;
            const int n_col1 = n0_wave + 1 * 16 + lane_x;
            const int n_col2 = n0_wave + 2 * 16 + lane_x;
            const int n_col3 = n0_wave + 3 * 16 + lane_x;

            i16x4 b0 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col0 * BK + k_row_base]);
            i16x4 b1 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col1 * BK + k_row_base]);
            i16x4 b2 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col2 * BK + k_row_base]);
            i16x4 b3 = *reinterpret_cast<const i16x4*>(&sB_ping[n_col3 * BK + k_row_base]);

            // 4 MFMA ops
            c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b0, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b1, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b2, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(avec, b3, c3, 0, 0, 0);
        }
        __syncthreads();
    }

    // --- Vectorized writeback to C (N-major: N × M with M contiguous) ---
    const int m_out_base = m0_cta + m0_wave + lane_y * 4; // 4 consecutive rows
    const int n_out0 = n0_cta + n0_wave + 0 * 16 + lane_x;
    const int n_out1 = n0_cta + n0_wave + 1 * 16 + lane_x;
    const int n_out2 = n0_cta + n0_wave + 2 * 16 + lane_x;
    const int n_out3 = n0_cta + n0_wave + 3 * 16 + lane_x;

    // Fast path: fully in-bounds → single 16B store per column
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
// Tail-safe scalar fallback
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

void gemm_logits(float* xout, const float* x_batch, const __hip_bfloat16* w, int N, int K, int M) {
    dim3 grid(CEIL_DIV(N, 64), CEIL_DIV(M, 256));
    dim3 block(1024, 1, 1);

    gemm_bf16_f32_mfma_logits<<<grid, block>>>(w, x_batch, xout, M, N, K);
}
