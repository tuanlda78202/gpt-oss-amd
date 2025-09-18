#include "BLAS.hip"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

static inline __host__ __device__ int ceil_div1(int a, int b) { return (a + b - 1) / b; }

// -------------------------------
// Vector helpers (128-bit I/O)
// -------------------------------
template <typename T>
__device__ inline bool is_aligned_16(const T* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 0xF) == 0);
}

// ---------------------------------------------
// Core MFMA kernel (parametric)
// MT_M × MT_N macro-tile, BK tile depth, WAVES_R × WAVES_C subtiles (16x16 each)
// A: [M,K] BF16; B: [N,K] F32 (read as B^T[k,n] = B[n*K + k]); C: [N,M] F32
// ---------------------------------------------
template <int MT_M, int MT_N, int BK, int WAVES_R, int WAVES_C>
__launch_bounds__(64 * WAVES_R * WAVES_C, 2) __global__
    void gemm_mfma_core(const __hip_bfloat16* __restrict__ A,    // [M,K] BF16
                        const float* __restrict__ B,             // [N,K] F32
                        float* __restrict__ C,                   // [N,M] F32
                        const __hip_bfloat16* __restrict__ bias, // [M] (optional)
                        int N, int M, int K // note: N is "batch" dimension in your layout
    ) {
    constexpr int WAVES = WAVES_R * WAVES_C; // # 16x16 subtiles per block
    constexpr int TPB = 64 * WAVES;          // threads per block
    static_assert((MT_M % 16) == 0 && (MT_N % 16) == 0, "Tiles must be multiples of 16");
    static_assert(WAVES >= 1 && WAVES <= 16, "WAVES in [1,16]");
    static_assert((MT_M / 16) * (MT_N / 16) == WAVES, "WAVES must cover subtiles");

    const int tid =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; // 0..TPB-1
    const int wave = tid >> 6; // 0..WAVES-1
    const int lane = tid & 63; // 0..63

    // Map wave to its 16x16 subtile in the MT_M x MT_N macro-tile
    const int subtile_r = wave / WAVES_C; // 0..WAVES_R-1
    const int subtile_c = wave % WAVES_C; // 0..WAVES_C-1

    // Block macro-tile origin
    const int n0 = blockIdx.x * MT_N; // along N
    const int m0 = blockIdx.y * MT_M; // along M

    const int n_base = n0 + subtile_c * 16;
    const int m_base = m0 + subtile_r * 16;

    // ---------------------------
    // Shared (LDS) double buffer
    // Pad by +1 on the stride dimension to avoid 32-bank conflicts
    // ---------------------------
    extern __shared__ char smem_gemm[];
    // A: [MT_M, BK+1] (BF16), B: [BK, MT_N+1] (F32)
    __hip_bfloat16* sA0 = reinterpret_cast<__hip_bfloat16*>(smem_gemm);
    __hip_bfloat16* sA1 = sA0 + MT_M * (BK + 1);
    float* sB0 = reinterpret_cast<float*>(sA1 + MT_M * (BK + 1));
    float* sB1 = sB0 + BK * (MT_N + 1);

    using f4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    f4 d = {0.f, 0.f, 0.f, 0.f}; // per-lane accum (4 floats) for 16x16x4f32

    // Lane decomposition for MFMA
    const int lane_x = lane & 15; // 0..15 (column within 16x16)
    const int lane_y = lane >> 4; // 0..3  (selects 4-way k-slice)

    // Helpers
    auto ldA = [&](int m, int k) -> __hip_bfloat16 {
        return (m >= 0 && m < M && k >= 0 && k < K) ? A[(size_t)m * (size_t)K + (size_t)k]
                                                    : __float2bfloat16(0.0f);
    };
    auto ldB = [&](int n, int k) -> float {
        return (n >= 0 && n < N && k >= 0 && k < K) ? B[(size_t)n * (size_t)K + (size_t)k] : 0.0f;
    };

    // Cooperative preload stage 0
    // Total elems: MT_M*(BK) for A, BK*(MT_N) for B
    {
        // A tile  : rows m0..m0+MT_M-1, cols k0..k0+BK-1
        // B tile  : rows k0..k0+BK-1,   cols n0..n0+MT_N-1
        const int elemsA = MT_M * BK;
        const int elemsB = BK * MT_N;

        // A tile: unchanged
        for (int p = tid; p < elemsA; p += TPB) {
            int mi = p / BK;
            int kk = p % BK;
            sA0[mi * (BK + 1) + kk] = ldA(m0 + mi, /*k0=*/0 + kk);
        }

        // B tile: make K the fast index -> ki = p % BK, nj = p / BK
        // This reads B[n*K + (k0+ki)] with adjacent p hitting adjacent K (contiguous)
        for (int p = tid; p < elemsB; p += TPB) {
            int ki = p % BK; // 0..BK-1  (fast)
            int nj = p / BK; // 0..MT_N-1
            sB0[ki * (MT_N + 1) + nj] = ldB(n0 + nj, /*k0=*/0 + ki);
        }
    }
    __syncthreads();

    const int num_k_tiles = (K + BK - 1) / BK;
    int ping = 0;

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        __hip_bfloat16* sA = (ping == 0) ? sA0 : sA1;
        float* sB = (ping == 0) ? sB0 : sB1;

        // Preload next stage while we compute this one
        const int next_k0 = (tile + 1) * BK;
        if (tile + 1 < num_k_tiles) {
            __hip_bfloat16* sA_next = (ping == 0) ? sA1 : sA0;
            float* sB_next = (ping == 0) ? sB1 : sB0;

            const int elemsA = MT_M * BK;
            const int elemsB = BK * MT_N;

            // A next: unchanged
            for (int p = tid; p < elemsA; p += TPB) {
                int mi = p / BK;
                int kk = p % BK;
                sA_next[mi * (BK + 1) + kk] = ldA(m0 + mi, next_k0 + kk);
            }

            // B next: K-contiguous scalar path (correctness-first)
            for (int p = tid; p < elemsB; p += TPB) {
                int ki = p % BK; // 0..BK-1  (fast)
                int nj = p / BK; // 0..MT_N-1
                sB_next[ki * (MT_N + 1) + nj] = ldB(n0 + nj, next_k0 + ki);
            }
        }

// Compute this stage: BK processed in 4-wide MFMA slices
// Each wave computes one 16x16 subtile at (m_base, n_base)
#pragma unroll
        for (int kk4 = 0; kk4 < BK; kk4 += 4) {
            const int k_cur = kk4 + lane_y; // choose 1 of 4 in this group

            // Rows/cols inside MT tile that belong to this wave
            const int a_row = (m_base - m0) + lane_x; // 0..15
            const int b_col = (n_base - n0) + lane_x; // 0..15

            float amk = 0.f, bkn = 0.f;
            if ((unsigned)a_row < (unsigned)MT_M && (unsigned)k_cur < (unsigned)BK)
                amk = __bfloat162float(sA[a_row * (BK + 1) + k_cur]);
            if ((unsigned)k_cur < (unsigned)BK && (unsigned)b_col < (unsigned)MT_N)
                bkn = sB[k_cur * (MT_N + 1) + b_col];

            d = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, d, 0, 0, 0);
        }

        __syncthreads();
        ping ^= 1;
    }

    // Store the 16x16 subtile for this wave (4 rows per lane)
    const int n_out = n_base + (lane & 15);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m_out = m_base + (lane >> 4) * 4 + i;
        if (n_out < N && m_out < M) {
            float out = d[i];
            if (bias)
                out += __bfloat162float(bias[m_out]); // fuse bias
            C[(size_t)n_out * (size_t)M + (size_t)m_out] = out;
        }
    }
}

// ---------------------------------------------
// Launcher helpers: choose tile shapes per case
// ---------------------------------------------
template <int MT_M, int MT_N, int BK, int WR, int WC>
inline void launch_core(float* xout, const float* x_batch, const __hip_bfloat16* w,
                        const __hip_bfloat16* bias, int N, int K, int M) {
    constexpr int WAVES = WR * WC;
    dim3 block(64 * WAVES, 1, 1); // linearized (hip will map properly)
    // For readability, we can set 1D block dim, since kernel internal uses tid 0..TPB-1.
    dim3 grid(ceil_div1(N, MT_N), ceil_div1(M, MT_M), 1);
    size_t shmem_bytes =
        2 * (MT_M * (BK + 1) * sizeof(__hip_bfloat16)) + 2 * (BK * (MT_N + 1) * sizeof(float));

    hipLaunchKernelGGL((gemm_mfma_core<MT_M, MT_N, BK, WR, WC>), grid, block, shmem_bytes, 0, w,
                       x_batch, xout, bias, N, M, K);
    CHECK_HIP(hipGetLastError());
}

// ---------------------------------------------
// Public entry points: tuned for your 4 shapes
// ---------------------------------------------

// 1) M=5120, K=2880, N=128
// Use 64x32 macro-tile ⇒ (64/16)*(32/16) = 4*2 = 8 waves (WR=4, WC=2).
// Good reuse on tall-M, keeps threads/block = 512.
void gemm_case_M5120_K2880_N128(float* xout, const float* x_batch, const __hip_bfloat16* w,
                                const __hip_bfloat16* bias, int N, int K, int M) {
    launch_core<64, 32, 32, 4, 2>(xout, x_batch, w, bias, N, K, M);
}

// 2) M=2880, K=4096, N=128
// K is larger; keep BK=32 first (BK=64 only if your VGPRs still allow ≥2 blocks/CU).
// Same 64x32 tiling (8 waves) performs well for N=128.
void gemm_case_M2880_K4096_N128(float* xout, const float* x_batch, const __hip_bfloat16* w,
                                const __hip_bfloat16* bias, int N, int K, int M) {
    launch_core<64, 32, 32, 4, 2>(xout, x_batch, w, bias, N, K, M);
    // If you confirm VGPR headroom: launch_core<64, 32, 64, 4, 2>(...);
}

// 3) M=32, K=2880, N=128
// Skinny-M: 32x64 macro-tile ⇒ (32/16)*(64/16) = 2*4 = 8 waves (WR=2, WC=4).
// One y-tile covers all rows; grid.x=2 covers N=128.
void gemm_case_M32_K2880_N128(float* xout, const float* x_batch, const __hip_bfloat16* w,
                              const __hip_bfloat16* bias, int N, int K, int M) {
    launch_core<32, 64, 32, 2, 4>(xout, x_batch, w, bias, N, K, M);
}

// 4) M=201088, K=2880, N=128
// Extremely tall: reuse the 64x32 setting (8 waves) so you keep ≥2 resident blocks/CU.
void gemm_case_M201088_K2880_N128(float* xout, const float* x_batch, const __hip_bfloat16* w,
                                  const __hip_bfloat16* bias, int N, int K, int M) {
    launch_core<64, 32, 32, 4, 2>(xout, x_batch, w, bias, N, K, M);
}

// Dispatcher (unchanged logic, now calls the fixed wrappers)
void gemm(float* xout, const float* x_batch, const __hip_bfloat16* w, const __hip_bfloat16* bias,
          int batch_size, int K, int M) {
    const int N = batch_size;
    if (M == 5120 && K == 2880 && N == 128) {
        gemm_case_M5120_K2880_N128(xout, x_batch, w, bias, N, K, M);
        return;
    }
    if (M == 2880 && K == 4096 && N == 128) {
        gemm_case_M2880_K4096_N128(xout, x_batch, w, bias, N, K, M);
        return;
    }
    if (M == 32 && K == 2880 && N == 128) {
        gemm_case_M32_K2880_N128(xout, x_batch, w, bias, N, K, M);
        return;
    }
    if (M == 201088 && K == 2880 && N == 128) {
        gemm_case_M201088_K2880_N128(xout, x_batch, w, bias, N, K, M);
        return;
    }

    // Fallback: safe 32x64 (8 waves)
    launch_core<32, 64, 32, 2, 4>(xout, x_batch, w, bias, N, K, M);
}
