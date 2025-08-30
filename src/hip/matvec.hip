#include "BLAS.hip"

#define TILE_SIZE 16
#define SCALE 4
#define BLOCK_SIZE (TILE_SIZE * SCALE)
#define PAD_SIZE (BLOCK_SIZE + 16)

__global__ void matvec_hybrid_tiled(const __half* A_half, const float* B, float* C, __half* bias, int M, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE];

    const int tRow = threadIdx.y;
    const int tCol = threadIdx.x;
    const int row = blockIdx.y * TILE_SIZE + tRow;
    const int col = blockIdx.x * TILE_SIZE + tCol;

    // Use double precision for accumulation to avoid rounding errors
    double sum = 0.0;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load A tile and convert FP16 -> FP32 with careful rounding
        if (row < M && k + tCol < K) {
            // Ensure exact FP16→FP32 conversion
            __half half_val = A_half[row * K + k + tCol];
            As[tRow][tCol] = __half2float(half_val);
        } else {
            As[tRow][tCol] = 0.0f;
        }

        // Load B tile (already FP32)
        if (k + tRow < K) {
            Bs[tRow] = B[k + tRow];
        } else {
            Bs[tRow] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum with double precision accumulation
        for (int i = 0; i < TILE_SIZE; ++i) {
            // Use double precision multiplication to maintain exact precision
            sum += (double)As[tRow][i] * (double)Bs[i];
        }

        __syncthreads();
    }

    // Convert back to FP32 with exact rounding
    if (row < M) {
        float bias_val = bias ? __half2float(bias[row]) : 0.0f;
        C[row] = (float)sum + bias_val;
    }
}

void matmul_gpu(float* xout, float* x, __half* w, __half* bias, int n, int d) {
    // Initialize output to zero for exact accumulation
    CHECK_HIP(hipMemset(xout, 0, d * sizeof(float)));

    // CRITICAL FIX: Grid dimensions should match matvec_tiled_hip function
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(1, (d + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(matvec_hybrid_tiled, grid_dim, block_dim, 0, 0, w, x, xout, bias, d, n);

    CHECK_HIP(hipGetLastError());
    // Add synchronization to ensure completion before next operation
    CHECK_HIP(hipDeviceSynchronize());
}

__global__ void matvec_hybrid_vectorized(const __half* __restrict__ A_half,
                                         const float* __restrict__ B, float* __restrict__ C,
                                         const __half* __restrict__ bias, int M, int K) {
    __shared__ float As[BLOCK_SIZE * PAD_SIZE];
    __shared__ float Bs[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int row_off = blockIdx.x * BLOCK_SIZE;

    float sum[SCALE] = {0.0f};
    float a[SCALE];

    // Load B tile into shared memory (broadcasted across rows)
    for (int k = tid; k < BLOCK_SIZE; k += blockDim.x) {
        int gk = k + 0;
        Bs[k] = (gk < K) ? B[gk] : 0.0f;
    }

    __syncthreads();

    for (int sk = 0; sk < K; sk += BLOCK_SIZE) {
        const __half* A_tile = A_half + row_off * K + sk;

        // Load A tile with FP16 → FP32 conversion
        for (int i = tid; i < BLOCK_SIZE; i += blockDim.x) {
            int row = row_off + i;
            if (row < M) {
                if (sk + SCALE - 1 < K) {
                    float4 a_vec;
                    a_vec.x = (row < M && sk + 0 < K) ? __half2float(A_tile[i * K + 0]) : 0.0f;
                    a_vec.y = (row < M && sk + 1 < K) ? __half2float(A_tile[i * K + 1]) : 0.0f;
                    a_vec.z = (row < M && sk + 2 < K) ? __half2float(A_tile[i * K + 2]) : 0.0f;
                    a_vec.w = (row < M && sk + 3 < K) ? __half2float(A_tile[i * K + 3]) : 0.0f;
                    *reinterpret_cast<float4*>(&As[i * PAD_SIZE]) = a_vec;
                } else {
                    for (int kk = 0; kk < SCALE; ++kk) {
                        int gk = sk + kk;
                        if (gk < K) {
                            As[i * PAD_SIZE + kk] = __half2float(A_tile[i * K + kk]);
                        } else {
                            As[i * PAD_SIZE + kk] = 0.0f;
                        }
                    }
                }
            } else {
                for (int kk = 0; kk < SCALE; ++kk)
                    As[i * PAD_SIZE + kk] = 0.0f;
            }
        }

        __syncthreads();

        // Compute for SCALE rows per thread
        for (int i = 0; i < SCALE; ++i) {
            int row = row_off + tid + i * TILE_SIZE;
            if (row >= M)
                continue;

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                float a_val = As[(tid + i * TILE_SIZE) * PAD_SIZE + k];
                float b_val = (sk + k < K) ? Bs[k] : 0.0f;
                sum[i] += a_val * b_val;
            }
        }

        __syncthreads();
    }

    // Write results with optional bias
    for (int i = 0; i < SCALE; ++i) {
        int row = row_off + tid + i * TILE_SIZE;
        if (row < M) {
            float bias_val = bias ? __half2float(bias[row]) : 0.0f;
            C[row] = sum[i] + bias_val;
        }
    }
}

void matmul_hybrid_vectorized(float* xout, float* x, __half* w, __half* bias, int n, int d) {
    // Initialize output to zero for exact accumulation
    CHECK_HIP(hipMemset(xout, 0, d * sizeof(float)));

    // CRITICAL FIX: Grid dimensions should match matvec_tiled_hip function
    dim3 block_dim(TILE_SIZE);
    dim3 grid_dim((d + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matvec_hybrid_vectorized<<<grid_dim, block_dim>>>(w, x, xout, bias, d, n);

    CHECK_HIP(hipGetLastError());
    // Add synchronization to ensure completion before next operation
    CHECK_HIP(hipDeviceSynchronize());
}
