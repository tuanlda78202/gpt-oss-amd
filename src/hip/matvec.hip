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

    double sum = 0.0;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load A tile and convert FP16 -> FP32
        if (row < M && k + tCol < K) {
            __half half_val = A_half[row * K + k + tCol];
            As[tRow][tCol] = __half2float(half_val);
        } else {
            As[tRow][tCol] = 0.0f;
        }

        // Load B tile (FP32)
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

    // Convert back to FP32
    if (row < M) {
        float bias_val = bias ? __half2float(bias[row]) : 0.0f;
        C[row] = (float)sum + bias_val;
    }
}

void matvec_gpu(float* xout, float* x, __half* w, __half* bias, int n, int d) {
    CHECK_HIP(hipMemset(xout, 0, d * sizeof(float)));

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(1, (d + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(matvec_hybrid_tiled, grid_dim, block_dim, 0, 0, w, x, xout, bias, d, n);

    CHECK_HIP(hipGetLastError());
}
