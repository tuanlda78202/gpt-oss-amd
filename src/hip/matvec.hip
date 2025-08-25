#include "../BLAS.hip"

#define TILE_SIZE 16

__global__ void matvec_tiled(const float* A, const float* B, float* C, int M, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE];

    const int tRow = threadIdx.y;
    const int tCol = threadIdx.x;
    const int row = blockIdx.y * TILE_SIZE + tRow;
    const int col = blockIdx.x * TILE_SIZE + tCol;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load A tile
        if (row < M && k + tCol < K) {
            As[tRow][tCol] = A[row * K + k + tCol];
        } else {
            As[tRow][tCol] = 0.0f;
        }

        // Load B tile
        if (k + tRow < K) {
            Bs[tRow] = B[k + tRow];
        } else {
            Bs[tRow] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[tRow][i] * Bs[i];
        }

        __syncthreads();
    }

    // Write result
    if (row < M) {
        C[row] = sum;
    }
}

void matvec_tiled_hip(const float* A, const float* B, float* C, int M, int K) {
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    hipLaunchKernelGGL(matvec_tiled, grid_dim, block_dim, 0, 0, A, B, C, M, K);

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

// Wrapper function that matches matmul_cpu signature
void matmul_hip(float* xout, float* x, float* w, int n, int d) {
    // Allocate device memory
    float *x_d, *w_d, *xout_d;
    CHECK_HIP(hipMalloc(&x_d, n * sizeof(float)));
    CHECK_HIP(hipMalloc(&w_d, n * d * sizeof(float)));
    CHECK_HIP(hipMalloc(&xout_d, d * sizeof(float)));


    // Copy data to device
    CHECK_HIP(hipMemcpy(x_d, x, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_d, w, n * d * sizeof(float), hipMemcpyHostToDevice));

    CHECK_HIP(hipMemset(xout_d, 0, d * sizeof(float)));


    // Call the HIP kernel
    matvec_tiled_hip(w_d, x_d, xout_d, d, n);

    // Copy result back to host
    CHECK_HIP(hipMemcpy(xout, xout_d, d * sizeof(float), hipMemcpyDeviceToHost));

    // Free device memory
    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(w_d));
    CHECK_HIP(hipFree(xout_d));
}
