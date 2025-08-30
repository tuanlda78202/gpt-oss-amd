#include "BLAS.hip"

__global__ void rmsnorm_hybrid_kernel(float* o, float* x, __half* weight_half, int size) {
    int lx = threadIdx.x;
    float tmp;
    // Use double precision for sum accumulation to avoid rounding errors
    double ss = 0.0;
    __shared__ float total_sum;

    // Compute sum of squares with double precision
    for (int i = lx; i < size; i += blockDim.x) {
        tmp = x[i];
        ss += (double)tmp * (double)tmp;
    }

    // Reduce with high precision
    float ss_float = (float)ss;
    ss_float = block_reduce_sum(ss_float);

    if (lx == 0) {
        ss_float /= size;
        ss_float += 1e-5f;
        ss_float = 1.0f / sqrtf(ss_float);
        total_sum = ss_float;
    }
    __syncthreads();

    ss_float = total_sum;
    for (int i = lx; i < size; i += blockDim.x) {
        // Convert FP16 weight to FP32 with exact precision
        // __half weight_val = weight_half[i];
        // float weight_fp32 = __half2float(weight_val);
        // Ensure exact computation order for reproducibility
        // float scaled_x = ss_float * x[i];
        // o[i] = weight_fp32 * scaled_x;
        o[i] = __half2float(weight_half[i]) * ss_float * x[i];
    }
}

void rmsnorm_hip_hybrid_device(float* o, float* x, __half* weight, int size) {
    dim3 blockDim(1024);
    dim3 gridDim(1);
    hipLaunchKernelGGL(rmsnorm_hybrid_kernel, gridDim, blockDim, 0, 0, o, x, weight, size);

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}
