#include "../BLAS.hip"

__global__ void thaDNN_s_rmsnorm_kernel_v2(float* o, float* x, float* weight, int size) {
    int lx = threadIdx.x;
    float tmp;
    float ss = 0.0;
    __shared__ float total_sum;

    for (int i = lx; i < size; i += blockDim.x) {
        tmp = x[i];
        ss += tmp * tmp;
    }

    ss = block_reduce_sum(ss);

    if (lx == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        total_sum = ss;
    }
    __syncthreads();

    ss = total_sum;
    for (int i = lx; i < size; i += blockDim.x) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

// '_s_' = single persion (float)
// input: o, x, weight allocated on device
// input: size = 1 -> 16384
void thaDNN_s_rmsnorm_v2(float* o, float* x, float* weight, int size) {
    if (size == 0 || o == nullptr || x == nullptr || weight == nullptr) {
        printf("THABLAS RMSNORM ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    // CHECK_HIP(hipSetDevice(handle.current_gpu_id));
    dim3 blockDim(1024);
    dim3 gridDim(1);
    hipLaunchKernelGGL(thaDNN_s_rmsnorm_kernel_v2, gridDim, blockDim, 0, 0, o, x, weight, size);

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void rmsnorm_hip(float* o, float* x, float* weight, int size) {
    float *o_d, *x_d, *weight_d;
    CHECK_HIP(hipMalloc(&o_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&x_d, size * sizeof(float)));
    CHECK_HIP(hipMalloc(&weight_d, size * sizeof(float)));
    CHECK_HIP(hipMemcpy(x_d, x, size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(weight_d, weight, size * sizeof(float), hipMemcpyHostToDevice));

    thaDNN_s_rmsnorm_v2(o_d, x_d, weight_d, size);

    CHECK_HIP(hipMemcpy(o, o_d, size * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(o_d));
    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(weight_d));
}

// High-precision hybrid RMSNorm kernel for FP16 weights + FP32 activations
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
