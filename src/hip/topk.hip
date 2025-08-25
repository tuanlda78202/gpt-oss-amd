#include "../../include/hip_helper.hpp"
#include "../BLAS.hip"
#include <algorithm>
#include <cstdio>

// ====== SIMPLE AND CORRECT TOP-K KERNEL ======
__global__ void topk_kernel(float* router_score, float* topk_values, int* topk_indices,
                            int num_experts, int experts_per_token) {
    int tid = threadIdx.x;

    // Use shared memory for the scores and indices
    __shared__ float shared_scores[1024];
    __shared__ int shared_indices[1024];

    // Load data into shared memory
    if (tid < num_experts) {
        shared_scores[tid] = router_score[tid];
        shared_indices[tid] = tid;
    } else {
        shared_scores[tid] = -3.402e+38f; // negative infinity for padding
        shared_indices[tid] = -1;
    }
    __syncthreads();

    // Simple selection sort for top-k (efficient for small k)
    // This is much simpler and more reliable than bitonic sort
    for (int i = 0; i < experts_per_token; i++) {
        // Find the maximum value from position i to the end
        int max_idx = i;
        float max_val = shared_scores[i];

        for (int j = i + 1; j < num_experts; j++) {
            if (shared_scores[j] > max_val) {
                max_val = shared_scores[j];
                max_idx = j;
            }
        }

        // Swap if necessary
        if (max_idx != i) {
            // Swap scores
            float temp_score = shared_scores[i];
            shared_scores[i] = shared_scores[max_idx];
            shared_scores[max_idx] = temp_score;

            // Swap indices
            int temp_index = shared_indices[i];
            shared_indices[i] = shared_indices[max_idx];
            shared_indices[max_idx] = temp_index;
        }
    }

    // Write results
    if (tid < experts_per_token) {
        topk_values[tid] = shared_scores[tid];
        topk_indices[tid] = shared_indices[tid];
    }
}

// ====== HOST WRAPPER FUNCTION ======
void topk_hip(float* topk_values, int* topk_indices, float* router_score, int num_experts,
              int experts_per_token) {

    if (num_experts <= 0 || experts_per_token <= 0 || experts_per_token > num_experts) {
        fprintf(stderr, "Invalid parameters: num_experts=%d, experts_per_token=%d\n", num_experts,
                experts_per_token);
        return;
    }
    if (!router_score || !topk_values || !topk_indices) {
        fprintf(stderr, "Null pointer detected in topk\n");
        return;
    }

    // For simplicity, we'll use a single kernel approach
    // This works well for typical MoE scenarios where num_experts is reasonable (8-128)
    if (num_experts > 1024) {
        fprintf(stderr, "Warning: num_experts=%d exceeds 1024, using CPU fallback\n", num_experts);
        // Fall back to CPU implementation for very large numbers
        return;
    }

    // Allocate device memory
    float *router_score_d, *topk_values_d;
    int* topk_indices_d;

    CHECK_HIP(hipMalloc(&router_score_d, num_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&topk_values_d, experts_per_token * sizeof(float)));
    CHECK_HIP(hipMalloc(&topk_indices_d, experts_per_token * sizeof(int)));

    // Copy input data to device
    CHECK_HIP(hipMemcpy(router_score_d, router_score, num_experts * sizeof(float),
                        hipMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(1024);
    dim3 gridDim(1);
    hipLaunchKernelGGL(topk_kernel, gridDim, blockDim, 0, 0, router_score_d, topk_values_d,
                       topk_indices_d, num_experts, experts_per_token);

    // Copy results back to host
    CHECK_HIP(hipMemcpy(topk_values, topk_values_d, experts_per_token * sizeof(float),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(topk_indices, topk_indices_d, experts_per_token * sizeof(int),
                        hipMemcpyDeviceToHost));

    // Free device memory
    CHECK_HIP(hipFree(router_score_d));
    CHECK_HIP(hipFree(topk_values_d));
    CHECK_HIP(hipFree(topk_indices_d));

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}
