#include "../../include/hip_helper.hpp"
#include "../BLAS.hip"
#include <algorithm>
#include <cstdio>

// ====== TOP-K KERNEL USING REDUCTION ======
__global__ void topk_reduction_kernel(float* router_score, float* topk_values, int* topk_indices,
                                      int num_experts, int experts_per_token) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ float shared_scores[1024];
    __shared__ int shared_indices[1024];

    // Load data into shared memory
    if (bid * blockDim.x + tid < num_experts) {
        shared_scores[tid] = router_score[bid * blockDim.x + tid];
        shared_indices[tid] = bid * blockDim.x + tid;
    } else {
        shared_scores[tid] = -3.402e+38f; // negative infinity
        shared_indices[tid] = -1;
    }
    __syncthreads();

    // Bitonic sort within block
    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    // Ascending sort
                    if (shared_scores[tid] < shared_scores[ixj]) {
                        float temp_score = shared_scores[tid];
                        int temp_index = shared_indices[tid];
                        shared_scores[tid] = shared_scores[ixj];
                        shared_indices[tid] = shared_indices[ixj];
                        shared_scores[ixj] = temp_score;
                        shared_indices[ixj] = temp_index;
                    }
                } else {
                    // Descending sort
                    if (shared_scores[tid] > shared_scores[ixj]) {
                        float temp_score = shared_scores[tid];
                        int temp_index = shared_indices[tid];
                        shared_scores[tid] = shared_scores[ixj];
                        shared_indices[tid] = shared_indices[ixj];
                        shared_scores[ixj] = temp_score;
                        shared_indices[ixj] = temp_index;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write top-k results for this block
    if (tid < experts_per_token && shared_indices[tid] != -1) {
        int global_idx = bid * experts_per_token + tid;
        if (global_idx < experts_per_token) {
            topk_values[global_idx] = shared_scores[tid];
            topk_indices[global_idx] = shared_indices[tid];
        }
    }
}

// ====== MERGE TOP-K RESULTS KERNEL ======
__global__ void merge_topk_kernel(float* topk_values, int* topk_indices, int num_blocks,
                                  int experts_per_token) {
    int tid = threadIdx.x;

    __shared__ float shared_values[1024];
    __shared__ int shared_indices[1024];

    // Load top-k values from all blocks
    for (int i = tid; i < num_blocks * experts_per_token; i += blockDim.x) {
        shared_values[i] = topk_values[i];
        shared_indices[i] = topk_indices[i];
    }
    __syncthreads();

    // Sort the merged results
    for (int k = 2; k <= num_blocks * experts_per_token; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid && ixj < num_blocks * experts_per_token) {
                if (shared_values[tid] < shared_values[ixj]) {
                    float temp_score = shared_values[tid];
                    int temp_index = shared_indices[tid];
                    shared_values[tid] = shared_values[ixj];
                    shared_indices[tid] = shared_indices[ixj];
                    shared_values[ixj] = temp_score;
                    shared_indices[ixj] = temp_index;
                }
            }
            __syncthreads();
        }
    }

    // Write final top-k results
    if (tid < experts_per_token) {
        topk_values[tid] = shared_values[tid];
        topk_indices[tid] = shared_indices[tid];
    }
}

// ====== SIMPLE TOP-K KERNEL (for smaller datasets) ======
__global__ void simple_topk_kernel(float* router_score, float* topk_values, int* topk_indices,
                                   int num_experts, int experts_per_token) {
    int tid = threadIdx.x;

    __shared__ float shared_scores[1024];
    __shared__ int shared_indices[1024];

    // Load data into shared memory
    if (tid < num_experts) {
        shared_scores[tid] = router_score[tid];
        shared_indices[tid] = tid;
    } else {
        shared_scores[tid] = -3.402e+38f;
        shared_indices[tid] = -1;
    }
    __syncthreads();

    // Simple bubble sort for top-k (efficient for small k)
    for (int i = 0; i < experts_per_token; i++) {
        for (int j = i + 1; j < num_experts; j++) {
            if (shared_scores[i] < shared_scores[j]) {
                // Swap scores
                float temp_score = shared_scores[i];
                shared_scores[i] = shared_scores[j];
                shared_scores[j] = temp_score;

                // Swap indices
                int temp_index = shared_indices[i];
                shared_indices[i] = shared_indices[j];
                shared_indices[j] = temp_index;
            }
        }
    }

    // Write results
    if (tid < experts_per_token) {
        topk_values[tid] = shared_scores[tid];
        topk_indices[tid] = shared_indices[tid];
    }
}

// ====== HOST WRAPPER FUNCTIONS ======

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

    // Allocate device memory
    float *router_score_d, *topk_values_d, *temp_values_d;
    int *topk_indices_d, *temp_indices_d;

    CHECK_HIP(hipMalloc(&router_score_d, num_experts * sizeof(float)));
    CHECK_HIP(hipMalloc(&topk_values_d, experts_per_token * sizeof(float)));
    CHECK_HIP(hipMalloc(&topk_indices_d, experts_per_token * sizeof(int)));

    // Copy input data to device
    CHECK_HIP(hipMemcpy(router_score_d, router_score, num_experts * sizeof(float),
                        hipMemcpyHostToDevice));

    // Choose implementation based on problem size
    if (num_experts <= 1024) {
        // Use simple kernel for smaller datasets
        dim3 blockDim(1024);
        dim3 gridDim(1);
        hipLaunchKernelGGL(simple_topk_kernel, gridDim, blockDim, 0, 0, router_score_d,
                           topk_values_d, topk_indices_d, num_experts, experts_per_token);
    } else {
        // Use reduction-based approach for larger datasets
        int block_size = 1024;
        int num_blocks = (num_experts + block_size - 1) / block_size;

        CHECK_HIP(hipMalloc(&temp_values_d, num_blocks * experts_per_token * sizeof(float)));
        CHECK_HIP(hipMalloc(&temp_indices_d, num_blocks * experts_per_token * sizeof(int)));

        // Step 1: Find top-k in each block
        dim3 blockDim(block_size);
        dim3 gridDim(num_blocks);
        hipLaunchKernelGGL(topk_reduction_kernel, gridDim, blockDim, 0, 0, router_score_d,
                           temp_values_d, temp_indices_d, num_experts, experts_per_token);

        // Step 2: Merge results from all blocks
        dim3 mergeBlockDim(1024);
        dim3 mergeGridDim(1);
        hipLaunchKernelGGL(merge_topk_kernel, mergeGridDim, mergeBlockDim, 0, 0, temp_values_d,
                           temp_indices_d, num_blocks, experts_per_token);

        // Copy final results
        CHECK_HIP(hipMemcpy(topk_values_d, temp_values_d, experts_per_token * sizeof(float),
                            hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(topk_indices_d, temp_indices_d, experts_per_token * sizeof(int),
                            hipMemcpyDeviceToDevice));

        CHECK_HIP(hipFree(temp_values_d));
        CHECK_HIP(hipFree(temp_indices_d));
    }

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
