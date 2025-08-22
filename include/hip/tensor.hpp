#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

using namespace std;

/* [Tensor Structure] */
struct Tensor {
    size_t ndim = 0;
    size_t shape[5] = {1, 1, 1, 1, 1};
    float* buf = nullptr;   // Host memory
    float* d_buf = nullptr; // Device memory

    Tensor(const vector<size_t>& shape_);
    Tensor(const vector<size_t>& shape_, float* buf_);
    Tensor();
    ~Tensor();

    size_t num_elem();
    void reshape(const vector<size_t>& shape_);

    // GPU memory management
    void to_device();
    void to_host();
    void clear();
};
