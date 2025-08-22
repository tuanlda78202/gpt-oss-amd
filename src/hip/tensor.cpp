#include "../../include/hip/tensor.hpp"

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(const vector<size_t>& shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
        shape[i] = shape_[i];
    }
    size_t N_ = num_elem();
    buf = (float*)calloc(N_, sizeof(float));
    CHECK_CUDA(hipMalloc(&d_buf, N_ * sizeof(float)));
}

Tensor::Tensor(const vector<size_t>& shape_, float* buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
        shape[i] = shape_[i];
    }
    size_t N_ = num_elem();
    buf = (float*)malloc(N_ * sizeof(float));
    memcpy(buf, buf_, N_ * sizeof(float));
    CHECK_CUDA(hipMalloc(&d_buf, N_ * sizeof(float)));
    CHECK_CUDA(hipMemcpy(d_buf, buf_, N_ * sizeof(float), hipMemcpyHostToDevice));
}

Tensor::Tensor() {}

Tensor::~Tensor() {
    if (buf != nullptr)
        free(buf);
    if (d_buf != nullptr) {

        CHECK_CUDA(hipFree(d_buf));
    }
}

size_t Tensor::num_elem() {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

void Tensor::reshape(const vector<size_t>& shape_) {
    size_t n = 1;
    ndim = shape_.size(); // ndim<=5
    for (size_t i = 0; i < ndim; i++) {
        shape[i] = shape_[i];
        n *= shape[i];
    }
}

void Tensor::to_device() {
    size_t N_ = num_elem();
    if (d_buf == nullptr) {
        CHECK_CUDA(hipMalloc(&d_buf, N_ * sizeof(float)));
        if (buf != nullptr) {
            CHECK_CUDA(hipMemcpy(d_buf, buf, N_ * sizeof(float), hipMemcpyHostToDevice));
        }
    }
}

void Tensor::to_host() {
    if (d_buf != nullptr) {
        size_t N_ = num_elem();
        if (buf != nullptr) {
            CHECK_CUDA(hipMemcpy(buf, d_buf, N_ * sizeof(float), hipMemcpyDeviceToHost));
        }
    }
}

void Tensor::clear() {
    if (buf != nullptr) {
        memset(buf, 0, num_elem() * sizeof(float));
    }
    if (d_buf != nullptr) {
        CHECK_CUDA(hipMemset(d_buf, 0, num_elem() * sizeof(float)));
    }
}