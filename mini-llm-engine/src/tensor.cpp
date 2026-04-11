#include "tensor.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

static size_t calc_numel(int* dims, int ndim) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= (size_t)dims[i];
    return n;
}

Tensor::Tensor(int* dims, int ndim_, DType dtype_, Device device_)
    : numel(calc_numel(dims, ndim_)), device(device_), dtype(dtype_), ndim(ndim_) {
    for (int i = 0; i < ndim; i++) shape[i] = dims[i];
    size_t nb = bytes();
    if (device == Device::CPU) {
        data = malloc(nb);
        if (!data) throw std::bad_alloc();
        memset(data, 0, nb);
    } else {
        cudaError_t err = cudaMalloc(&data, nb);
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
        cudaMemset(data, 0, nb);
    }
}

Tensor::~Tensor() {
    if (!data) return;
    if (device == Device::CPU) free(data);
    else                       cudaFree(data);
}

Tensor::Tensor(Tensor&& o) noexcept
    : data(o.data), numel(o.numel), device(o.device), dtype(o.dtype), ndim(o.ndim) {
    for (int i = 0; i < ndim; i++) shape[i] = o.shape[i];
    o.data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this == &o) return *this;
    this->~Tensor();
    data = o.data; numel = o.numel; device = o.device; dtype = o.dtype; ndim = o.ndim;
    for (int i = 0; i < ndim; i++) shape[i] = o.shape[i];
    o.data = nullptr;
    return *this;
}

Tensor Tensor::to(Device dst) const {
    if (dst == device) {
        // Return a copy
        Tensor t(const_cast<int*>(shape), ndim, dtype, dst);
        if (dst == Device::CPU) memcpy(t.data, data, bytes());
        else cudaMemcpy(t.data, data, bytes(), cudaMemcpyDeviceToDevice);
        return t;
    }
    Tensor t(const_cast<int*>(shape), ndim, dtype, dst);
    if (dst == Device::CUDA)
        cudaMemcpy(t.data, data, bytes(), cudaMemcpyHostToDevice);
    else
        cudaMemcpy(t.data, data, bytes(), cudaMemcpyDeviceToHost);
    return t;
}
