#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

enum class Device { CPU, CUDA };
enum class DType  { FP32, FP16 };

// Minimal tensor: owns its memory, tracks device placement.
// Layout: row-major, contiguous only (no strides for now).
struct Tensor {
    void*  data    = nullptr;
    size_t numel   = 0;
    Device device  = Device::CPU;
    DType  dtype   = DType::FP32;

    // shape — up to 4 dims
    int    shape[4] = {0, 0, 0, 0};
    int    ndim      = 0;

    Tensor() = default;
    Tensor(int* dims, int ndim, DType dtype, Device device);
    ~Tensor();

    // Non-copyable, movable
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept;
    Tensor& operator=(Tensor&& o) noexcept;

    size_t elem_size() const { return dtype == DType::FP16 ? 2 : 4; }
    size_t bytes()     const { return numel * elem_size(); }

    // Move between devices (returns new tensor)
    Tensor to(Device dst) const;

    // Typed accessors (unsafe — caller ensures correct dtype/device)
    float*  fp32() { return static_cast<float*>(data); }
    __half* fp16() { return static_cast<__half*>(data); }
    const float*  fp32() const { return static_cast<const float*>(data); }
    const __half* fp16() const { return static_cast<const __half*>(data); }
};
