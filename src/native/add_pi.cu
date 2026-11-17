#include "add_pi.hpp"
#include "torch/csrc/stable/macros.h"

#include <numbers>

#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>

constexpr int64_t ADD_PI_BLOCK_SIZE = 256;

__global__ void add_pi_kernel(const double* src, double *dst, int64_t num_elements) {
    const int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements) {
        dst[idx] = src[idx] + std::numbers::pi;
    }
}

torch::stable::Tensor add_pi_cuda(const torch::stable::Tensor &tensor) {
    STD_TORCH_CHECK(tensor.scalar_type() == torch::stable::ScalarType::Double, "\"tensor\" must contain float64 data");
    const auto tensor_contiguous = torch::stable::contiguous(tensor);
    auto output = torch::stable::empty_like(tensor_contiguous);
    auto stream = static_cast<cudaStream_t>(
        torch::stable::accelerator::getCurrentStream(tensor.get_device_index()).nativeHandle()
    );

    const double* tensor_ptr = tensor_contiguous.const_data_ptr<double>();
    double* output_ptr = output.mutable_data_ptr<double>();

    int64_t block_size = ADD_PI_BLOCK_SIZE;
    int64_t num_elements = tensor_contiguous.numel();
    int64_t block_count = (num_elements + block_size - 1) / block_size;
    add_pi_kernel<<<block_count, block_size, 0, stream>>>(tensor_ptr, output_ptr, num_elements);
    STD_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
