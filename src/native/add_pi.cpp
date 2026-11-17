#include "add_pi.hpp"

#include <numbers>

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>

torch::stable::Tensor add_pi_cpu(const torch::stable::Tensor &tensor) {
    STD_TORCH_CHECK(tensor.scalar_type() == torch::stable::ScalarType::Double, "\"tensor\" must contain float64 data");
    const auto tensor_contiguous = torch::stable::contiguous(tensor);
    auto output = torch::stable::empty_like(tensor_contiguous);

    const double* tensor_ptr = tensor_contiguous.const_data_ptr<double>();
    double* output_ptr = output.mutable_data_ptr<double>();

    for (size_t idx = 0; idx < static_cast<size_t>(tensor_contiguous.numel()); ++idx) {
        output_ptr[idx] = tensor_ptr[idx] + std::numbers::pi;
    }

    return output;
}
