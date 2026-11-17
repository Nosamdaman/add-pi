#include "add_pi.hpp"

#include <torch/csrc/stable/ops.h>

torch::stable::Tensor add_pi_cpu(const torch::stable::Tensor &tensor) {
    return torch::stable::empty_like(tensor);
}
