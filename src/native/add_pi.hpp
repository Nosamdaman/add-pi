#pragma once

#include <torch/csrc/stable/tensor.h>

torch::stable::Tensor add_pi_cpu(const torch::stable::Tensor &tensor);

torch::stable::Tensor add_pi_cuda(const torch::stable::Tensor &tensor);
