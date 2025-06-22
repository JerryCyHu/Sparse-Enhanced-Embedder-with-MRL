#include <torch/extension.h>

void launch_relu_scatter(
    at::Tensor hidden, at::Tensor weight,
    at::Tensor token_id, at::Tensor output);

at::Tensor relu_amax_scatter_forward(
    at::Tensor hidden,
    at::Tensor weight,
    at::Tensor token_id)
{

    auto B = hidden.size(0);
    auto vocab =  (int64_t) token_id.max().item<int64_t>() + 1;  // or pass in
    auto output = at::zeros({B, vocab}, hidden.options());

    launch_relu_scatter(hidden, weight, token_id, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relu_amax_scatter_forward,
          "ReLU + amax scatter (CUDA)");
}
