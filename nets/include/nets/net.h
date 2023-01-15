#pragma once

#include "torch/torch.h"

// struct Net1 : torch::nn::Module {
//     torch::Tensor W, b;

//     Net1(int64_t N, int64_t M) {
//         W = register_parameter("W", torch::randn({N, M}));
//         b = register_parameter("b", torch::randn(M));
//     };
//     torch::Tensor forward(torch::Tensor input) {
//         return torch::addmm(b, input, W);
//     };
// };

// struct Net2 : torch::nn::Module {
//     torch::nn::Linear linear;
//     torch::Tensor another_bias;

//     Net2(int64_t N, int64_t M)
//         : linear(register_module("linear", torch::nn::Linear(N, M))) {
//         another_bias = register_parameter("b", torch::randn(M));
//     }
//     torch::Tensor forward(torch::Tensor input) {
//         return linear(input) + another_bias;
//     }
// };

struct NetImpl : torch::nn::Module {
    torch::nn::Linear fc1, fc2, out;
    NetImpl(int fc1_dim, int fc2_dims) : fc1(fc1_dim, fc1_dim), fc2(fc1_dim, fc2_dims), out(fc2_dims, 1) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("out", out);
    }
    torch::Tensor foward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = out(x);
        return x;
    }
};

TORCH_MODULE(Net);
