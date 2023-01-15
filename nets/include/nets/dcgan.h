#pragma once

#include "torch/torch.h"

namespace DCGAN {
using namespace torch;
nn::Sequential generator(
    // Layer 1
    nn::ConvTranspose2d(nn::ConvTranspose2dOptions(100, 256, 4)
                            .bias(false)),
    nn::BatchNorm2d(256),
    nn::Functional(torch::relu),
    // Layer 2
    nn::ConvTranspose2d(nn::ConvTranspose2dOptions(256, 128, 3)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
    nn::BatchNorm2d(128),
    nn::Functional(torch::relu),
    // Layer 3
    nn::ConvTranspose2d(nn::ConvTranspose2dOptions(128, 64, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
    nn::BatchNorm2d(64),
    nn::Functional(torch::relu),
    // Layer 4
    nn::ConvTranspose2d(nn::ConvTranspose2dOptions(64, 1, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
    nn::Functional(torch::tanh));

nn::Sequential discriminator(
    // Layer 1
    nn::Conv2d(
        nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 2
    nn::Conv2d(
        nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
    nn::BatchNorm2d(128),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 3
    nn::Conv2d(
        nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
    nn::BatchNorm2d(256),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 4
    nn::Conv2d(
        nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
    nn::Functional(torch::sigmoid));

}