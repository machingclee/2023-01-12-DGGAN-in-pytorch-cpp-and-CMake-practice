#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <string>
#include <vector>

std::string model_pt_path{"C:\\Users\\user\\Repos\\C++\\2023-01-12-DGGAN-in-pytorch-cpp-and-CMake-practice\\models\\net.pt"};

int main() {
    torch::jit::script::Module net = torch::jit::load(model_pt_path);
    torch::Tensor x = torch::randn({1, 100});
    torch::Tensor y = torch::randn({1, 100});
    torch::Tensor inputs = torch::cat({x, y});
    std::vector<torch::IValue> x_{inputs};
    torch::Tensor yTensor = net.forward(x_).toTensor();
    size_t ySize = yTensor.sizes()[0];
    float* yDataPtr = (float*)yTensor.data_ptr();
    try {
        // float result = output.toTensor().item<float>();
        for (int i = 0; i < ySize; i++) {
            float value = yDataPtr[i];
            std::cout << "The IValue output: " << value << std::endl;
        }
        //   << "The float output: " << result << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << e.msg() << std::endl;
    }
}