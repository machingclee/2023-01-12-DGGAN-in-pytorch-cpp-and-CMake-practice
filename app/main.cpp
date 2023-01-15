#include "nets/dcgan.h"
#include "nets/net.h"
#include <iostream>

// std::string MNIST_FOLDER_LOCATION{"C://Users//user//Repos//C++//2023-01-12-DGGAN-in-pytorch-cpp-and-CMake-practice//mnist"};
// int kNumberOfEpochs = 100;
// int kBatchSize = 64;
// int kNoiseSize = 100;

// auto dataset = torch::data::datasets::MNIST(MNIST_FOLDER_LOCATION)
//                    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
//                    .map(torch::data::transforms::Stack<>());

// auto dataloader = torch::data::make_data_loader(
//     std::move(dataset),
//     torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

using namespace torch;

int main() {
    // torch::optim::Adam generator_optimizer(
    //     DCGAN::generator->parameters(),
    //     torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.999)));
    // torch::optim::Adam discriminator_optimizer(
    //     DCGAN::discriminator->parameters(),
    //     torch::optim::AdamOptions(5e-4).betas(std::make_tuple(0.5, 0.999)));

    // for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    //     int64_t batch_index = 0;
    //     for (torch::data::Example<>& batch : *dataloader) {
    //         // Train discriminator with real images.
    //         DCGAN::discriminator->zero_grad();
    //         torch::Tensor real_images = batch.data;
    //         torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0);
    //         torch::Tensor real_output = DCGAN::discriminator->forward(real_images);
    //         torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
    //         d_loss_real.backward();
    //         // Train discriminator with fake images.
    //         torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
    //         torch::Tensor fake_images = DCGAN::generator->forward(noise);
    //         torch::Tensor fake_labels = torch::zeros(batch.data.size(0));
    //         torch::Tensor fake_output = DCGAN::discriminator->forward(fake_images.detach());
    //         torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
    //         d_loss_fake.backward();
    //         torch::Tensor d_loss = d_loss_real + d_loss_fake;
    //         discriminator_optimizer.step();
    //         // Train generator.
    //         DCGAN::generator->zero_grad();
    //         fake_labels.fill_(1);
    //         fake_output = DCGAN::discriminator->forward(fake_images);
    //         torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
    //         g_loss.backward();
    //         generator_optimizer.step();
    //         std::printf(
    //             "//r[%2ld/%2ld][batch: %3ld] D_loss: %.4f | G_loss: %.4f\n",
    //             epoch,
    //             kNumberOfEpochs,
    //             ++batch_index,
    //             d_loss.item<float>(),
    //             g_loss.item<float>());
    //     }
    // }

    Net mlp(50, 10);
    std::cout << mlp << std::endl;
    Tensor x, output;
    x = torch::randn({2, 50});
    output = mlp->foward(x);
    std::cout << output << std::endl;
}