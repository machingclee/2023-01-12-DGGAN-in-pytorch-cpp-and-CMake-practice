#include "memory"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

int DEFAULT_HEIGHT = 720;
int DEFAULT_WIDTH = 1280;
int IMG_SIZE = 512;

cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model);
torch::jit::Module load_model(std::string model_name);

int main() {
    torch::jit::script::Module module;
    cv::VideoCapture cap(1);

    if (!cap.isOpened()) {
        std::cout << "Cannot open camera\n";
        return 1;
    }

    cv::Mat frame;
    cv::Mat gray;
    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            std::cout << "Can't receive frame (stream end?). Exiting ...\n";
            break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        imshow("live", gray);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}