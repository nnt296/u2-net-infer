#include "opencv2/opencv.hpp"
#include "torch/script.h"
#include <torch/torch.h>

#include "src/common.h"
#include "src/salient_detector.h"


using namespace torch::indexing;


int main() {
    SalientDetector sd("../models/u2net.pt", true);
    cv::Mat im = cv::imread("/mnt/MinusAsian/Datasets/image_border.png");
    // SalientDetector::PreProcess(im);
    auto output = sd.Infer(im);

    cv::imshow("image", output);
    cv::waitKey(0);

    std::cout << output.size() << std::endl;
}