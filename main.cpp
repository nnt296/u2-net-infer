#include "opencv2/opencv.hpp"
#include "torch/script.h"
#include <torch/torch.h>

#include "src/common.h"
#include "src/salient_detector.h"


using namespace torch::indexing;


int main() {
    SalientDetector sd("../models/u2net.pt", true);
    cv::Mat im = cv::imread("../models/crop_image.png");
    // cv::resize(im, im, cv::Size(), 0.3, 0.3, cv::INTER_CUBIC);
    // auto output = sd.Infer(im);
    auto mask = sd.FindBinaryMask(im, 0.1);
    cv::imshow("mask", mask);
    cv::imwrite("../models/output_mask.png", mask);
    cv::waitKey(0);
}