#include "opencv2/opencv.hpp"
#include "salient_detector.h"

using namespace torch::indexing;


int main() {
    SalientDetector sd("/mnt/MinusAsian/u2-net-infer/models/u2net.pt", true);
    cv::Mat im = cv::imread("/mnt/MinusAsian/u2-net-infer/models/crop_image.png");
    // cv::resize(im, im, cv::Size(), 0.3, 0.3, cv::INTER_CUBIC);
    // auto mask = sd.Infer(im);
    auto mask = sd.FindBinaryMask(im, 0.1);
    cv::imshow("mask", mask);

    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    cv::Mat object = mask & im;
    cv::imshow("object", object);

    cv::imwrite("../models/output_mask.png", mask);
    cv::waitKey(0);
}