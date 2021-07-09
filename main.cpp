#include "opencv2/opencv.hpp"
#include "torch/script.h"
#include "src/salient_detector.h"

using namespace torch::indexing;


int main() {
    SalientDetector sd("../models/u2net.pt", true);

    cv::Mat mask = cv::imread("../models/drawn.png");
    cv::Mat im = cv::imread("../models/raw.bmp");

    cv::resize(mask, mask, cv::Size(), 0.3, 0.3, cv::INTER_CUBIC);
    cv::resize(im, im, cv::Size(), 0.3, 0.3, cv::INTER_CUBIC);

    auto out = sd.RefineMask(im, mask);

    cv::Mat oRaw, oMask;

    std::tie(oRaw, oMask) = SalientDetector::CropMaskByContour(im, out, 0.08);
    // Disable dilate
    // std::tie(oRaw, oMask) = SalientDetector::CropMaskByContour(im, out, 0);

    cv::imshow("out raw", oRaw);
    cv::imshow("out mask", oMask);

    oMask = SalientDetector::DilateBinaryMask(oMask, 0.05);
    cv::imshow("out mask dilate", oMask);

    cv::waitKey(0);
}