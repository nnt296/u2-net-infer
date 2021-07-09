#include "opencv2/opencv.hpp"
#include "salient_detector.h"

using namespace torch::indexing;


int main() {
    // Path/To/model_segment.pt
    SalientDetector sd("../models/u2net.pt", true);

    // User's drawn image
    cv::Mat mask = cv::imread("../models/drawn.png");
    // Raw image on which user draws
    cv::Mat im = cv::imread("../models/raw.bmp");

    // THIS RESIZE FOR VISUALIZE ONLY
    cv::resize(mask, mask, cv::Size(), 0.3, 0.3, cv::INTER_CUBIC);
    cv::resize(im, im, cv::Size(), 0.3, 0.3, cv::INTER_CUBIC);

    // Call this function to get refined boundary mask
    // User recheck the mask & re-draw
    // User's re-draw image will be the final mask
    auto out = sd.RefineMask(im, mask);

    // Input final mask to get "final cropped image & cropped mask"
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