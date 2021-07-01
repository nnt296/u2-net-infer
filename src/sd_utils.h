#ifndef FLEXIBLE_ID_VS_COMMON_H
#define FLEXIBLE_ID_VS_COMMON_H


#include <torch/script.h>
#include "opencv2/opencv.hpp"

// Convert inputs
at::Tensor ToTensor(cv::Mat &img);

std::vector<torch::jit::IValue> ToInput(at::Tensor tensor_image);

cv::Mat ToCvImage(at::Tensor &tensor, int cvType=CV_8UC3);

int GetMaxAreaContourId(std::vector<std::vector<cv::Point>> contours);

#endif //FLEXIBLE_ID_VS_COMMON_H
