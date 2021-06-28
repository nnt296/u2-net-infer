#ifndef U2_NET_INFER_SALIENT_DETECTOR_H
#define U2_NET_INFER_SALIENT_DETECTOR_H


#include "torch/script.h"
#include "torch/torch.h"
#include "opencv2/opencv.hpp"


class SalientDetector {
private:
    torch::jit::script::Module model;
    torch::Device device = torch::kCPU;

public:
    explicit SalientDetector(const std::string &modelPath, bool useGPU = false);

    ~SalientDetector();

    cv::Mat FindBinaryMask(cv::Mat &cropImage, float threshold = 0.1);

    cv::Mat Infer(cv::Mat &srcImage);

    static at::Tensor PreProcess(cv::Mat &srcImage);
};


#endif //U2_NET_INFER_SALIENT_DETECTOR_H
