#ifndef FLEXIBLE_ID_VS_COMMON_H
#define FLEXIBLE_ID_VS_COMMON_H


#include <torch/script.h> // One-stop header.
#include "opencv2/opencv.hpp"

// Common datatype
struct DataSample {
    cv::Mat image;
    int label;
};

struct SplitData {
    // Train/Val/Test data;
    std::vector<DataSample> trainSamples;
    std::vector<DataSample> valSamples;
    std::vector<DataSample> testSamples;
};

// Torch utils
at::Tensor Cov(const at::Tensor &input);

float Mahalanobis(const at::Tensor &input, const at::Tensor &mean, const at::Tensor &CovInv);

void SaveTensor(at::Tensor &tensor, const std::string &outputPath);

at::Tensor LoadTensor(const std::string &inputPath);

// Process image
cv::Mat PadImageToSquare(const cv::Mat &image);

at::Tensor PreprocessImage(const cv::Mat &srcImage, bool unsqueeze = false, int resize = 224);

// Convert inputs
at::Tensor ToTensor(cv::Mat &img);

std::vector<torch::jit::IValue> ToInput(at::Tensor tensor_image);

cv::Mat ToCvImage(at::Tensor tensor, int cvType=CV_8UC3);

// Other support functions
std::vector<std::string> ListSubDirectories(const std::string &root);

void ShowImage(cv::Mat &img, std::string title);

cv::Mat GenerateDefect(cv::Mat &srcImage, cv::Mat &mask, float proportion = 8e-2);

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
ROC_Curve(std::vector<int> &binaryLabels, std::vector<float> &scores);

std::tuple<float, float, float, float>
GetThresholds(std::vector<int> &binaryLabels, std::vector<float> &scores);

#endif //FLEXIBLE_ID_VS_COMMON_H
