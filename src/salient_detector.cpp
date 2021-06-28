#include "salient_detector.h"
#include "common.h"

using namespace torch::indexing;

SalientDetector::SalientDetector(const std::string &modelPath, bool useGPU) {
    if (useGPU) {
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available!" << std::endl;
            this->device = torch::kCUDA;
        } else {
            std::cout << "CUDA is not available! Switch back to default CPU" << std::endl;
        }
    }

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // Make sure ScriptModule is saved on CPU when being exported
        model = torch::jit::load("../models/u2net.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading Segmentation model\n";
    }
}

SalientDetector::~SalientDetector() = default;

cv::Mat SalientDetector::Infer(cv::Mat &srcImage) {
    // Init result
    cv::Mat scoreMap;

    // Change to evaluation mode to disable Batch norm or dropout
    this->model.eval();
    this->model.to(this->device);
    // Disable gradient for memory saving.
    // This mode equivalent to "with torch.no_grad()" in python
    {
        torch::NoGradGuard no_grad;

        // Preprocess image
        auto tensorImage = PreProcess(srcImage);

        // Check if image cannot be found
        if (!tensorImage.numel()) {
            return cv::Mat();
        }

        // Move to device
        tensorImage = tensorImage.to(this->device);
        std::vector<torch::jit::IValue> testInputs = ToInput(tensorImage);
        auto tupleOutput = this->model.forward(testInputs);

        auto d1 = tupleOutput.toTuple()->elements()[0].toTensor();
        d1.squeeze_();
        auto maxV = torch::max(d1);
        auto minV = torch::min(d1);

        d1 = (d1 - minV) / (maxV - minV);

        scoreMap = ToCvImage(d1.cpu(), CV_32FC1);
        // cv::resize(scoreMap, scoreMap, srcImage.size(), 0, 0, cv::INTER_LINEAR);
    }

    return scoreMap;
}

at::Tensor SalientDetector::PreProcess(cv::Mat &srcImage) {
    // Resize
    cv::Mat processedIm;
    cv::resize(srcImage, processedIm, cv::Size(320, 320), 0, 0, cv::INTER_CUBIC);

    // Convert to BGR
    // Assume srcImage is 3 channels BGR image
    cv::cvtColor(processedIm, processedIm, cv::COLOR_BGR2RGB);

    // To Tensor, expect uint8 opencv image
    at::Tensor tensor = ToTensor(processedIm);


    // Normalize, output normalized tensor float32
    auto mean = at::tensor(at::ArrayRef < float > ({ 0.485, 0.456, 0.406 }));
    auto std = at::tensor(at::ArrayRef < float > ({ 0.229, 0.224, 0.225 }));
    auto maxV = torch::max(tensor);

    tensor = tensor.toType(at::kFloat).div(maxV).sub(mean).div(std);

    // Swap axis
    tensor = tensor.permute({(2), (0), (1)});

    // Add batch dim (an inplace operation just like in pytorch)
    tensor.unsqueeze_(0);

    return tensor;
}
