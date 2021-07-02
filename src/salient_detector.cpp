#include "salient_detector.h"
#include "sd_utils.h"

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
        model = torch::jit::load(modelPath);
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
        d1 = d1.cpu();

        scoreMap = ToCvImage(d1, CV_32FC1);
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
    auto mean = at::tensor(at::ArrayRef<float>({0.485, 0.456, 0.406}));
    auto std = at::tensor(at::ArrayRef<float>({0.229, 0.224, 0.225}));
    auto maxV = torch::max(tensor);

    tensor = tensor.toType(at::kFloat).div(maxV).sub(mean).div(std);

    // Swap axis
    tensor = tensor.permute({(2), (0), (1)});

    // Add batch dim (an inplace operation just like in pytorch)
    tensor.unsqueeze_(0);

    return tensor;
}

cv::Mat SalientDetector::FindBinaryMask(cv::Mat &cropImage, float threshold) {
    // First enlarge crop image
    int w = cropImage.cols;
    int h = cropImage.rows;

    cv::Mat enlarged;
    cv::copyMakeBorder(cropImage, enlarged, h / 2, h / 2, w / 2, w / 2, cv::BORDER_REPLICATE);

    // Run inference
    cv::Mat probMap = this->Infer(enlarged);

    // Resize to enlarged
    cv::resize(probMap, probMap, enlarged.size(), 0, 0, cv::INTER_LANCZOS4);

    // Threshold for 1-0 value
    cv::Mat threshIm;
    cv::threshold(probMap, threshIm, threshold, 1., cv::THRESH_BINARY);

    // Convert to 8UC1
    threshIm = threshIm * 255;
    threshIm.convertTo(threshIm, CV_8UC1);

    // Find & visualize contours
    std::vector<std::vector<cv::Point> > contours;
    findContours(threshIm, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Naive check if cannot find any salient object
    if (contours.empty())
        return cv::Mat();

    int maxIdx = GetMaxAreaContourId(contours);
    cv::drawContours(threshIm, contours, maxIdx, cv::Scalar(255), cv::FILLED);

    cv::Rect roi(w / 2, h / 2, w, h);
    threshIm = threshIm(roi);

    return threshIm;
}

cv::Mat SalientDetector::DilateBinaryMask(cv::Mat &binaryMask, float dilateRatio) {
    cv::Mat out = binaryMask.clone();

    // Dilate mask
    int maskWidth = binaryMask.cols;
    int maskHeight = binaryMask.rows;
    int maxSize = (maskWidth > maskHeight) ? maskWidth : maskHeight;

    // Kernel size of max(10 % max_size, 10 pixels)
    int kernelSize = int((float) maxSize * dilateRatio);

    // If 0 or negative size, return raw
    if (kernelSize < 1)
        return out;

    // Dilate with rectangular structure, so that the bordering will
    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::dilate(binaryMask, out, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT,
               cv::morphologyDefaultBorderValue());

    return out;
}