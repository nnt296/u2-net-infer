#include "salient_detector.h"
#include "sd_utils.h"

using namespace torch::indexing;

SalientDetector::SalientDetector() = default;

SalientDetector::SalientDetector(const std::string &modelPath, int gpuID) {

#ifdef USE_CUDA
    if (gpuID >= 0) {
        if (torch::cuda::is_available()) {
            // If GPU is available
            if (gpuID < torch::cuda::device_count()) {
                // If gpuID is valid
                std::cout << "CUDA is available, running SD on GPU: " << gpuID << std::endl;
                this->device = torch::Device(torch::kCUDA, (int8_t) gpuID);
            } else
                std::cout << "GPU: " << gpuID << " is not available, switch back to CPU" << std::endl;
        } else
            std::cout << "CUDA isn't available, switch back to CPU" << std::endl;
    }
#endif

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // Make sure ScriptModule is saved on CPU when being exported
        model = torch::jit::load(modelPath);
    }
    catch (const c10::Error &e) {
        std::cerr << "Error loading Segmentation model\n";
    }

    // Move to device
    this->model.to(this->device);
}

SalientDetector::~SalientDetector() = default;

cv::Mat SalientDetector::Infer(cv::Mat &srcImage) {
    // Init result
    cv::Mat scoreMap;

    // Change to evaluation mode to disable Batch norm or dropout
    this->model.eval();
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

cv::Mat SalientDetector::RefineMask(cv::Mat &raw, cv::Mat &rawMask, float threshold, int thickness) {
    cv::Mat srcMask;
    if (rawMask.channels() != 1)
        cv::cvtColor(rawMask, srcMask, cv::COLOR_BGR2GRAY);
    else
        srcMask = rawMask;

    cv::Rect boundingRect = GetBoundingRect(srcMask);

    int offset_top = boundingRect.tl().y;
    int offset_left = boundingRect.tl().x;
    int offset_btm = raw.size().height - boundingRect.br().y;
    int offset_right = raw.size().width - boundingRect.br().x;

    cv::Mat cropped = raw(boundingRect);
    cv::Mat newMask = this->FindBinaryMask(cropped, threshold);

    cv::copyMakeBorder(newMask, newMask, offset_top, offset_btm, offset_left, offset_right,
                       cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat result(newMask.size(), CV_8UC1, cv::Scalar(0));
    VisualizeLargestContour(result, newMask, thickness);

    return result;
}

std::pair<cv::Mat, cv::Mat> SalientDetector::CropMaskByContour(
        cv::Mat &raw, cv::Mat &mask, bool doWarpAffine, float expandRatio) {
    cv::Mat binMask;

    if (mask.channels() != 1)
        cv::cvtColor(mask, binMask, cv::COLOR_BGR2GRAY);
    else
        binMask = mask;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int index = GetMaxAreaContourId(contours);

    if (index < 0)
        throw std::runtime_error("No contours found");

    cv::drawContours(binMask, contours, index, cv::Scalar(255, 255, 255), -1);

    cv::Mat croppedRaw;
    cv::Mat croppedMask;

    if (!doWarpAffine) {
        cv::Rect rect = GetBoundingRect(binMask);

        if (expandRatio > 0 && expandRatio < 1) {
            cv::Point shiftPixel(int((float) rect.width * expandRatio / 2), int((float) rect.height * expandRatio / 2));
            cv::Size newSize(int((float) rect.width * expandRatio), int((float) rect.height * expandRatio));
            // Shift center to tl by shiftPixel
            // Expand rect by adding newSize
            rect -= shiftPixel;
            rect += newSize;
        }

        croppedRaw = raw(rect);
        croppedMask = binMask(rect);
    } else {
        cv::RotatedRect rect = cv::minAreaRect(contours[index]);

        if (expandRatio > 0 && expandRatio < 1) {
            cv::Size2f newSize(rect.size.width * expandRatio, rect.size.height * expandRatio);
            rect.size += newSize;
        }

        cv::Mat M = getRotationMatrix2D(rect.center, rect.angle, 1.0);

        // Assign = copy
        croppedMask = binMask;
        croppedRaw = raw.clone();

        cv::warpAffine(croppedMask, croppedMask, M, croppedMask.size(), cv::INTER_AREA,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::warpAffine(croppedRaw, croppedRaw, M, croppedRaw.size(), cv::INTER_AREA,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::getRectSubPix(croppedMask, rect.size, rect.center, croppedMask);
        cv::getRectSubPix(croppedRaw, rect.size, rect.center, croppedRaw);
    }

    // Convert to 3D for & operation with BGR croppedRaw
    cv::cvtColor(croppedMask, croppedMask, cv::COLOR_GRAY2BGR);

    return std::make_pair(croppedRaw, croppedMask);
}