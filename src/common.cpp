#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

#include <random>
#include "common.h"
#include "boost/filesystem.hpp"

cv::Mat PadImageToSquare(const cv::Mat &image) {
    // Pad image to square without and return to newly created image
    int longSize = std::max(image.size().width, image.size().height);
    int deltaWidth = longSize - image.size().width;
    int deltaHeight = longSize - image.size().height;
    int top = deltaHeight / 2;
    int bottom = deltaHeight - top;
    int left = deltaWidth / 2;
    int right = deltaWidth - left;
    cv::Mat paddedImage{};
    cv::copyMakeBorder(image, paddedImage, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return paddedImage;
}

at::Tensor PreprocessImage(const cv::Mat &srcImage, bool unsqueeze, int resize) {
    // Pad object to square with border constant black, expect image templateWidth x templateHeight
    auto object = PadImageToSquare(srcImage);

    // Output opencv image of type uint8 with shape 224x224x3 in RBG format
    cv::resize(object, object, cv::Size(resize, resize), 0, 0, cv::INTER_AREA);
    cv::cvtColor(object, object, cv::COLOR_BGR2RGB);

    // To Tensor, expect uint8 opencv image
    at::Tensor tensor = ToTensor(object);

    // Normalize, output normalized tensor float32
    auto mean = at::tensor(at::ArrayRef < float > ({ 0.485, 0.456, 0.406 }));
    auto std = at::tensor(at::ArrayRef < float > ({ 0.229, 0.224, 0.225 }));
    tensor = tensor.toType(at::kFloat).div(255).sub(mean).div(std);

    // swap axis
    tensor = tensor.permute({(2), (0), (1)});

    if (unsqueeze)
        // add batch dim (an inplace operation just like in pytorch)
        tensor.unsqueeze_(0);

    return tensor;
}

void SaveTensor(at::Tensor &tensor, const std::string &outputPath) {
    auto bytes = torch::jit::pickle_save(tensor);
    std::ofstream fOut(outputPath, std::ios::out | std::ios::binary);
    fOut.write(bytes.data(), bytes.size());
    fOut.close();
}

at::Tensor LoadTensor(const std::string &inputPath) {
    // COPY from
    // https://stackoverflow.com/questions/15138353/how-to-read-a-binary-file-into-a-vector-of-unsigned-chars/21802936

    // FIXME take ~ 3 minutes to load 2.5G tensor
    // open the file, and does not need to manually close with file.close()
    std::ifstream file(inputPath, std::ios::binary);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    std::vector<char> vecBytes;
    vecBytes.reserve(fileSize);

    // read the data:
    vecBytes.insert(vecBytes.begin(), std::istream_iterator<char>(file), std::istream_iterator<char>());

    auto tensor = torch::jit::pickle_load(vecBytes).toTensor();
    return tensor;
}

at::Tensor Cov(const at::Tensor &input) {
    // Match output with numpy.cov(x, rowvar=false)
    auto tensor = input.transpose(-1, -2);
    tensor = tensor - tensor.mean(-1, true);
    double factor = 1.0 / (double) (tensor.size(-1) - 1);
    at::Tensor cov = factor * torch::matmul(tensor, tensor.transpose(-1, -2).conj());
    return cov;
}

float Mahalanobis(const at::Tensor &input, const at::Tensor &mean, const at::Tensor &CovInv) {
    // Match output with scipy.spatial.distance.mahalanobis function
    at::Tensor delta = input - mean;
    at::Tensor distance = torch::dot(delta, torch::matmul(CovInv, delta));
    return torch::sqrt(distance).item().toFloat();
}

std::string get_image_type(const cv::Mat &img) {
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar numChannels = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
    }

    r += "C";
    r += (std::to_string(numChannels) + '0');

#ifdef DEBUG
    std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;
#endif
    return r;
}

void ShowImage(cv::Mat &img, std::string title) {
    std::string image_type = get_image_type(img);
    cv::imshow(title + " type:" + image_type, img);
    cv::waitKey(0);
}

at::Tensor ToTensor(cv::Mat &image) {
#ifdef DEBUG
    std::cout << "image shape: " << image.size() << std::endl;
#endif
    // Convert the image and label to a tensor.
    // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
    // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
    // would be de-allocated after leaving the scope of this get method, which results in undefined behavior.
    at::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, 3}, at::kByte).clone();

    return tensor_image;
}

std::vector<torch::jit::IValue> ToInput(at::Tensor tensor_image) {
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

cv::Mat ToCvImage(at::Tensor tensor, int cvType) {
    int height = tensor.sizes()[0];
    int width = tensor.sizes()[1];
    try {
        switch (cvType) {
            case CV_8UC3: {
                cv::Mat output_mat(cv::Size{width, height}, CV_8UC3, tensor.data_ptr<uchar>());
                return output_mat.clone();
            }
            case CV_8UC1: {
                cv::Mat output_mat(cv::Size{width, height}, CV_8UC1, tensor.data_ptr<uchar>());
                return output_mat.clone();
            }
            case CV_32FC3: {
                cv::Mat output_mat(cv::Size{width, height}, CV_32FC3, tensor.data_ptr<float>());
                return output_mat.clone();
            }
            case CV_32FC1: {
                cv::Mat output_mat(cv::Size{width, height}, CV_32FC1, tensor.data_ptr<float>());
                return output_mat.clone();
            }
            default:
                return cv::Mat(height, width, CV_8UC3);
        }
    }
    catch (const c10::Error &e) {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }

    return cv::Mat(height, width, CV_8UC3);
}

std::vector<std::string> ListSubDirectories(const std::string &root) {
    std::vector<std::string> subDirectories;
    for (auto &p : boost::filesystem::recursive_directory_iterator(root))
        if (boost::filesystem::is_directory(p))
            subDirectories.push_back(p.path().string());
    return subDirectories;
}

cv::Mat GenerateDefect(cv::Mat &srcImage, cv::Mat &mask, float proportion) {
    // Set random seed
    std::random_device rd;
    std::mt19937 eng(rd());

    auto genImage = srcImage.clone();

    if (mask.channels() > 2)
        throw std::runtime_error("Mask must be binary mask");

    cv::Mat coordinates;
    cv::findNonZero(mask, coordinates);

    int count = 0;
    for (int i = 0; i < coordinates.total(); i++)
        count++;

    std::uniform_int_distribution<> centerDist(0, coordinates.total()); // define the range

    int position = centerDist(eng);

    int c_x = coordinates.at<cv::Point>(position).x;
    int c_y = coordinates.at<cv::Point>(position).y;

    int h = srcImage.size().height;
    int w = srcImage.size().width;

    std::uniform_real_distribution<> hDist(h * proportion / 2, h * proportion); // define the range
    std::uniform_real_distribution<> wDist(w * proportion / 2, w * proportion); // define the range
    std::uniform_real_distribution<> angleDist(0, 179); // define the range

    float r_h = hDist(eng);
    float r_w = wDist(eng);
    float angle = angleDist(eng);

    cv::RotatedRect rect(cv::Point(c_x, c_y), cv::Size(r_h, r_w), angle);
    cv::Point2f vertices2f[4];
    rect.points(vertices2f);

    // Convert them so we can use them in a fillConvexPoly
    cv::Point vertices[4];
    for (int i = 0; i < 4; ++i) {
        vertices[i] = vertices2f[i];
    }
    cv::fillConvexPoly(genImage, vertices, 4, cv::Scalar(128, 128, 128));
    return genImage;
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
ROC_Curve(std::vector<int> &binaryLabels, std::vector<float> &scores) {
    if (binaryLabels.size() != scores.size())
        throw std::runtime_error("Expect labels and scores to have a same length!");

    auto maxScore = std::max_element(scores.begin(), scores.end());
    auto minScore = std::min_element(scores.begin(), scores.end());

    // + 1 to avoid 0
    float step = (*maxScore - *minScore + 2) / float(scores.size() + 1);

    float thresh = *maxScore + 1;
    std::vector<float> listTPR;
    std::vector<float> listFPR;
    std::vector<float> listThresholds;

    // Add small epsilon to make sure denominator to != 0;
    float eps = 1e-9;

    for (int k = 0; k < scores.size() + 2; k++) {
        float TP = 0, TN = 0, FP = 0, FN = 0;
        for (int i = 0; i < binaryLabels.size(); i++) {
            int trueLabel = binaryLabels[i];
            int predLabel = (scores[i] >= thresh) ? 1 : 0;

            if (trueLabel == 1) {
                if (predLabel == 1)
                    TP++;
                else
                    FN++;
            } else {
                if (predLabel == 1)
                    FP++;
                else
                    TN++;
            }
        }

        float TPR = TP / (TP + FN + eps);
        float FPR = FP / (FP + TN + eps);

        listTPR.emplace_back(TPR);
        listFPR.emplace_back(FPR);
        listThresholds.emplace_back(thresh);
        thresh -= step;
    }
    return std::make_tuple(listTPR, listFPR, listThresholds);
}

std::tuple<float, float, float, float>
GetThresholds(std::vector<int> &binaryLabels, std::vector<float> &scores) {
    if (binaryLabels.size() != scores.size())
        throw std::runtime_error("Expect labels and scores to have a same length!");

    auto maxScore = std::max_element(scores.begin(), scores.end());
    auto minScore = std::min_element(scores.begin(), scores.end());

    // Fix 20 steps
    int numSteps = 20;
    float step = (*maxScore - *minScore + 2) / float(numSteps);

    float thresh = *maxScore + 1;
    float maxPrecisionThresh = *maxScore;
    float maxRecallThresh = *minScore;
    bool stopRecall = false;
    float maxF1Thresh = 0, maxF1 = std::numeric_limits<float>::min();
    float accuracy = 0;

    // Add small epsilon to make sure denominator to != 0;
    float eps = 1e-9;

    for (int k = 0; k < numSteps + 1; k++) {
        float TP = 0, TN = 0, FP = 0, FN = 0;
        for (int i = 0; i < binaryLabels.size(); i++) {
            int trueLabel = binaryLabels[i];
            int predLabel = (scores[i] >= thresh) ? 1 : 0;

            if (trueLabel == 1) {
                if (predLabel == 1)
                    TP++;
                else
                    FN++;
            } else {
                if (predLabel == 1)
                    FP++;
                else
                    TN++;
            }
        }
        float _precision = TP / (TP + FP + eps);
        float _recall = TP / (TP + FN + eps);
        float _f1 = 2 * _precision * _recall / (_precision + _recall);

        if (_precision > 1 - eps)
            // Precision decrease with thresh
            maxPrecisionThresh = thresh;

        if (_recall > 1 - eps && !stopRecall) {
            // Recall increase with thresh, stop at first recall ~ 1
            maxRecallThresh = thresh;
            stopRecall = true;
        }

        if (_f1 > maxF1) {
            maxF1 = _f1;
            maxF1Thresh = thresh;
            accuracy = (TP + TN) / (TP + FP + FN + TN + eps);
        }

        thresh -= step;
    }

    // Max precision: high confident if DEFECT is DEFECT
    // Max recall: high confident detect all DEFECT
    // Max F1: balance between precision - recall
    // Acc: accuracy at max f1 threshold;
    return std::make_tuple(maxPrecisionThresh, maxRecallThresh, maxF1Thresh, accuracy);
}

#pragma clang diagnostic pop