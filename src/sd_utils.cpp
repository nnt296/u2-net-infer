#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

#include <random>
#include "sd_utils.h"

at::Tensor ToTensor(cv::Mat &image) {
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

cv::Mat ToCvImage(at::Tensor &tensor, int cvType) {
    int height = int(tensor.sizes()[0]);
    int width = int(tensor.sizes()[1]);
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
        std::cout << "An error has occurred when converting Cv2Tensor : " << e.msg() << std::endl;
    }

    return cv::Mat(height, width, CV_8UC3);
}

int GetMaxAreaContourId(std::vector<std::vector<cv::Point>> contours) {
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        } // End if
    } // End for
    return maxAreaContourId;
} // End function

#pragma clang diagnostic pop