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
    SalientDetector();

    explicit SalientDetector(const std::string &modelPath, int gpuID = 0);

    ~SalientDetector();

    cv::Mat FindBinaryMask(cv::Mat &cropImage, float threshold = 0.1);

    /**
     * Expand filled binary mask with MORPH_RECT
     * @param binaryMask: filled binary mask
     * @param dilateRatio: dilate kernel = dilateRatio * max(width, height)
     * @return
     */
    static cv::Mat DilateBinaryMask(cv::Mat &binaryMask, float dilateRatio = 0.05);

    /**
     * Auto refine the mask drawn by User
     * @param raw: BGR image to do auto-segmentation
     * @param rawMask: User drawn mask (8UC3 or 8UC1)
     * @param threshold: params for segment function
     * @return: newMask surrounding object
     *
     * @throw: std::runtime_error if cannot find any contour
     */
    cv::Mat RefineMask(cv::Mat &raw, cv::Mat &rawMask, float threshold = 0.1, int thickness = 2);

    /**
     * Crop an image given its binary mask using contour
     * @param raw: raw BGR image
     * @param mask: User drawn mask (8UC3 or 8UC1)
     * @param doWarpAffine: Whether to align object
     * @param expandRatio: Expand cropped image by a fraction
     * @return: pair of croppedRaw and filled croppedMask after dilation
     *
     * @throw: std::runtime_error if cannot find any contour
     */
    static std::pair<cv::Mat, cv::Mat> CropMaskByContour(
            cv::Mat &raw, cv::Mat &mask,
            bool doWarpAffine = true,
            float expandRatio = 0.08);

    /**
     * Output segmentation mask for COLOR input image
     * Image will be padded x2 in case of big object
     * Heat map [0,1] with 1 denotes salient
     */
    cv::Mat Infer(cv::Mat &srcImage);

    static at::Tensor PreProcess(cv::Mat &srcImage);
};


#endif //U2_NET_INFER_SALIENT_DETECTOR_H
