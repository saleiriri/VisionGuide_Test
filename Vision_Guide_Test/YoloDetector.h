#pragma once

#include "FeatureDetectorBase.h"
#include <opencv2/dnn.hpp>
#include <vector>

struct YoloConfig {
    std::string model_path;
    float confidence_threshold = 0.5;
    float nms_threshold = 0.4;
    int input_width = 640;
    int input_height = 640;
    int hole_class_id = 0;
};

class YoloDetector : public FeatureDetectorBase {
public:
    explicit YoloDetector(const YoloConfig& config);
    ~YoloDetector() = default;

    // 继承自 FeatureDetectorBase
    bool detect(const cv::Mat& image,
        std::vector<DetectedFeature>& features) override;

    void setROIParams(const std::vector<ROIParams>& params) override {
        (void)params;
        // YOLO 不需要手动 ROI
    }

    // 新增：只返回 ROI
    bool detectROIs(const cv::Mat& image, std::vector<cv::Rect>& rois);

    // 新增：返回 ROI + 置信度
    bool detectROIsWithConfidence(const cv::Mat& image,
        std::vector<cv::Rect>& rois,
        std::vector<float>* confidences);

private:
    YoloConfig config_;
    cv::dnn::Net net_;

    cv::Mat preprocess(const cv::Mat& image);

    bool postprocess(const cv::Mat& image,
        const std::vector<cv::Mat>& outputs,
        std::vector<cv::Rect>& rois,
        std::vector<float>* confidences);
};