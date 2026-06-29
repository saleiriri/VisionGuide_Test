#pragma once
#include "FeatureDetectorBase.h"
#include <opencv2/dnn.hpp>

struct YoloConfig {
    std::string model_path;      // ONNX模型路径
    float confidence_threshold = 0.5;
    float nms_threshold = 0.4;
    int input_width = 640;
    int input_height = 640;
    int hole_class_id = 0;       // 孔洞在模型中的类别ID
};

class YoloDetector : public FeatureDetectorBase {
public:
    explicit YoloDetector(const YoloConfig& config);
    ~YoloDetector() = default;

    bool detect(const cv::Mat& image,
        std::vector<DetectedFeature>& features) override;

    void setROIParams(const std::vector<ROIParams>& params) override {
        // YOLO 不需要 ROI，留空
        (void)params;  // 消除未使用参数警告
        std::cout << "[YoloDetector] setROIParams called but ignored." << std::endl;
    }

private:
    YoloConfig config_;
    cv::dnn::Net net_;

    cv::Mat preprocess(const cv::Mat& image);
    bool postprocess(const cv::Mat& image,
        const std::vector<cv::Mat>& outputs,
        std::vector<DetectedFeature>& features);
    bool refineCenter(const cv::Mat& image,
        const cv::Rect& bbox,
        cv::Point2f& refined_center);
};
#pragma once
