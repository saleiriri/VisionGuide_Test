#pragma once
#include "FeatureDetectorBase.h"
#include "RealImageFeatureDetector.h"

class TraditionalDetector : public FeatureDetectorBase {
public:
    TraditionalDetector() = default;
    ~TraditionalDetector() = default;

    // 重写基类方法
    bool detect(const cv::Mat& image,
        std::vector<DetectedFeature>& features) override;

    void setROIParams(const std::vector<ROIParams>& params) override;

    bool detectAllCandidates(const cv::Mat& image,
        std::vector<std::vector<DetectedFeature>>& allCandidates);

private:
    RealImageFeatureDetector detector_;
};
#pragma once