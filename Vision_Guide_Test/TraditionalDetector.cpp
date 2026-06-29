#include "TraditionalDetector.h"
#include "FeatureDetectorBase.h"
#include "RealImageFeatureDetector.h"

bool TraditionalDetector::detect(const cv::Mat& image,
    std::vector<DetectedFeature>& features) {
    return detector_.detectWithEllipse(image, features);
}

void TraditionalDetector::setROIParams(const std::vector<ROIParams>& params) {
    detector_.setROIParams(params);
}

bool TraditionalDetector::detectAllCandidates(
    const cv::Mat& image,
    std::vector<std::vector<DetectedFeature>>& allCandidates) {
    return detector_.detectAllCandidates(image, allCandidates);
}