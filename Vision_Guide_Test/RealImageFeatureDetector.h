#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// 单个ROI的独立参数
struct ROIParams {
    cv::Rect roi;                   // 感兴趣区域（像素坐标）
    double minRadius = 10.0;        // 最小半径（像素）
    double maxRadius = 80.0;        // 最大半径（像素）
    double minCircularity = 0.3;    // 最小圆度（正圆为1.0）
    double minAspectRatio = 0.3;    // 长短轴比下限
    double maxAspectRatio = 2.0;    // 长短轴比上限
};

class RealImageFeatureDetector {
public:
    RealImageFeatureDetector() = default;    

    // 设置所有ROI参数（数量即为期望提取的点数）
    void setROIParams(const std::vector<ROIParams>& params) { roiParams_ = params; }

    // 执行检测：每个ROI输出一个最佳点（按圆度分数最高）
    // 返回 true 表示所有ROI都成功找到点
    bool detect(const cv::Mat& image, std::vector<cv::Point2f>& points);

private:
    std::vector<ROIParams> roiParams_;

    // 计算轮廓圆度
    double calculateCircularity(const std::vector<cv::Point>& contour);

    // 在单个ROI内检测，返回最佳点及其分数
    bool detectInSingleROI(const cv::Mat& image, const ROIParams& params,
        cv::Point2f& bestPoint, double& bestScore);
};