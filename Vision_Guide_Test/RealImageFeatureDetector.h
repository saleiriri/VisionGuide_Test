#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// ROI参数（已修改为每个ROI独立配置）
struct ROIParams {
    cv::Rect roi;                   // 感兴趣区域（单位：像素）
    double minRadius = 10.0;        // 最小半径
    double maxRadius = 180.0;        // 最大半径
    double minCircularity = 0.3;    // 最小圆度
    double minAspectRatio = 0.3;    // 长短轴比下限
    double maxAspectRatio = 4.0;    // 长短轴比上限
};

struct CircleScore {
    double edge_strength = 0.0;       // 边缘强度（灰度梯度幅值）
    double direction_consistency = 0.0; // 方向一致性
    double final_score = 0.0;          // 综合评分
};

// 用于返回特征点 + 拟合椭圆
struct DetectedFeature {
    cv::Point2f point;
    cv::RotatedRect ellipse;
    CircleScore score;        // 评分信息

    // 调试用
    double circularity;         // 轮廓圆度（4πA/P²）
    double avgRadius;           // 平均半径 (长短轴平均)
};

class RealImageFeatureDetector {
public:

    // 同时返回点和椭圆
    bool detectWithEllipse(const cv::Mat& image, std::vector<DetectedFeature>& features);

    RealImageFeatureDetector() = default;    

    // 设置所有ROI参数
    void setROIParams(const std::vector<ROIParams>& params) { roiParams_ = params; }

    // 返回每个ROI中所有候选椭圆（用于调试可视化）
    bool detectAllCandidates(const cv::Mat& image,
        std::vector<std::vector<DetectedFeature>>& allCandidatesPerROI);

private:
    std::vector<ROIParams> roiParams_;

    // 计算轮廓圆度（）
    double calculateCircularity(const std::vector<cv::Point>& contour);

    // 在单个ROI内检测，返回最佳点及其分数
    bool detectInSingleROI(const cv::Mat& image, const ROIParams& params,
        DetectedFeature& outFeature, double& bestScore,const cv::Mat& gray_full,
        const cv::Mat& grad_x,
        const cv::Mat& grad_y);
};