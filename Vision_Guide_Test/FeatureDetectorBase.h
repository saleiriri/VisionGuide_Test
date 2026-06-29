#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// ROI参数（已修改为每个ROI独立配置,因此此处设定值为初始值）
struct ROIParams {
    cv::Rect roi;                   // 像素
    double minRadius = 10.0;        // 最小半径
    double maxRadius = 180.0;        // 最大半径
    double minCircularity = 0.3;    // 最小圆度
    double minAspectRatio = 0.3;    // 长短轴比下限
    double maxAspectRatio = 4.0;    // 长短轴比上限
};

struct CircleScore {
    double edge_strength = 0.0;           // 边缘强度
    double direction_consistency = 0.0;   // 方向一致性
    double final_score = 0.0;             // 综合评分
};

// 检测结果结构体（与原有保持一致）
struct DetectedFeature {
    cv::Point2f point;           // 中心点
    cv::RotatedRect ellipse;     // 拟合椭圆
    CircleScore score;           // 评分信息
    double avgRadius = 0.0;      // 平均半径
    double confidence = 0.0;     // 置信度
    double edge_strength = 0.0;  // 边缘强度
    double direction_consistency = 0.0;
    // 方便调试
    DetectedFeature() = default;
};

// 抽象接口类
class FeatureDetectorBase {
public:
    virtual ~FeatureDetectorBase() = default;

    // 核心检测接口
    virtual bool detect(const cv::Mat& image,
        std::vector<DetectedFeature>& features) = 0;

    // 设置 ROI 参数
    virtual void setROIParams(const std::vector<ROIParams>& params) = 0;
};
