#include "RealImageFeatureDetector.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>

double RealImageFeatureDetector::calculateCircularity(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0 || area <= 0) return 0.0;
    return 4.0 * CV_PI * area / (perimeter * perimeter);
}

bool RealImageFeatureDetector::detectInSingleROI(const cv::Mat& image, const ROIParams& params,
    cv::Point2f& bestPoint, double& bestScore) {
    // 裁剪ROI
    cv::Rect valid_roi = params.roi & cv::Rect(0, 0, image.cols, image.rows);
    if (valid_roi.width <= 0 || valid_roi.height <= 0) return false;
    cv::Mat roi_img = image(valid_roi);

    // 转为灰度
    cv::Mat gray;
    if (roi_img.channels() == 3)
        cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roi_img.clone();

    // 高斯模糊
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);

    // 自适应阈值二值化
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV, 15, 8);

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    bestScore = -1.0;
    bool found = false;

    for (const auto& contour : contours) {
        // 粗略半径筛选（基于周长）
        double perimeter = cv::arcLength(contour, true);
        double approx_radius = perimeter / (2.0 * CV_PI);
        if (approx_radius < params.minRadius || approx_radius > params.maxRadius)
            continue;

        // 圆度筛选
        double circularity = calculateCircularity(contour);
        if (circularity < params.minCircularity)
            continue;

        // 椭圆拟合（需要至少5个点）
        if (contour.size() < 5) continue;
        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        cv::Size2f size = ellipse.size;
        double aspect = std::max(size.width, size.height) / std::min(size.width, size.height);
        if (aspect < params.minAspectRatio || aspect > params.maxAspectRatio)
            continue;

        // 分数 = 圆度 * (1 / 长宽比) 鼓励更圆更正的孔
        double score = circularity / aspect;
        if (score > bestScore) {
            bestScore = score;
            bestPoint = ellipse.center + cv::Point2f(valid_roi.x, valid_roi.y);
            found = true;
        }
    }
    return found;
}

bool RealImageFeatureDetector::detect(const cv::Mat& image, std::vector<cv::Point2f>& points) {
    if (image.empty() || roiParams_.empty()) {
        std::cerr << "[FeatureDetector] Empty image or no ROI parameters set." << std::endl;
        return false;
    }

    points.clear();
    points.reserve(roiParams_.size());

    for (size_t i = 0; i < roiParams_.size(); ++i) {
        cv::Point2f bestPt;
        double bestScore;
        bool ok = detectInSingleROI(image, roiParams_[i], bestPt, bestScore);
        if (!ok) {
            std::cerr << "[FeatureDetector] ROI " << i << " failed to find a valid feature." << std::endl;
            return false;  // 任何一个ROI失败都返回false
        }
        points.push_back(bestPt);
        std::cout << "[FeatureDetector] ROI " << i << " found point at ("
            << bestPt.x << ", " << bestPt.y << ") score=" << bestScore << std::endl;
    }

    std::cout << "[FeatureDetector] All " << points.size() << " ROIs detected successfully." << std::endl;
    return true;
}