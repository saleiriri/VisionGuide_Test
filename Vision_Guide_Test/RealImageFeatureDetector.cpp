#include "RealImageFeatureDetector.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>
#include "Logger.h"

double RealImageFeatureDetector::calculateCircularity(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0 || area <= 0) return 0.0;
    return 4.0 * CV_PI * area / (perimeter * perimeter);
}

// 双线性差值亚像素灰度值采样
static float getPixelValue(const cv::Mat& img, float x, float y) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    if (x0 < 0 || y0 < 0 || x0 >= img.cols - 1 || y0 >= img.rows - 1) {
        return 0.0f;
    }
    float dx = x - x0;
    float dy = y - y0;

    // 处理不同类型
    if (img.type() == CV_8UC1) {
        float v00 = img.at<uchar>(y0, x0);
        float v10 = img.at<uchar>(y0, x0 + 1);
        float v01 = img.at<uchar>(y0 + 1, x0);
        float v11 = img.at<uchar>(y0 + 1, x0 + 1);
        return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
            (1 - dx) * dy * v01 + dx * dy * v11;
    }
    else 
    {
        // 先读取为 float
        cv::Mat tmp;
        if (img.type() == CV_32FC1) {
            // 直接读取
            float v00 = img.at<float>(y0, x0);
            float v10 = img.at<float>(y0, x0 + 1);
            float v01 = img.at<float>(y0 + 1, x0);
            float v11 = img.at<float>(y0 + 1, x0 + 1);
            return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
                (1 - dx) * dy * v01 + dx * dy * v11;
        }
        else 
        {
            // 转换 CV_32F
            img.convertTo(tmp, CV_32F);
            float v00 = tmp.at<float>(y0, x0);
            float v10 = tmp.at<float>(y0, x0 + 1);
            float v01 = tmp.at<float>(y0 + 1, x0);
            float v11 = tmp.at<float>(y0 + 1, x0 + 1);
            return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
                (1 - dx) * dy * v01 + dx * dy * v11;
        }
    }
}

// 计算椭圆点的法线方向
static cv::Point2f getEllipseNormal(const cv::RotatedRect& ellipse, float angle_rad) {
    double a = ellipse.size.width / 2.0;
    double b = ellipse.size.height / 2.0;
    if (a < 0.001 || b < 0.001) return cv::Point2f(0, 0);
    double theta = ellipse.angle * CV_PI / 180.0;

    // 局部坐标系中的点
    double x_local = a * cos(angle_rad);
    double y_local = b * sin(angle_rad);

    // 局部坐标系中的法线方向（指向外部）
    double nx_local = x_local / (a * a);
    double ny_local = y_local / (b * b);
    double norm = sqrt(nx_local * nx_local + ny_local * ny_local);
    if (norm < 0.001) return cv::Point2f(0, 0);
    nx_local /= norm;
    ny_local /= norm;

    // 旋转回图像坐标系
    cv::Point2f normal;
    normal.x = nx_local * cos(theta) - ny_local * sin(theta);
    normal.y = nx_local * sin(theta) + ny_local * cos(theta);
    return normal;
}

// 评估得分
static CircleScore evaluateEllipse(
    const cv::Mat& gray,
    const cv::Mat& grad_x,
    const cv::Mat& grad_y,
    const cv::RotatedRect& ellipse,
    int num_samples = 100)
{
    CircleScore score;
    score.edge_strength = 0.0;
    score.direction_consistency = 0.0;
    score.final_score = 0.0;

    double total_strength = 0.0;
    double total_consistency = 0.0;
    int valid_count = 0;

    double a = ellipse.size.width / 2.0;
    double b = ellipse.size.height / 2.0;
    if (a < 1.0 || b < 1.0) return score;      // 当长短轴过小，无法采样时，直接返回 0 分

    cv::Point2f center = ellipse.center;
    double theta = ellipse.angle * CV_PI / 180.0;


    // 等弧长采样     预计算弧长查找表，然后二分查找等弧长对应的角度
    const int TABLE_SIZE = 1000;                           // 查找表精度

    // 预计算弧长表（从 0 到 2π）
    std::vector<double> arc_lengths(TABLE_SIZE + 1);       // 弧长累加值
    std::vector<double> angles(TABLE_SIZE + 1);            // 对应的角度

    double total_arc = 0.0;                                // 椭圆总周长

    for (int idx = 0; idx <= TABLE_SIZE; ++idx) {
        double t = (2.0 * CV_PI * idx) / TABLE_SIZE;       // 当前角度
        angles[idx] = t;                                   // 记录角度

        if (idx > 0) {
            // 弧长微元：ds = sqrt(a²·sin²(t) + b²·cos²(t)) · dt
            double dt = (2.0 * CV_PI) / TABLE_SIZE;        // 角度步长
            double t_prev = angles[idx - 1];

            // 用梯形法近似弧长积分
            double ds_prev = sqrt(a * a * sin(t_prev) * sin(t_prev) +
                b * b * cos(t_prev) * cos(t_prev));
            double ds_curr = sqrt(a * a * sin(t) * sin(t) +
                b * b * cos(t) * cos(t));
            total_arc += (ds_prev + ds_curr) / 2.0 * dt;   // 累加弧长
        }
        arc_lengths[idx] = total_arc;                      // 记录累加弧长
    }

    // 等弧长采样
    for (int i = 0; i < num_samples; ++i) {
        // 每个弧在 0 到 total_arc 均匀分布
        double target_arc = (total_arc * i) / num_samples;

        // 二分查找：找到 arc_lengths 中第一个 >= target_arc 的位置
        int idx = std::lower_bound(arc_lengths.begin(), arc_lengths.end(), target_arc)
            - arc_lengths.begin();

        // 边界保护
        if (idx > TABLE_SIZE) idx = TABLE_SIZE;
        if (idx == 0) idx = 1;

        // 根据弧长反推精确角度
        double t_prev = angles[idx - 1];
        double t_curr = angles[idx];
        double arc_prev = arc_lengths[idx - 1];
        double arc_curr = arc_lengths[idx];
        double t = t_prev + (t_curr - t_prev) * (target_arc - arc_prev) / (arc_curr - arc_prev);   // t为当前采样点的角度

        // 用插值后的角度计算椭圆上的点
        float x_local = a * cos(t);
        float y_local = b * sin(t);
        float px = center.x + x_local * cos(theta) - y_local * sin(theta);
        float py = center.y + x_local * sin(theta) + y_local * cos(theta);

        // 边界检查
        if (px < 1 || px >= gray.cols - 1 || py < 1 || py >= gray.rows - 1) continue;

        // 采样梯度值
        float gx = getPixelValue(grad_x, px, py);
        float gy = getPixelValue(grad_y, px, py);
        float mag = sqrt(gx * gx + gy * gy);
        if (mag < 0.001) continue;

        // 获取精确法线方向（指向外部）
        cv::Point2f normal = getEllipseNormal(ellipse, t);   // 用 t 而不是 angle
        float n_len = sqrt(normal.x * normal.x + normal.y * normal.y);
        if (n_len < 0.001) continue;
        normal.x /= n_len;
        normal.y /= n_len;

        // 归一化
        float g_norm_x = gx / mag;
        float g_norm_y = gy / mag;

        // 为保证方向一致性，点积取绝对值
        float dot = std::abs(g_norm_x * normal.x + g_norm_y * normal.y);

        total_strength += mag;
        total_consistency += dot;
        valid_count++;
    }


    /*  已将逻辑修改为等弧长采样，将等角度采样的代码保留作为参考
    for (int i = 0; i < num_samples; ++i) {
        float angle = 2.0f * CV_PI * i / num_samples;     // 等角度采样不适用于椭圆，会导致采样点在长轴方向过于密集，短轴方向过于稀疏。

        // 椭圆参数方程（考虑旋转）
        float x_local = a * cos(angle);
        float y_local = b * sin(angle);
        float theta = ellipse.angle * CV_PI / 180.0f;
        float px = center.x + x_local * cos(theta) - y_local * sin(theta);
        float py = center.y + x_local * sin(theta) + y_local * cos(theta);

        // 边界检查
        if (px < 1 || px >= gray.cols - 1 || py < 1 || py >= gray.rows - 1) continue;

        // 采样梯度值（亚像素）
        float gx = getPixelValue(grad_x, px, py);
        float gy = getPixelValue(grad_y, px, py);
        float mag = sqrt(gx * gx + gy * gy);
        if (mag < 0.001) continue;

        // 获取精确法线方向（指向外部）
        cv::Point2f normal = getEllipseNormal(ellipse, angle);
        float n_len = sqrt(normal.x * normal.x + normal.y * normal.y);
        if (n_len < 0.001) continue;
        normal.x /= n_len;
        normal.y /= n_len;

        // 归一化
        float g_norm_x = gx / mag;
        float g_norm_y = gy / mag;

        // 方向一致性：点积绝对值（梯度方向与法线方向的夹角余弦）
        float dot = std::abs(g_norm_x * normal.x + g_norm_y * normal.y);

        total_strength += mag;
        total_consistency += dot;
        valid_count++;
    }
    */

    if (valid_count > 0) {
        score.edge_strength = total_strength / valid_count;
        score.direction_consistency = total_consistency / valid_count;
        // 综合评分 = 边缘强度 × 方向一致性
        score.final_score = score.edge_strength * score.direction_consistency;
    }
    return score;
}




// 返回单个ROI内所有通过初筛的候选椭圆
static bool collectCandidatesInROI(const cv::Mat& image, const ROIParams& params,
    std::vector<DetectedFeature>& candidates,
    const cv::Mat& gray_full,     // 全局灰度图（用于打分）
    const cv::Mat& grad_x,        // X方向梯度
    const cv::Mat& grad_y) {      // Y方向梯度
    cv::Rect valid_roi = params.roi & cv::Rect(0, 0, image.cols, image.rows);
    if (valid_roi.width <= 0 || valid_roi.height <= 0) return false;
    cv::Mat roi_img = image(valid_roi);

    cv::Mat gray;
    if (roi_img.channels() == 3) cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);
    else gray = roi_img.clone();

    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.5);
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV, 15, 8);

    // canny
    cv::Mat edges;
    cv::Canny(gray, edges, 20, 100);   // 提取边缘
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);  //闭运算

    // 膨胀
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    // cv::dilate(binary, binary, kernel, cv::Point(-1, -1), 1);

    std::vector<cv::Point> edge_points;
    cv::findNonZero(edges, edge_points);
    if (edge_points.size() < 10) return false;

    // 初筛
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    candidates.clear();
    for (const auto& contour : contours) {

        //double perimeter = cv::arcLength(contour, true);
        //double approx_radius = perimeter / (2.0 * CV_PI);    // 因为边缘可能断裂，因此统计周长并非椭圆实际周长，平均半径也非真实半径
        //if (approx_radius < params.minRadius || approx_radius > params.maxRadius)
        //    continue;

        if (contour.size() < 10) continue;
        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        cv::Size2f size = ellipse.size;

        double avg_radius = (size.width + size.height) / 4.0;   // 由长短轴计算平均半径
        if (avg_radius < params.minRadius || avg_radius > params.maxRadius)
            continue;

        double aspect = std::max(size.width, size.height) / std::min(size.width, size.height);
        if (aspect < params.minAspectRatio || aspect > params.maxAspectRatio)
            continue;

        // 将椭圆偏移到全局坐标，进行打分
        cv::RotatedRect ellipse_global = ellipse;
        ellipse_global.center += cv::Point2f(valid_roi.x, valid_roi.y);

        // 计算综合评分
        CircleScore score = evaluateEllipse(gray_full, grad_x, grad_y, ellipse_global, 100);



        // 将椭圆偏移回原图坐标
        ellipse.center += cv::Point2f(valid_roi.x, valid_roi.y);
        DetectedFeature feat;
        feat.point = ellipse.center;
        feat.ellipse = ellipse;
        feat.avgRadius = avg_radius;
        // 新增评分
        feat.score = score;

        candidates.push_back(feat);
    }
    


    // 新加
    // 按综合评分降序排序
    std::sort(candidates.begin(), candidates.end(),
        [](const DetectedFeature& a, const DetectedFeature& b) {
            return a.score.final_score > b.score.final_score;
        });

    // 打印每个候选的分数（调试用）
    for (size_t i = 0; i < candidates.size() && i < 5; ++i) {
        std::cout << "[Cand " << i << "] 半径=" << candidates[i].avgRadius
            << ", 分数=" << candidates[i].score.final_score
            << " (强度=" << candidates[i].score.edge_strength
            << ", 方向=" << candidates[i].score.direction_consistency << ")"
            << std::endl;
    }


    return !candidates.empty();
}

bool RealImageFeatureDetector::detectWithEllipse(const cv::Mat& image, std::vector<DetectedFeature>& features) {
    if (image.empty() || roiParams_.empty()) {
        std::cout << "[Debug] 检测失败：image.empty()=" << image.empty() << ", roiParams_.empty()=" << roiParams_.empty() << std::endl;
        LOG_ERROR("检测失败：image.empty()={}, roiParams_.empty()={}", image.empty(), roiParams_.empty());
        return false;
    }
    features.clear();
    features.reserve(roiParams_.size());


    // 预计算灰度图和梯度图（全图一次，避免重复计算）
    cv::Mat gray_full;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_full, cv::COLOR_BGR2GRAY);
    }
    else {
        gray_full = image.clone();
    }

    cv::Mat grad_x, grad_y;
    cv::Sobel(gray_full, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray_full, grad_y, CV_32F, 0, 1, 3);
    //


    for (size_t i = 0; i < roiParams_.size(); ++i) {
        // 调用 collectCandidatesInROI 获取所有候选
        std::vector<DetectedFeature> candidates;
        bool ok = collectCandidatesInROI(image, roiParams_[i], candidates,
            gray_full, grad_x, grad_y);

        if (!ok || candidates.empty()) {
            std::cout << "[ROI " << i << "] 无候选" << std::endl;
            return false;
        }

        // 候选已按分数排序
        const DetectedFeature& best = candidates[0];

        // 动态阈值判断
        if (best.score.final_score < 10.0) {
            std::cout << "[ROI " << i << "] 最高分：" << best.score.final_score << "，可信度不足" << std::endl;
            LOG_INFO("[ROI {}] 最高分：{}，可信度不足", i, best.score.final_score);
            return false;
        }

        std::cout << "[ROI " << i << "] 选中: 半径=" << best.avgRadius
            << ", 分数=" << best.score.final_score
            << " (强度=" << best.score.edge_strength
            << ", 方向=" << best.score.direction_consistency << ")"
            << std::endl;

        features.push_back(best);
    }
    return true;
}

// 返回所有ROI中符合要求的椭圆
bool RealImageFeatureDetector::detectAllCandidates(const cv::Mat& image,
    std::vector<std::vector<DetectedFeature>>& allCandidatesPerROI) {

    if (image.empty() || roiParams_.empty()) return false;
    allCandidatesPerROI.clear();
    allCandidatesPerROI.reserve(roiParams_.size());

    // 预计算灰度图和梯度图
    cv::Mat gray_full;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_full, cv::COLOR_BGR2GRAY);
    }
    else {
        gray_full = image.clone();
    }

    cv::Mat grad_x, grad_y;
    cv::Sobel(gray_full, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray_full, grad_y, CV_32F, 0, 1, 3);

    for (size_t i = 0; i < roiParams_.size(); ++i) {
        std::vector<DetectedFeature> candidates;

        // 传入 5 个参数
        collectCandidatesInROI(image, roiParams_[i], candidates,
            gray_full, grad_x, grad_y);

        allCandidatesPerROI.push_back(candidates);
    }
    return true;
}

