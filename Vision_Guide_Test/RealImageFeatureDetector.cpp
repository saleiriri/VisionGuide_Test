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

/*bool RealImageFeatureDetector::detectInSingleROI(const cv::Mat& image, const ROIParams& params,
    DetectedFeature& outFeature, double& bestScore) {
    

    // 裁剪ROI
    cv::Rect valid_roi = params.roi & cv::Rect(0, 0, image.cols, image.rows);
    if (valid_roi.width <= 0 || valid_roi.height <= 0) return false;
    cv::Mat roi_img = image(valid_roi);
    
    
    
    
    // 转为灰度
    cv::Mat gray;
    if (roi_img.channels() == 3) {
        cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = roi_img.clone();  // 假设已经是灰度图
    }

    // 高斯模糊
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.5);

    // 自适应阈值二值化
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY_INV, 15, 8);


    cv::Mat edges;
    cv::Canny(gray, edges, 20, 60); // 提取边缘
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    
    //cv::morphologyEx(edges, edges, cv::MORPH_OPEN, kernel);   // 开运算去除杂点
    //cv::morphologyEx(edges, binary, cv::MORPH_CLOSE, kernel); // 闭合断开的孔边缘
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    // 查看canny图片
    cv::imshow("Canny Edges", edges);
    cv::waitKey(0); // 等待按键，防止窗口一闪而过
    

    return true;
}*/

// 双线性差值亚像素灰度值采样
// cpp
static float getPixelValue(const cv::Mat& img, float x, float y) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    if (x0 < 0 || y0 < 0 || x0 >= img.cols - 1 || y0 >= img.rows - 1) {
        return 0.0f;
    }
    float dx = x - x0;
    float dy = y - y0;

    // 处理常见类型：CV_8UC1 和 CV_32FC1
    if (img.type() == CV_8UC1) {
        float v00 = img.at<uchar>(y0, x0);
        float v10 = img.at<uchar>(y0, x0 + 1);
        float v01 = img.at<uchar>(y0 + 1, x0);
        float v11 = img.at<uchar>(y0 + 1, x0 + 1);
        return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
            (1 - dx) * dy * v01 + dx * dy * v11;
    }
    else {
        // 其他类型（例如 CV_32F/CV_32FC1），先读取为 float
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
        else {
            // 通用路径：转换到 CV_32F（低频率调用时可接受）
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

// 计算椭圆上某点的精确法线方向
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

// 评估椭圆的得分
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
    if (a < 1.0 || b < 1.0) return score;

    cv::Point2f center = ellipse.center;

    for (int i = 0; i < num_samples; ++i) {
        float angle = 2.0f * CV_PI * i / num_samples;

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

        // 获取精确法线方向（指向椭圆外部）
        cv::Point2f normal = getEllipseNormal(ellipse, angle);
        float n_len = sqrt(normal.x * normal.x + normal.y * normal.y);
        if (n_len < 0.001) continue;
        normal.x /= n_len;
        normal.y /= n_len;

        // 梯度方向（归一化）
        float g_norm_x = gx / mag;
        float g_norm_y = gy / mag;

        // 方向一致性：点积绝对值（梯度方向与法线方向的夹角余弦）
        float dot = std::abs(g_norm_x * normal.x + g_norm_y * normal.y);

        total_strength += mag;
        total_consistency += dot;
        valid_count++;
    }

    if (valid_count > 0) {
        score.edge_strength = total_strength / valid_count;
        score.direction_consistency = total_consistency / valid_count;
        // 综合评分 = 边缘强度 × 方向一致性（两者越高越可信）
        score.final_score = score.edge_strength * score.direction_consistency;
    }
    return score;
}




// 返回单个ROI内所有通过初筛的候选椭圆
static bool collectCandidatesInROI(const cv::Mat& image, const ROIParams& params,
    std::vector<DetectedFeature>& candidates,
    const cv::Mat& gray_full,   // 新增：全局灰度图（用于打分）
    const cv::Mat& grad_x,      // 新增：X方向梯度
    const cv::Mat& grad_y) {      // 新增：Y方向梯度
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
    cv::Canny(gray, edges, 20, 100); // 提取边缘
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);  //闭运算

    // 膨胀
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    // cv::dilate(binary, binary, kernel, cv::Point(-1, -1), 1);

    std::vector<cv::Point> edge_points;
    cv::findNonZero(edges, edge_points);
    if (edge_points.size() < 10) return false;

    // 但更简单的方式：我们仍然用轮廓法，但需要闭合，所以我们用findContours，并收集所有通过半径筛选的轮廓
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

        // ===== 新增：将椭圆偏移到全局坐标，进行打分 =====
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
        //  新增评分
        feat.score = score;

        candidates.push_back(feat);
    }
    


    // 新加
    // 按综合评分降序排序（分数最高的排在前面）
    std::sort(candidates.begin(), candidates.end(),
        [](const DetectedFeature& a, const DetectedFeature& b) {
            return a.score.final_score > b.score.final_score;
        });

    // 可选：打印每个候选的分数（调试用）
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
        std::cout << "[Debug] 检测失败：image.empty()=" << image.empty()
            << ", roiParams_.empty()=" << roiParams_.empty() << std::endl;
        return false;
    }
    features.clear();
    features.reserve(roiParams_.size());


    // ===== 新增：预计算灰度图和梯度图（全图一次，避免重复计算） =====
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
        // ===== 改动1：调用 collectCandidatesInROI 获取所有候选 =====
        std::vector<DetectedFeature> candidates;
        bool ok = collectCandidatesInROI(image, roiParams_[i], candidates,
            gray_full, grad_x, grad_y);

        if (!ok || candidates.empty()) {
            std::cout << "[ROI " << i << "] 无候选" << std::endl;
            return false;
        }

        // ===== 改动2：候选已按分数排序，直接取第一个 =====
        const DetectedFeature& best = candidates[0];

        // ===== 改动3：动态阈值判断 =====
        if (best.score.final_score < 10.0) {
            std::cout << "[ROI " << i << "] 最高分仅 " << best.score.final_score
                << "，可信度不足，拒绝" << std::endl;
            return false;
        }

        std::cout << "[ROI " << i << "] ✅ 选中: 半径=" << best.avgRadius
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

