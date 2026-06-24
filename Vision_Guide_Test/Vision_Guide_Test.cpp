/*#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>
#include <iostream>
#include <vector>
#include <random>
#include "MultiFramePoseAverager.h"  

// 打印矩阵
void printMat(const std::string& name, const cv::Mat& mat) {
    std::cout << name << ":\n" << mat << std::endl;
}

// 误差计算
double rotationVectorDiffInDegrees(const cv::Mat& rvec1, const cv::Mat& rvec2) {
    cv::Mat R1, R2, R_diff;
    cv::Rodrigues(rvec1, R1);
    cv::Rodrigues(rvec2, R2);
    R_diff = R1.t() * R2;
    double trace = cv::trace(R_diff)[0];
    double angle_rad = acos(std::min(1.0, std::max(-1.0, (trace - 1.0) / 2.0)));
    return angle_rad * 180.0 / CV_PI;
}

// 添加噪声
void addGaussianNoise(std::vector<cv::Point2f>& points, double sigma_pixel) {
    if (sigma_pixel <= 0.0) return; 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, sigma_pixel);
    for (auto& pt : points) {
        pt.x += static_cast<float>(noise(gen));
        pt.y += static_cast<float>(noise(gen));
    }
}

// 核心解算
bool solveWorkpiecePose(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    const cv::Mat& K,
    const cv::Mat& D,
    cv::Mat& rvec,
    cv::Mat& tvec,
    double& reproj_error,
    double ransac_threshold = 3.0
) {
    if (object_pts.size() < 3 || image_pts.size() < 3) return false;

    std::vector<int> inliers;
    bool success = cv::solvePnPRansac(
        object_pts, image_pts, K, D,
        rvec, tvec, false, 100, ransac_threshold, 0.99, inliers, cv::SOLVEPNP_EPNP
    );
    if (!success || inliers.size() < 3) return false;

    // 提取点
    std::vector<cv::Point3f> inlier_obj;
    std::vector<cv::Point2f> inlier_img;
    for (int idx : inliers) {
        inlier_obj.push_back(object_pts[idx]);
        inlier_img.push_back(image_pts[idx]);
    }

    // 精修
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001);
    success = cv::solvePnP(
        inlier_obj, inlier_img, K, D,
        rvec, tvec, true, cv::SOLVEPNP_ITERATIVE
    );
    if (!success) return false;

    // 重投影误差
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_pts, rvec, tvec, K, D, projected);
    double sum_err = 0.0;
    for (size_t i = 0; i < image_pts.size(); ++i) {
        sum_err += cv::norm(image_pts[i] - projected[i]);
    }
    reproj_error = sum_err / image_pts.size();

    return true;
}

int main() {
    std::cout << "\n========== Phase 4: Multi-Frame Averaging Test ==========\n" << std::endl;

    // 相机内参 
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 2000.0, 0.0, 640.0, 0.0, 2000.0, 512.0, 0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);

    // 工件特征点
    std::vector<cv::Point3f> object_points = {
        cv::Point3f(-200.0f, -150.0f, 0.0f),
        cv::Point3f(200.0f, -150.0f, 5.0f),
        cv::Point3f(250.0f,  0.0f,   2.0f),
        cv::Point3f(200.0f,  150.0f, -2.0f),
        cv::Point3f(-200.0f,  150.0f, 3.0f),
        cv::Point3f(-250.0f,  0.0f,   0.0f),
        cv::Point3f(0.0f,   -100.0f, 8.0f),
        cv::Point3f(0.0f,   100.0f,  6.0f)
    };
    size_t total_points = object_points.size();

    // 真值位姿
    double base_working_distance = 1500.0;
    double delta_x_true = 5.0;
    double delta_z_deg_true = 2.0;
    cv::Mat rvec_true = (cv::Mat_<double>(3, 1) << 0.0, 0.0, delta_z_deg_true * CV_PI / 180.0);
    cv::Mat tvec_true = (cv::Mat_<double>(3, 1) << delta_x_true, 0.0, base_working_distance);

    std::cout << "[Ground Truth] Translation X: " << delta_x_true << " mm, Rotation Z: " << delta_z_deg_true << " deg" << std::endl;

    // 多帧平均器
    const int FRAMES_TO_AVG = 8;
    MultiFramePoseAverager averager(FRAMES_TO_AVG);

    // 循环模拟连续拍摄 
    // 为了测试，我们连续生成 FRAMES_TO_AVG 帧带噪声的图像
    for (int frame = 0; frame < FRAMES_TO_AVG; ++frame) {
        std::cout << "\n--- Frame " << frame + 1 << " ---" << std::endl;

        // 投影生成理想像素点
        std::vector<cv::Point2f> image_points_clean;
        cv::projectPoints(object_points, rvec_true, tvec_true, camera_matrix, dist_coeffs, image_points_clean);

        // 施加干扰（仅加高斯噪声，模拟不同帧的随机抖动）
        double noise_sigma = 0.5;  // 0.5像素噪声
        std::vector<cv::Point2f> image_points_corrupted = image_points_clean;
        addGaussianNoise(image_points_corrupted, noise_sigma);

        // 也可验证误匹配与遮挡

        // 单帧解算
        cv::Mat rvec, tvec;
        double reproj_err;
        bool success = solveWorkpiecePose(
            object_points, image_points_corrupted,
            camera_matrix, dist_coeffs,
            rvec, tvec, reproj_err
        );

        if (!success) {
            std::cout << "[Frame] Solve failed, skipping." << std::endl;
            continue;
        }

        // 平均
        bool is_ready = averager.addPose(rvec, tvec, reproj_err);

        // 输出融合结果并与真值对比
        if (is_ready) {
            cv::Mat final_rvec = averager.getFinalRvec();
            cv::Mat final_tvec = averager.getFinalTvec();

            // 计算融合结果的误差
            double trans_err = cv::norm(final_tvec - tvec_true);
            double rot_err = rotationVectorDiffInDegrees(final_rvec, rvec_true);

            std::cout << "\n========== Final Fused Pose vs Ground Truth ==========" << std::endl;
            std::cout << "[Fused] Translation (X, Y, Z): "
                << final_tvec.at<double>(0) << ", "
                << final_tvec.at<double>(1) << ", "
                << final_tvec.at<double>(2) << std::endl;
            std::cout << "[Fused] Translation Error: " << trans_err << " mm" << std::endl;
            std::cout << "[Fused] Rotation Error: " << rot_err << " deg" << std::endl;

            // 显示结果
            cv::Mat vis_img(900, 1200, CV_8UC3, cv::Scalar(50, 50, 50));
            // 最后一帧（黄）
            for (const auto& pt : image_points_corrupted) {
                cv::circle(vis_img, cv::Point((int)pt.x, (int)pt.y), 8, cv::Scalar(0, 255, 255), -1);
            }
            // 最终融合位姿（青）
            std::vector<cv::Point2f> reprojected;
            cv::projectPoints(object_points, final_rvec, final_tvec, camera_matrix, dist_coeffs, reprojected);
            for (const auto& pt : reprojected) {
                cv::circle(vis_img, cv::Point((int)pt.x, (int)pt.y), 4, cv::Scalar(255, 255, 0), -1);
            }
            cv::putText(vis_img, "Yellow: Last Frame Input | Cyan: Fused Pose Reprojection",
                cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            cv::putText(vis_img, "Phase 4: Multi-Frame Averaging (8 frames)",
                cv::Point(350, 850), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 1);
            cv::imshow("Phase 4 - Multi-Frame Averaging", vis_img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    std::cout << "\n>>> Phase 4 Completed! <<<" << std::endl;
    return 0;
}*/



// 测试代码   C:\\Users\\ZhuanZ（无密码）\\Desktop\\visionguide\\test.png
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "RealImageFeatureDetector.h"


double g_scale = 1.0;              // 缩放倍数
// ===== 全局变量（用于平移+缩放） =====
cv::Mat g_original_image;          // 原始图像（带检测结果）
cv::Mat g_display_image;           // 当前显示的图像
cv::Mat g_transform;               // 2x3 仿射变换矩阵（控制平移和缩放）
bool g_need_update = true;
const cv::Size g_window_size = cv::Size(1200, 800); // 窗口尺寸

// 鼠标交互
cv::Point2d g_mouse_pos = cv::Point2d(-1, -1); 
bool g_dragging = false;
cv::Point2d g_drag_start;

// 检测参数与结果
std::vector<ROIParams> g_roiParams;
std::vector<cv::Point2f> g_detected_points;
bool g_detection_success = false;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    static cv::Point2d drag_start;  // 拖拽起始点（屏幕坐标）
    static cv::Mat start_transform; // 拖拽开始时的变换矩阵

    if (event == cv::EVENT_MOUSEWHEEL) {
        // -------- 滚轮缩放（以鼠标位置为中心） --------
        int delta = cv::getMouseWheelDelta(flags);
        double scale_factor = (delta > 0) ? 1.1 : 0.9;
        double new_scale = g_transform.at<double>(0, 0) * scale_factor;
        new_scale = std::max(0.1, std::min(10.0, new_scale)); // 限制 0.1~10 倍

        // 获取鼠标在原始图像中的坐标（基于当前变换的逆变换）
        cv::Mat inv_transform;
        cv::invertAffineTransform(g_transform, inv_transform);
        cv::Point2d mouse_orig;
        mouse_orig.x = inv_transform.at<double>(0, 0) * x + inv_transform.at<double>(0, 1) * y + inv_transform.at<double>(0, 2);
        mouse_orig.y = inv_transform.at<double>(1, 0) * x + inv_transform.at<double>(1, 1) * y + inv_transform.at<double>(1, 2);

        // 构建新的变换矩阵：先缩放到 new_scale，再平移使得鼠标指向的点保持不变
        cv::Mat new_transform = cv::Mat::eye(2, 3, CV_64F);
        new_transform.at<double>(0, 0) = new_scale;
        new_transform.at<double>(1, 1) = new_scale;
        // 计算平移量：使得鼠标指向的原图点 (mouse_orig) 在缩放后仍然映射到屏幕坐标 (x, y)
        new_transform.at<double>(0, 2) = x - new_scale * mouse_orig.x;
        new_transform.at<double>(1, 2) = y - new_scale * mouse_orig.y;

        g_transform = new_transform;
        g_need_update = true;
    }
    else if (event == cv::EVENT_LBUTTONDOWN) {
        // 计算点击位置对应的原始图像坐标
        cv::Mat inv_transform;
        cv::invertAffineTransform(g_transform, inv_transform);
        cv::Point2d mouse_orig;
        mouse_orig.x = inv_transform.at<double>(0, 0) * x + inv_transform.at<double>(0, 1) * y + inv_transform.at<double>(0, 2);
        mouse_orig.y = inv_transform.at<double>(1, 0) * x + inv_transform.at<double>(1, 1) * y + inv_transform.at<double>(1, 2);
        std::cout << "[Mouse] Original image coordinate: (" << mouse_orig.x << ", " << mouse_orig.y << ")" << std::endl;
        drag_start = cv::Point2d(x, y);
        start_transform = g_transform.clone();
    }
    else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
        // -------- 拖拽平移 --------
        double dx = x - drag_start.x;
        double dy = y - drag_start.y;
        // 在起始变换的基础上增加平移量
        g_transform = start_transform.clone();
        g_transform.at<double>(0, 2) += dx;
        g_transform.at<double>(1, 2) += dy;
        g_need_update = true;
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        // 左键松开，不做特殊处理
    }
}

void updateDisplay() {
    if (g_original_image.empty()) return;

    // 应用仿射变换到目标窗口尺寸
    cv::Mat warped;
    cv::warpAffine(g_original_image, warped, g_transform, g_window_size, cv::INTER_LINEAR);

    // 操作提示
    cv::putText(warped, "Mouse: Scroll to zoom | Left-drag to pan | R to reset",
        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    g_display_image = warped;
    cv::imshow("Feature Extraction + Grid + Zoom", g_display_image);
    g_need_update = false;
}


int main() {
    std::cout << "\n========== PHASE 5: REAL IMAGE WITH GRID, ZOOM, PAN & ROI ==========\n" << std::endl;

    // 加载图像
    std::string image_path = "C:\\Users\\ZhuanZ（无密码）\\Desktop\\visionguide\\test.png";  // 请修改为你的图片路径
    cv::Mat raw_image = cv::imread(image_path);
    if (raw_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Image size: " << raw_image.cols << " x " << raw_image.rows << std::endl;

    // 配置6个ROI
    std::vector<ROIParams> roiParams(6);
    // 单独调整每个ROI
    roiParams[0].roi = cv::Rect(800, 1100, 300, 250);
    // roiParams[0].minRadius = 10.0;                    //可设置其他参数，此处暂时不设
    roiParams[0].minRadius = 50;
    roiParams[0].maxRadius = 60;

    roiParams[1].roi = cv::Rect(1000, 1840, 300, 250);
    roiParams[1].minRadius = 30;
    roiParams[1].maxRadius = 50;

    roiParams[2].roi = cv::Rect(1720, 2300, 300, 250);
    roiParams[2].minRadius = 40;
    roiParams[2].maxRadius = 50;

    roiParams[3].roi = cv::Rect(2300, 2550, 300, 250);
    roiParams[3].minRadius = 30;
    roiParams[3].maxRadius = 40;

    roiParams[4].roi = cv::Rect(3400, 1850, 500, 400);
    roiParams[4].minRadius = 85;
    roiParams[4].maxRadius = 95;

    roiParams[5].roi = cv::Rect(1860, 1890, 300, 250);
    roiParams[5].minRadius = 45;
    roiParams[5].maxRadius = 55;


    // 暂用,后面要逐个设置
    for (auto& p : roiParams) {
        // p.minRadius = 20.0;
        // p.maxRadius = 180.0;
        p.minCircularity = 0.1;
        p.minAspectRatio = 0.5;
        p.maxAspectRatio = 4.0;
    }

    g_roiParams = roiParams;

    // 在原图上绘制ROI矩形（调试用）
    cv::Mat result_img = raw_image.clone();
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)
    };
    for (size_t i = 0; i < roiParams.size(); ++i) {
        cv::rectangle(result_img, roiParams[i].roi, colors[i % colors.size()], 2);
        cv::putText(result_img, "ROI" + std::to_string(i + 1),
            cv::Point(roiParams[i].roi.x + 5, roiParams[i].roi.y + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i % colors.size()], 1);
    }

    // 执行检测
    RealImageFeatureDetector detector;
    detector.setROIParams(roiParams);

    // std::vector<cv::Point2f> points;
    // bool success = detector.detect(raw_image, points);

    std::vector<DetectedFeature> features;
    bool success = detector.detectWithEllipse(raw_image, features);

    // 新增调试：获取所有候选（仅当需要时，不影响原有输出）
    std::vector<std::vector<DetectedFeature>> allCandidates;
    detector.detectAllCandidates(raw_image, allCandidates);

    // 在绘制循环中，先画所有候选（用黄色或橙色），再画最佳（红色）
    
    for (size_t i = 0; i < allCandidates.size(); ++i) {
        for (const auto& cand : allCandidates[i]) {
            // 画候选椭圆（浅色，比如黄色）
            cv::ellipse(result_img, cand.ellipse, cv::Scalar(0, 255, 255), 1); // 黄色细线

            // ===== 新增：显示长轴、短轴、平均半径、圆度 =====
            std::string info = cv::format(
                "R=%.1f  L=%.1f  S=%.1f",
                cand.avgRadius,
                cand.ellipse.size.width,
                cand.ellipse.size.height
            );
            cv::putText(result_img, info,
                cv::Point((int)cand.point.x - 60, (int)cand.point.y - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            // ================================================
            // 
            // 画候选中心小点
            cv::circle(result_img, cv::Point((int)cand.point.x, (int)cand.point.y), 3, cv::Scalar(0, 255, 255), -1);
        }
    }





    g_detected_points.clear();
    for (const auto& feat : features) {
        g_detected_points.push_back(feat.point);
    }
    g_detection_success = success;

    if (success) {
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feat = features[i];
            // 绘制红色拟合椭圆（新增）
            cv::ellipse(result_img, feat.ellipse, cv::Scalar(0, 0, 255), 2);
            
            /*
            std::string axis_info = "W:" + std::to_string((int)feat.ellipse.size.width) +
                " H:" + std::to_string((int)feat.ellipse.size.height);
            cv::putText(result_img, axis_info,
                cv::Point((int)feat.point.x + 20, (int)feat.point.y - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            */

            // 绘制中心点（绿色大圆）
            cv::circle(result_img, cv::Point((int)feat.point.x, (int)feat.point.y), 12, cv::Scalar(0, 255, 0), 3);
            // 绘制十字（红色）
            cv::drawMarker(result_img, cv::Point((int)feat.point.x, (int)feat.point.y),
                cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
            // 序号
            cv::putText(result_img, std::to_string(i + 1),
                cv::Point((int)feat.point.x + 15, (int)feat.point.y - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        }
        std::cout << "\nDetection SUCCESS. Found " << features.size() << " points." << std::endl;
    }
    else {
        std::cout << "\nDetection FAILED. Check ROI parameters or image." << std::endl;
    }

    // 设置全局图像并启动窗口
    g_original_image = result_img;

    // 调试
    // 初始化仿射变换矩阵（初始为全图显示，居中）
    g_transform = cv::Mat::eye(2, 3, CV_64F);
    // 计算初始缩放，使图像完整显示在窗口中
    double scale_x = (double)g_window_size.width / g_original_image.cols;
    double scale_y = (double)g_window_size.height / g_original_image.rows;
    double init_scale = std::min(scale_x, scale_y);
    // 如果图像比窗口小，不放大；如果大，缩小到窗口尺寸
    if (init_scale < 1.0) {
        g_transform.at<double>(0, 0) = init_scale;
        g_transform.at<double>(1, 1) = init_scale;
    }
    // 居中显示
    g_transform.at<double>(0, 2) = (g_window_size.width - g_original_image.cols * init_scale) / 2.0;
    g_transform.at<double>(1, 2) = (g_window_size.height - g_original_image.rows * init_scale) / 2.0;

    // 然后创建窗口并设置回调
    cv::namedWindow("Feature Extraction + Grid + Zoom", cv::WINDOW_NORMAL);
    cv::resizeWindow("Feature Extraction + Grid + Zoom", g_window_size.width, g_window_size.height);
    cv::setMouseCallback("Feature Extraction + Grid + Zoom", onMouse);



    // g_center = cv::Point2d(g_original_image.cols / 2.0, g_original_image.rows / 2.0);
    g_scale = 1.0;
    g_need_update = true;

    // 初始化显示图像大小
    /*g_display_image = cv::Mat(800, 1200, g_original_image.type());

    cv::namedWindow("Feature Extraction + Grid + Zoom + Pan", cv::WINDOW_NORMAL);
    cv::resizeWindow("Feature Extraction + Grid + Zoom + Pan", 1200, 800);
    cv::setMouseCallback("Feature Extraction + Grid + Zoom + Pan", onMouse);
    */
    // 主循环
    while (true) {
        if (g_need_update) {
            updateDisplay();
        }
        char key = cv::waitKey(30);
        if (key == 27) break;  // ESC退出
        if (key == 'r' || key == 'R') {
            //g_center = cv::Point2d(g_original_image.cols / 2.0, g_original_image.rows / 2.0);
            g_scale = 1.0;
            g_need_update = true;
            std::cout << "[Reset] View reset to center." << std::endl;
        }
    }

    cv::destroyAllWindows();
    return 0;
}