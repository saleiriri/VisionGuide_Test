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


cv::Mat g_original_image;          // 带标记和ROI的原始图像（用于显示）
cv::Mat g_display_image;           // 当前显示图像（缩放/裁剪后）
double g_scale = 1.0;              // 缩放倍数
cv::Point2d g_center;              // 显示中心在原始图像中的坐标 (x,y)
bool g_need_update = true;         // 需要刷新显示

// 鼠标交互
cv::Point2d g_mouse_pos = cv::Point2d(-1, -1); 
bool g_dragging = false;
cv::Point2d g_drag_start;

// 检测参数与结果
std::vector<ROIParams> g_roiParams;
std::vector<cv::Point2f> g_detected_points;
bool g_detection_success = false;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    // 计算鼠标在原始图像中的坐标
    double img_x = (x - g_display_image.cols / 2.0) / g_scale + g_center.x;
    double img_y = (y - g_display_image.rows / 2.0) / g_scale + g_center.y;

    if (event == cv::EVENT_MOUSEMOVE) {
        // 更新鼠标位置
        if (img_x >= 0 && img_x < g_original_image.cols &&
            img_y >= 0 && img_y < g_original_image.rows) {
            g_mouse_pos = cv::Point2d(img_x, img_y);
        }
        else {
            g_mouse_pos = cv::Point2d(-1, -1);
        }

        // 如果正在拖拽，平移图像
        if (g_dragging) {
            double dx = img_x - g_drag_start.x;
            double dy = img_y - g_drag_start.y;
            g_center.x -= dx;
            g_center.y -= dy;
            g_drag_start = cv::Point2d(img_x, img_y);
            g_need_update = true;
        }
        g_need_update = true;
    }
    else if (event == cv::EVENT_LBUTTONDOWN) {
        g_dragging = true;
        g_drag_start = cv::Point2d(img_x, img_y);
        std::cout << "[Mouse] Start drag at (" << img_x << ", " << img_y << ")" << std::endl;
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        g_dragging = false;
        std::cout << "[Mouse] End drag." << std::endl;
    }
    else if (event == cv::EVENT_MOUSEWHEEL) {
        int delta = cv::getMouseWheelDelta(flags);
        double scale_factor = (delta > 0) ? 1.1 : 0.9;
        double new_scale = g_scale * scale_factor;
        new_scale = std::max(0.1, std::min(10.0, new_scale));

        // 以鼠标位置为中心缩放
        g_center.x = img_x;
        g_center.y = img_y;
        g_scale = new_scale;
        g_need_update = true;
        std::cout << "[Zoom] Scale: " << g_scale << ", center: (" << g_center.x << ", " << g_center.y << ")" << std::endl;
    }
}

void updateDisplay() {
    if (g_original_image.empty()) return;

    // 计算要裁剪的区域
    int half_w = (int)(g_display_image.cols / (2.0 * g_scale));
    int half_h = (int)(g_display_image.rows / (2.0 * g_scale));
    cv::Rect roi;
    roi.x = (int)(g_center.x - half_w);
    roi.y = (int)(g_center.y - half_h);
    roi.width = 2 * half_w;
    roi.height = 2 * half_h;

    // 边界保护
    if (roi.x < 0) roi.x = 0;
    if (roi.y < 0) roi.y = 0;
    if (roi.x + roi.width > g_original_image.cols) roi.width = g_original_image.cols - roi.x;
    if (roi.y + roi.height > g_original_image.rows) roi.height = g_original_image.rows - roi.y;

    cv::Mat cropped;
    if (roi.width > 0 && roi.height > 0) {
        cropped = g_original_image(roi).clone();
    }
    else {
        cropped = g_original_image.clone();
    }

    // 缩放至显示窗口大小
    if (g_scale != 1.0) {
        cv::resize(cropped, cropped, cv::Size(g_display_image.cols, g_display_image.rows), 0, 0, cv::INTER_LINEAR);
    }
    else {
        if (cropped.size() != g_display_image.size())
            cv::resize(cropped, cropped, g_display_image.size());
    }

    if (g_mouse_pos.x >= 0 && g_mouse_pos.y >= 0) {
        // 将鼠标位置映射到当前显示图像坐标
        double disp_x = (g_mouse_pos.x - g_center.x) * g_scale + g_display_image.cols / 2.0;
        double disp_y = (g_mouse_pos.y - g_center.y) * g_scale + g_display_image.rows / 2.0;

        if (disp_x >= 0 && disp_x < g_display_image.cols &&
            disp_y >= 0 && disp_y < g_display_image.rows) {
            cv::Point pt((int)disp_x, (int)disp_y);
            cv::line(cropped, cv::Point(pt.x - 15, pt.y), cv::Point(pt.x + 15, pt.y), cv::Scalar(0, 0, 255), 1);
            cv::line(cropped, cv::Point(pt.x, pt.y - 15), cv::Point(pt.x, pt.y + 15), cv::Scalar(0, 0, 255), 1);
            cv::circle(cropped, pt, 3, cv::Scalar(0, 0, 255), -1);
            std::string coord_text = "(" + std::to_string((int)g_mouse_pos.x) + ", " + std::to_string((int)g_mouse_pos.y) + ")";
            cv::putText(cropped, coord_text, cv::Point(pt.x + 20, pt.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
    }

    // 辅助信息（缩放倍数、操作提示）
    cv::putText(cropped, "Zoom: " + std::to_string(g_scale), cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(cropped, "Mouse: move to see coords | Left-drag to pan | Scroll to zoom | R to reset",
        cv::Point(10, cropped.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    g_display_image = cropped;
    cv::imshow("Feature Extraction + Grid + Zoom + Pan", g_display_image);
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
    // 
    roiParams[0].roi = cv::Rect(800, 1100, 300, 250);
    roiParams[1].roi = cv::Rect(800, 1100, 300, 250);
    roiParams[2].roi = cv::Rect(800, 1100, 300, 250);
    roiParams[3].roi = cv::Rect(800, 1100, 300, 250);
    roiParams[4].roi = cv::Rect(800, 1100, 300, 250);
    roiParams[5].roi = cv::Rect(800, 1100, 300, 250);

    // 检测参数，单独调整每个ROI
    for (auto& p : roiParams) {
        p.minRadius = 20.0;
        p.maxRadius = 60.0;
        p.minCircularity = 0.5;
        p.minAspectRatio = 0.5;
        p.maxAspectRatio = 2.0;
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

    std::vector<cv::Point2f> points;
    bool success = detector.detect(raw_image, points);
    g_detected_points = points;
    g_detection_success = success;

    if (success) {
        for (size_t i = 0; i < points.size(); ++i) {
            cv::circle(result_img, cv::Point((int)points[i].x, (int)points[i].y), 12, cv::Scalar(0, 255, 0), 3);
            cv::drawMarker(result_img, cv::Point((int)points[i].x, (int)points[i].y),
                cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
            cv::putText(result_img, std::to_string(i + 1),
                cv::Point((int)points[i].x + 15, (int)points[i].y - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        }
        std::cout << "\nDetection SUCCESS. Found " << points.size() << " points." << std::endl;
    }
    else {
        std::cout << "\nDetection FAILED. Check ROI parameters or image." << std::endl;
    }

    // 设置全局图像并启动窗口
    g_original_image = result_img;
    g_center = cv::Point2d(g_original_image.cols / 2.0, g_original_image.rows / 2.0);
    g_scale = 1.0;
    g_need_update = true;

    // 初始化显示图像大小
    g_display_image = cv::Mat(800, 1200, g_original_image.type());

    cv::namedWindow("Feature Extraction + Grid + Zoom + Pan", cv::WINDOW_NORMAL);
    cv::resizeWindow("Feature Extraction + Grid + Zoom + Pan", 1200, 800);
    cv::setMouseCallback("Feature Extraction + Grid + Zoom + Pan", onMouse);

    // 主循环
    while (true) {
        if (g_need_update) {
            updateDisplay();
        }
        char key = cv::waitKey(30);
        if (key == 27) break;  // ESC退出
        if (key == 'r' || key == 'R') {
            g_center = cv::Point2d(g_original_image.cols / 2.0, g_original_image.rows / 2.0);
            g_scale = 1.0;
            g_need_update = true;
            std::cout << "[Reset] View reset to center." << std::endl;
        }
    }

    cv::destroyAllWindows();
    return 0;
}