#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "FeatureDetectorBase.h"
#include "RealImageFeatureDetector.h"
#include "PoseSolver.h" 
#include "TraditionalDetector.h"
#include "YoloDetector.h"
#include "PoseIO.h"
#include "ConfigManager.h"
#include <spdlog/spdlog.h>
#include "Logger.h"


double g_scale = 1.0;              // 缩放倍数
// 用于平移、缩放的变量
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
    
    if (g_transform.empty() || g_transform.rows != 2 || g_transform.cols != 3 || g_transform.type() != CV_64F) {
    g_transform = cv::Mat::eye(2, 3, CV_64F);
    }
    static cv::Point2d drag_start;  // 拖拽起始点
    static cv::Mat start_transform; // 拖拽开始时的变换矩阵

    if (event == cv::EVENT_MOUSEWHEEL) {
        // 以鼠标位置为中心滚轮缩放
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
        // 拖拽平移
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
    // std::cout << "\nlytest\n" << std::endl;
    Logger::init("logs/vision_guide.log");

    LOG_INFO("程序启动成功！");
    
    // 生成配置模板只需运行一次，之后请修改 config.json调参
    // ConfigManager::saveConfigTemplate("config.json");
    // std::cout << "[系统] 配置模板已生成" << std::endl;
    // return 0;

    // 加载图像
    std::string image_path = "image\\test.png";  // 图片路径(调试暂用)
    cv::Mat raw_image = cv::imread(image_path);
    if (raw_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Image size: " << raw_image.cols << " x " << raw_image.rows << std::endl;

    // 配置6个ROI (需根据实际情况设置，可以修改位置、数量)
    std::vector<ROIParams> roiParams(6);
    // 单独调整每个ROI
    roiParams[0].roi = cv::Rect(800, 1100, 300, 250);
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

    // 是否使用yolo找点
    bool use_yolo = false;  // true=使用YOLO自动找ROI，false=使用手动ROI

    if (use_yolo) {
        YoloConfig config;
        config.model_path = "hole_detector.onnx";
        config.confidence_threshold = 0.5;
        config.nms_threshold = 0.4;

        YoloDetector yolo_detector(config);
        std::vector<cv::Rect> yolo_rois;

        if (yolo_detector.detectROIs(raw_image, yolo_rois)) {
            size_t count = std::min(roiParams.size(), yolo_rois.size());
            for (size_t i = 0; i < count; ++i) {
                roiParams[i].roi = yolo_rois[i];
                // minRadius、maxRadius 等参数保持不变
            }
            std::cout << "[YOLO] 已更新 " << count << " 个ROI" << std::endl;
        }
        else {
            std::cout << "[YOLO] 检测失败，使用手动ROI" << std::endl;
        }
    }


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
    //RealImageFeatureDetector detector;
    //detector.setROIParams(roiParams);      // 设置ROI参数

    // 添加传统及深度学习模型选择判断
    TraditionalDetector detector;     // 改用包装类

    // 当使用深度学习模型时，将上一行注释，并将下面三行解开注释即可
    // YoloConfig config;
    // config.model_path = "D:/models/hole_detector.onnx";
    // YoloDetector detector(config);
    
    detector.setROIParams(roiParams);

    std::vector<DetectedFeature> features;
    // bool success = detector.detectWithEllipse(raw_image, features);
    bool success = detector.detect(raw_image, features);    // 接口统一为detect

    // 调试，查看所有输出
    std::vector<std::vector<DetectedFeature>> allCandidates;
    detector.detectAllCandidates(raw_image, allCandidates);
    // 在绘制循环中，先画所有候选（黄色），再画最佳（红色）
    for (size_t i = 0; i < allCandidates.size(); ++i) {
        for (const auto& cand : allCandidates[i]) {
            // 画候选椭圆
            cv::ellipse(result_img, cand.ellipse, cv::Scalar(0, 255, 255), 1); // 黄色细线

            // 显示平均半径、长轴、短轴
            std::string info = cv::format(
                "R=%.1f  L=%.1f  S=%.1f",
                cand.avgRadius,
                cand.ellipse.size.width,
                cand.ellipse.size.height
            );
            cv::putText(result_img, info,
                cv::Point((int)cand.point.x - 60, (int)cand.point.y - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            // 画候选中心
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
            // 绘制红色拟合椭圆
            cv::ellipse(result_img, feat.ellipse, cv::Scalar(0, 0, 255), 2);

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

    // 2026.6.25
    // 提取 2D 点
    std::vector<cv::Point2f> image_points;
    for (const auto& feat : features) {
        image_points.push_back(feat.point);
        std::cout << "提取到点: (" << feat.point.x << ", " << feat.point.y << ")" << std::endl;
    }

    // PnP 解算
    // 定义 3D 点库（CAD坐标）
    std::vector<cv::Point3f> object_points = {
        cv::Point3f(2262.615f, -608.766f, 481.019f),
        cv::Point3f(2188.383f, -608.058f, 511.965f),
        cv::Point3f(2148.132f,  -609.187f,   588.711f),
        cv::Point3f(2132.353f,  -612.087f, 646.559f),
        cv::Point3f(2198.0f,  -591.0f, 755.0f),
        cv::Point3f(2193.449f,  -606.384f,  600.541f)
    };

    // 配置 PoseSolver
    PoseSolverConfig config;
    config.ransac_threshold = 3.0;
    config.match_tolerance_ratio = 0.04;

    PoseSolver solver(config);

    // 当前图片相机参数（调试用，需根据实际使用的相机内参做出对应修改）
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        8730.097601293379, 0.0, 2032.5411458089795, 0.0, 8729.93196788731, 1590.2239152932418, 0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << -0.059419770213628355, 0.6242804643577333, 0.0023748112509786178, 0.0005345488879690347, -1.5262933488921209);


    solver.setCameraParams(camera_matrix, dist_coeffs);
    solver.setObjectPoints(object_points);

    // 执行解算
    PoseResult result = solver.solveDebug(image_points);

    // 将结果保存为 JSON 文件
    // 标准位姿，应使用三坐标标准工件，保存结果为standard_pose.json，内容包括旋转向量、平移向量、旋转矩阵
    if (result.success) {
        bool save_as_standard = true;  // 设置为 true 时保存

        if (save_as_standard) {
            PoseIO::savePose(
                "standard_pose.json",
                result.rvec,
                result.tvec,
                "standard_pose",
                result.reprojection_error
            );
            std::cout << "标准位姿已保存到 standard_pose.json" << std::endl;
        }
    }
    else {
        std::cout << "PnP解算失败，无法保存标准位姿" << std::endl;
    }

    // pnp结果可视化（调试）
    // 绘制检测到的点（绿）
    for (const auto& pt : image_points) {
        cv::circle(result_img, cv::Point((int)pt.x, (int)pt.y), 8, cv::Scalar(0, 255, 0), -1);
    }

    // 解算成功时绘制重投影点（红色）
    if (result.success) {
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(object_points, result.rvec, result.tvec,
            camera_matrix, dist_coeffs, reprojected);
        for (const auto& pt : reprojected) {
            cv::circle(result_img, cv::Point((int)pt.x, (int)pt.y), 5, cv::Scalar(0, 0, 255), 2);
        }
    }


    // 设置全局图像并启动窗口
    g_original_image = result_img;

    // 调试
    // 初始化仿射变换矩阵
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