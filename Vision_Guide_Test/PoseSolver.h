#pragma once
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <functional>
#include <numeric>


// 位姿解算结果
struct PoseResult {
    bool success = false;              // 是否成功
    cv::Mat rvec;                      // 旋转向量
    cv::Mat tvec;                      // 平移向量
    cv::Mat R;                         // 旋转矩阵
    double reprojection_error = 0.0;   // 重投影误差（像素）
    int inlier_count = 0;              // RANSAC 内点数
    int matched_count = 0;             // 成功匹配的点数
    std::vector<int> matched_indices;  // 匹配到的 3D 点索引
    std::vector<int> inlier_indices;   // RANSAC 内点索引

    std::string solver_used;// 使用的解算器（EPNP、IPPE)
};


// 解算配置参数
struct PoseSolverConfig {
    // RANSAC 参数
    double ransac_threshold = 3.0;     // 重投影误差阈值（像素）
    int ransac_iterations = 100;       // 迭代次数
    double ransac_confidence = 0.99;   // 置信度

    // 距离匹配参数
    double match_tolerance_ratio = 0.04;

    // 非线性优化参数
    int lm_max_iterations = 30;        // LM 最大迭代次数
    double lm_epsilon = 0.001;         // LM 收敛精度

    // 最小点数要求
    int min_points_for_pnp = 4;        // PnP 最少需要 4 个点

    // 共面检测参数
    double coplanar_threshold_ratio = 0.05;   // 共面判定阈值（厚度/平面尺寸）
    bool enable_plane_detection = true;       // 是否启用共面检测

    // IPPE 共面场景专用参数
    int ippe_refine_iterations = 8;            // IPPE 精修迭代次数
    double ippe_refine_epsilon = 0.001;        // IPPE 精修收敛精度
    int ippe_ransac_samples = 100;             // IPPE RANSAC 采样次数
    bool enable_ippe_ransac = true;            // 是否启用 IPPE RANSAC
    double ippe_ransac_threshold = 2.5;        // IPPE RANSAC 重投影阈值
    double coplanar_distance_threshold = 1.0;  // 共面判定阈值（毫米）

    // 解算器策略
    bool prefer_ippe_for_plane = true;         // 共面时优先使用 IPPE
    bool fallback_to_iterative = true;         // 失败时回退到 ITERATIVE

    // 解的验证参数
    bool enable_pose_validation = true;        // 是否验证解的有效性
    double min_depth_mm = 100.0;               // 最小深度（毫米）
    double max_depth_mm = 5000.0;              // 最大深度（毫米）
    double max_rotation_deg = 45.0;            // 最大旋转角度（度）
};

// 位姿解算器（PNP解算+自动匹配）
class PoseSolver {
public:
    PoseSolver();
    explicit PoseSolver(const PoseSolverConfig& config);

    // 设置相机参数
    void setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

    // 设置 3D 点库
    void setObjectPoints(const std::vector<cv::Point3f>& object_points);

    // 输入 2D 点（可能少于 3D 点库），输出位姿
    PoseResult solve(const std::vector<cv::Point2f>& image_points);

    // 带调试信息的版本
    PoseResult solveDebug(const std::vector<cv::Point2f>& image_points);

    // 重置
    void reset();

    // 获取配置（用于调参）
    PoseSolverConfig& getConfig() { return config_; }

private:
    // 共勉检测
    bool arePointsCoplanar(const std::vector<cv::Point3f>& points, double& out_score);

    // 解的有效性验证
    bool validatePose(const cv::Mat& rvec, const cv::Mat& tvec);

    // 三种解算策略
    bool solveCoplanarPnP(
        const std::vector<cv::Point3f>& object_pts,
        const std::vector<cv::Point2f>& image_pts,
        PoseResult& result
    );

    bool solveGeneralPnP(
        const std::vector<cv::Point3f>& object_pts,
        const std::vector<cv::Point2f>& image_pts,
        PoseResult& result
    );

    bool solveIterativePnP(
        const std::vector<cv::Point3f>& object_pts,
        const std::vector<cv::Point2f>& image_pts,
        PoseResult& result
    );

    // IPPE RANSAC 辅助
    bool runIppeRansac(
        const std::vector<cv::Point3f>& object_pts,
        const std::vector<cv::Point2f>& image_pts,
        std::vector<int>& inliers
    );

    // 距离矩阵自动匹配
    bool matchByDistance(
        const std::vector<cv::Point3f>& object_3d_lib,
        const std::vector<cv::Point2f>& image_2d_pts,
        std::vector<cv::Point3f>& matched_3d,
        std::vector<cv::Point2f>& matched_2d,
        std::vector<int>& matched_indices
    );

    // 核心 PnP 解算（已重构）
    bool solvePnP(
        const std::vector<cv::Point3f>& object_pts,
        const std::vector<cv::Point2f>& image_pts,
        PoseResult& result
    );

    // 计算重投影误差
    double computeReprojectionError(
        const std::vector<cv::Point3f>& object_pts,
        const std::vector<cv::Point2f>& image_pts,
        const cv::Mat& rvec,
        const cv::Mat& tvec
    );

private:
    PoseSolverConfig config_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::vector<cv::Point3f> object_points_;  // 完整的 3D 点库
    bool is_initialized_ = false;
};