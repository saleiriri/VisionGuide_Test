/*
当前存在的问题：
若检测到的3D点共面，则PnP解算可能会失败，或者解算结果不稳定。
因此需要在使用PnP解算前，先检查匹配的3D点是否共面

由于特征点为白车身孔，不会出现完全共面，因此不考虑共面情况，或者在选点时，尽量选择不共面的点。
*/

#include "PoseSolver.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>


// 构造函数
PoseSolver::PoseSolver() {
    config_ = PoseSolverConfig();
}

PoseSolver::PoseSolver(const PoseSolverConfig& config)
    : config_(config) {
}


// 设置相机参数

void PoseSolver::setCameraParams(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    camera_matrix_ = camera_matrix.clone();
    dist_coeffs_ = dist_coeffs.clone();
    is_initialized_ = true;
}

// 设置 3D 点库
void PoseSolver::setObjectPoints(const std::vector<cv::Point3f>& object_points) {
    object_points_ = object_points;
    if (object_points_.size() < 4) {
        std::cerr << "3D点库少于4个点，无法解算PnP" << std::endl;
    }
}

// 检查点是否共面，特征点共面会导致PnP解算失败或不稳定
bool arePointsCoplanar(const std::vector<cv::Point3f>& points, double threshold = 0.01) {
    if (points.size() < 4) {
        // 少于4个点无法判断，PnP也无法解算
        return false;
    }

    // 计算质心
    cv::Point3f centroid(0, 0, 0);
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= (float)points.size();

    // 构建协方差矩阵
    cv::Mat_<double> cov = cv::Mat_<double>::zeros(3, 3);
    for (const auto& p : points) {
        cv::Vec3d v(p.x - centroid.x, p.y - centroid.y, p.z - centroid.z);
        cov += v * v.t();
    }
    cov /= (double)points.size();

    // 计算特征值
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);        // eigenvalues 按降序排列

    // 最小特征值 / 最大特征值，如果比值小于阈值，则认为共面
    double lambda_min = eigenvalues.at<double>(2, 0);
    double lambda_max = eigenvalues.at<double>(0, 0);
    // 防止除以 0
    if (lambda_max < 1e-10) {
        return true;  // 所有点重合，视为退化
    }

    return (lambda_min / lambda_max) < threshold;
}



void PoseSolver::reset() {
    // 不重置参数，只清空缓存
}

// 距离矩阵自动匹配
bool PoseSolver::matchByDistance(
    const std::vector<cv::Point3f>& object_3d_lib,
    const std::vector<cv::Point2f>& image_2d_pts,
    std::vector<cv::Point3f>& matched_3d,
    std::vector<cv::Point2f>& matched_2d,
    std::vector<int>& matched_indices)
{
    matched_3d.clear();
    matched_2d.clear();
    matched_indices.clear();

    int N_3d = (int)object_3d_lib.size();
    int N_2d = (int)image_2d_pts.size();

    if (N_3d < 4 || N_2d < 4) {
        std::cerr << "点数不足：3D库=" << N_3d << ", 2D点=" << N_2d << std::endl;
        return false;
    }

    // 计算 3D 距离矩阵
    std::vector<std::vector<double>> dist_3d(N_3d, std::vector<double>(N_3d, 0.0));
    for (int i = 0; i < N_3d; ++i) {
        for (int j = i + 1; j < N_3d; ++j) {
            double d = cv::norm(object_3d_lib[i] - object_3d_lib[j]);
            dist_3d[i][j] = dist_3d[j][i] = d;
        }
    }

    // 计算 2D 距离矩阵
    std::vector<std::vector<double>> dist_2d(N_2d, std::vector<double>(N_2d, 0.0));
    for (int i = 0; i < N_2d; ++i) {
        for (int j = i + 1; j < N_2d; ++j) {
            double d = cv::norm(image_2d_pts[i] - image_2d_pts[j]);
            dist_2d[i][j] = dist_2d[j][i] = d;
        }
    }

    // 暴力匹配所有组合（如果匹配点数过多时会耽误节拍，需要调整）
    std::vector<int> best_indices;
    double best_error = std::numeric_limits<double>::max();

    // 如果点数相同，直接匹配
    if (N_3d == N_2d) {
        best_indices.resize(N_2d);
        std::iota(best_indices.begin(), best_indices.end(), 0);
        best_error = 0.0;
    }
    else {
        // 递归枚举组合
        std::vector<int> selected;
        std::function<void(int, int)> dfs = [&](int start, int depth) {
            if (depth == N_2d) {
                double total_error = 0.0;
                int pair_count = 0;
                for (int i = 0; i < N_2d; ++i) {
                    for (int j = i + 1; j < N_2d; ++j) {
                        int idx_i = selected[i];
                        int idx_j = selected[j];
                        double d_3d = dist_3d[idx_i][idx_j];
                        double d_2d = dist_2d[i][j];
                        if (d_3d > 0.001 && d_2d > 0.001) {
                            double ratio_err = std::abs(1.0 - d_2d / d_3d);
                            total_error += ratio_err;
                            pair_count++;
                        }
                    }
                }
                if (pair_count > 0) {
                    total_error /= pair_count;
                    if (total_error < best_error) {
                        best_error = total_error;
                        best_indices = selected;
                    }
                }
                return;
            }
            for (int i = start; i < N_3d; ++i) {
                selected.push_back(i);
                dfs(i + 1, depth + 1);
                selected.pop_back();
            }
            };
        dfs(0, 0);
    }

    // 验证结果
    if (best_indices.empty() || best_error > config_.match_tolerance_ratio) {
        std::cerr << "[PoseSolver] 匹配失败，误差=" << best_error
            << " (阈值=" << config_.match_tolerance_ratio << ")" << std::endl;
        return false;
    }

    // 输出
    for (int idx : best_indices) {
        matched_3d.push_back(object_3d_lib[idx]);
        matched_indices.push_back(idx);
    }
    matched_2d = image_2d_pts;

    std::cout << "[PoseSolver]匹配成功，误差=" << best_error
        << "，匹配索引: ";
    for (int idx : best_indices) std::cout << idx << " ";
    std::cout << std::endl;

    return matched_3d.size() >= 4;
}

// 计算重投影误差
double PoseSolver::computeReprojectionError(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    const cv::Mat& rvec,
    const cv::Mat& tvec)
{
    if (object_pts.empty() || image_pts.empty()) return 0.0;

    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_pts, rvec, tvec, camera_matrix_, dist_coeffs_, projected);

    double total_error = 0.0;
    for (size_t i = 0; i < image_pts.size(); ++i) {
        total_error += cv::norm(image_pts[i] - projected[i]);
    }
    return total_error / image_pts.size();
}

//  PnP 解算
bool PoseSolver::solvePnP(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    PoseResult& result)
{
    if (object_pts.size() < 4 || image_pts.size() < 4) {
        return false;
    }

    // 检查 3D 点是否共面
    bool is_coplanar = arePointsCoplanar(object_pts, 0.01);
    if (is_coplanar) {
        std::cerr << "[PnP] 错误：3D 点共面！" << std::endl;
        std::cerr << "[PnP] 确保选取的 3D 点不在同一平面上。" << std::endl;
        result.success = false;
        result.reprojection_error = -1.0;
        // return false;
    }

    // RANSAC + EPNP
    std::vector<int> inliers;
    bool success = cv::solvePnPRansac(
        object_pts, image_pts, camera_matrix_, dist_coeffs_,
        result.rvec, result.tvec, false,
        config_.ransac_iterations,
        config_.ransac_threshold,
        config_.ransac_confidence,
        inliers,
        cv::SOLVEPNP_EPNP
    );

    if (!success || inliers.size() < 4) {
        std::cerr << "[PoseSolver] RANSAC 失败，内点数=" << inliers.size() << std::endl;
        return false;
    }

    result.inlier_count = (int)inliers.size();
    result.inlier_indices = inliers;

    // 提取内点
    std::vector<cv::Point3f> inlier_3d;
    std::vector<cv::Point2f> inlier_2d;
    for (int idx : inliers) {
        inlier_3d.push_back(object_pts[idx]);
        inlier_2d.push_back(image_pts[idx]);
    }

    // LM 非线性优化
    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
        config_.lm_max_iterations,
        config_.lm_epsilon
    );

    success = cv::solvePnP(
        inlier_3d, inlier_2d, camera_matrix_, dist_coeffs_,
        result.rvec, result.tvec, true, cv::SOLVEPNP_ITERATIVE
    );

    if (!success) {
        std::cerr << "[PoseSolver] LM 优化失败" << std::endl;
        return false;
    }

    // 转换为旋转矩阵
    cv::Rodrigues(result.rvec, result.R);

    // 计算重投影误差
    result.reprojection_error = computeReprojectionError(
        object_pts, image_pts, result.rvec, result.tvec
    );

    result.success = true;
    result.matched_count = (int)object_pts.size();

    return true;
}

// 自动匹配 + PnP 解算
PoseResult PoseSolver::solve(const std::vector<cv::Point2f>& image_points) {
    PoseResult result;

    if (!is_initialized_) {
        std::cerr << "相机内参未初始化" << std::endl;
        return result;
    }

    if (object_points_.empty()) {
        std::cerr << "未设置 3D 点库" << std::endl;
        return result;
    }

    if (image_points.size() < 4) {
        std::cerr << "2D 点不足：" << image_points.size() << " < 4" << std::endl;
        return result;
    }

    // 自动匹配
    std::vector<cv::Point3f> matched_3d;
    std::vector<cv::Point2f> matched_2d;
    std::vector<int> matched_indices;

    bool match_ok = matchByDistance(
        object_points_, image_points,
        matched_3d, matched_2d, matched_indices
    );

    if (!match_ok || matched_3d.size() < 4) {
        std::cerr << "距离匹配失败" << std::endl;
        return result;
    }

    result.matched_indices = matched_indices;
    result.matched_count = (int)matched_3d.size();

    // PnP 解算
    bool pnp_ok = solvePnP(matched_3d, matched_2d, result);

    if (!pnp_ok) {
        std::cerr << "PnP 解算失败" << std::endl;
        return result;
    }

    result.success = true;
    return result;
}


// 调试用
PoseResult PoseSolver::solveDebug(const std::vector<cv::Point2f>& image_points) {
    std::cout << "\n========== PoseSolver Debug ==========" << std::endl;
    std::cout << "输入 2D 点数: " << image_points.size() << std::endl;
    std::cout << "3D 点库大小: " << object_points_.size() << std::endl;

    auto result = solve(image_points);

    if (result.success) {
        std::cout << "  解算成功！" << std::endl;
        std::cout << "  平移向量 (mm): " << result.tvec.t() << std::endl;
        std::cout << "  旋转向量: " << result.rvec.t() << std::endl;
        std::cout << "  重投影误差: " << result.reprojection_error << " 像素" << std::endl;
        std::cout << "  内点数: " << result.inlier_count << std::endl;
        std::cout << "  匹配点数: " << result.matched_count << std::endl;
    }
    else {
        std::cout << "  解算失败" << std::endl;
    }

    return result;
}