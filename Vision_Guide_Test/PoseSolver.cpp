/*
当前存在的问题：
若检测到的3D点共面，则PnP解算可能会失败，或者解算结果不稳定。
因此需要在使用PnP解算前，先检查匹配的3D点是否共面

由于特征点为白车身孔，不会出现完全共面，因此不考虑共面情况，或者在选点时，尽量选择不共面的点。
*/

#include "PoseSolver.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <random>


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
bool PoseSolver::arePointsCoplanar(const std::vector<cv::Point3f>& points, double& out_score) {
    if (points.size() < 4) {
        out_score = 1.0;
        return false;
    }

    cv::Point3f centroid(0, 0, 0);
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= (float)points.size();

    cv::Mat_<double> cov = cv::Mat_<double>::zeros(3, 3);
    for (const auto& p : points) {
        cv::Vec3d v(p.x - centroid.x, p.y - centroid.y, p.z - centroid.z);
        cov += v * v.t();
    }
    cov /= (double)points.size();

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    double lambda_min = eigenvalues.at<double>(2, 0);
    double lambda_max = eigenvalues.at<double>(0, 0);

    if (lambda_max < 1e-10) {
        out_score = 0.0;
        return true;
    }

    out_score = lambda_min / lambda_max;
    return out_score < config_.coplanar_threshold_ratio;
}

// 解的有效性验证
bool PoseSolver::validatePose(const cv::Mat& rvec, const cv::Mat& tvec) {
    if (!config_.enable_pose_validation) return true;

    // 检查旋转矩阵行列式
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    double det = cv::determinant(R);
    if (std::abs(det - 1.0) > 0.01) {
        std::cerr << "[PnP] 无效旋转矩阵，det(R)=" << det << std::endl;
        return false;
    }

    // 检查深度
    double depth = tvec.at<double>(2);
    if (depth < config_.min_depth_mm || depth > config_.max_depth_mm) {
        std::cerr << "[PnP] 深度异常，t.z=" << depth << " mm" << std::endl;
        return false;
    }

    // 检查旋转角度
    double angle_deg = cv::norm(rvec) * 180.0 / CV_PI;
    if (angle_deg > config_.max_rotation_deg) {
        std::cerr << "[PnP] 旋转角度过大，angle=" << angle_deg << "°" << std::endl;
        return false;
    }

    return true;
}

bool PoseSolver::runIppeRansac(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    std::vector<int>& inliers) {

    int n_points = (int)object_pts.size();
    if (n_points < 4) {
        std::cerr << "[IPPE RANSAC] 点数不足: " << n_points << std::endl;
        return false;
    }

    inliers.clear();
    std::vector<int> best_inliers;
    double best_error = std::numeric_limits<double>::max();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n_points - 1);

    int num_samples = config_.ippe_ransac_samples;
    if (n_points <= 6) num_samples = std::min(num_samples, 50);

    std::cout << "[IPPE RANSAC] 采样 " << num_samples << " 次，总点数=" << n_points << std::endl;

    // 检查4个点是否适合 IPPE（必须共面且不共线）
    auto isDegenerateForIPPE = [](const std::vector<cv::Point3f>& pts) -> bool {
        if (pts.size() != 4) return true;

        // 取前三点构成平面
        cv::Point3f v1 = pts[1] - pts[0];
        cv::Point3f v2 = pts[2] - pts[0];
        cv::Point3f normal = v1.cross(v2);
        double normLen = cv::norm(normal);

        if (normLen < 1e-6) return true;  // 三点共线或重合，无法构成平面

        // 计算第四点到平面的距离
        double d = std::abs(normal.dot(pts[3] - pts[0])) / normLen;

        // 如果距离 > 0.1mm，说明不共面 → IPPE 不适用
        return d > 0.1;
        };


    // 随机采样
    for (int iter = 0; iter < num_samples; ++iter) {
        // 随机选 4 个不重复的点
        std::vector<int> sample;
        while (sample.size() < 4) {
            int idx = dis(gen);
            if (std::find(sample.begin(), sample.end(), idx) == sample.end()) {
                sample.push_back(idx);
            }
        }

        // 构造采样点
        std::vector<cv::Point3f> sample_3d;
        std::vector<cv::Point2f> sample_2d;
        for (int idx : sample) {
            sample_3d.push_back(object_pts[idx]);
            sample_2d.push_back(image_pts[idx]);
        }

        // 检查是否适合 IPPE
        if (isDegenerateForIPPE(sample_3d)) {
            continue;  // 不适合 IPPE，跳过这个样本
        }

        // 尝试 IPPE 解算
        cv::Mat rvec_temp, tvec_temp;
        bool ok = false;

        try {
            ok = cv::solvePnP(
                sample_3d, sample_2d, camera_matrix_, dist_coeffs_,
                rvec_temp, tvec_temp, false, cv::SOLVEPNP_IPPE
            );
        }
        catch (const cv::Exception& e) {
            std::cerr << "[IPPE RANSAC] solvePnP(IPPE) 异常: " << e.what() << std::endl;
            continue;
        }

        if (!ok) continue;

        // 验证解的有效性
        if (!validatePose(rvec_temp, tvec_temp)) continue;

        // 计算所有点的重投影误差
        std::vector<cv::Point2f> projected;
        try {
            cv::projectPoints(object_pts, rvec_temp, tvec_temp,
                camera_matrix_, dist_coeffs_, projected);
        }
        catch (const cv::Exception& e) {
            std::cerr << "[IPPE RANSAC] projectPoints 异常: " << e.what() << std::endl;
            continue;
        }

        if (projected.size() != (size_t)n_points) {
            std::cerr << "[IPPE RANSAC] projectPoints 输出尺寸不匹配: "
                << projected.size() << " vs " << n_points << std::endl;
            continue;
        }

        // ===== 统计内点 =====
        std::vector<int> inlier_candidates;
        double total_error = 0.0;
        for (int idx = 0; idx < n_points; ++idx) {
            double err = cv::norm(image_pts[idx] - projected[idx]);
            if (err < config_.ippe_ransac_threshold) {
                inlier_candidates.push_back(idx);
                total_error += err;
            }
        }

        // ===== 更新最优解 =====
        if (inlier_candidates.size() >= 4) {
            double avg_error = total_error / inlier_candidates.size();
            if (avg_error < best_error) {
                best_error = avg_error;
                best_inliers = inlier_candidates;
                std::cout << "[IPPE RANSAC] 找到更好解: 内点数="
                    << inlier_candidates.size()
                    << ", 误差=" << avg_error << std::endl;
            }
        }
    }

    // 返回结果
    if (best_inliers.size() < 4) {
        std::cerr << "[IPPE RANSAC] 未找到有效解" << std::endl;
        return false;
    }

    inliers = best_inliers;
    std::cout << "[IPPE RANSAC] 完成，内点数=" << inliers.size()
        << ", 误差=" << best_error << std::endl;
    return true;
}

// IPPE + RANSAC 解算共面点的PnP问题
bool PoseSolver::solveCoplanarPnP(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    PoseResult& result) {
    result.solver_used = "IPPE + RANSAC";

    std::vector<int> inliers;
    bool success = false;

    if (config_.enable_ippe_ransac) {
        success = runIppeRansac(object_pts, image_pts, inliers);
    }
    else {
        inliers.resize(object_pts.size());
        std::iota(inliers.begin(), inliers.end(), 0);
        success = true;
    }

    if (!success || inliers.size() < 4) {
        if (config_.fallback_to_iterative) {
            return solveIterativePnP(object_pts, image_pts, result);
        }
        return false;
    }

    std::vector<cv::Point3f> inlier_3d;
    std::vector<cv::Point2f> inlier_2d;
    for (int idx : inliers) {
        inlier_3d.push_back(object_pts[idx]);
        inlier_2d.push_back(image_pts[idx]);
    }

    success = cv::solvePnP(
        inlier_3d, inlier_2d, camera_matrix_, dist_coeffs_,
        result.rvec, result.tvec, false, cv::SOLVEPNP_IPPE
    );

    if (!success) {
        if (config_.fallback_to_iterative) {
            return solveIterativePnP(inlier_3d, inlier_2d, result);
        }
        return false;
    }

    if (!validatePose(result.rvec, result.tvec)) {
        if (config_.fallback_to_iterative) {
            return solveIterativePnP(inlier_3d, inlier_2d, result);
        }
        return false;
    }

    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
        config_.ippe_refine_iterations,
        config_.ippe_refine_epsilon
    );
    cv::solvePnP(
        inlier_3d, inlier_2d, camera_matrix_, dist_coeffs_,
        result.rvec, result.tvec, true, cv::SOLVEPNP_ITERATIVE
    );

    cv::Rodrigues(result.rvec, result.R);
    result.reprojection_error = computeReprojectionError(
        object_pts, image_pts, result.rvec, result.tvec
    );
    result.success = true;
    result.inlier_count = (int)inliers.size();
    result.matched_count = (int)object_pts.size();
    result.inlier_indices = inliers;

    std::cout << "[PnP] IPPE 解算成功，内点数=" << inliers.size()
        << "，重投影误差=" << result.reprojection_error << " px" << std::endl;
    return true;
}

// EPNP + RANSAC 解算非共面点的PnP问题
bool PoseSolver::solveGeneralPnP(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    PoseResult& result) {
    result.solver_used = "EPNP + RANSAC";

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
        if (config_.fallback_to_iterative) {
            return solveIterativePnP(object_pts, image_pts, result);
        }
        return false;
    }

    result.inlier_count = (int)inliers.size();
    result.inlier_indices = inliers;

    std::vector<cv::Point3f> inlier_3d;
    std::vector<cv::Point2f> inlier_2d;
    for (int idx : inliers) {
        inlier_3d.push_back(object_pts[idx]);
        inlier_2d.push_back(image_pts[idx]);
    }

    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
        config_.lm_max_iterations,
        config_.lm_epsilon
    );

    success = cv::solvePnP(
        inlier_3d, inlier_2d, camera_matrix_, dist_coeffs_,
        result.rvec, result.tvec, true, cv::SOLVEPNP_ITERATIVE
    );

    if (!success) return false;

    if (!validatePose(result.rvec, result.tvec)) return false;

    cv::Rodrigues(result.rvec, result.R);
    result.reprojection_error = computeReprojectionError(
        object_pts, image_pts, result.rvec, result.tvec
    );
    result.success = true;
    result.matched_count = (int)object_pts.size();

    std::cout << "[PnP] EPNP 解算成功，内点数=" << inliers.size()
        << "，重投影误差=" << result.reprojection_error << " px" << std::endl;
    return true;
}

// IPPE + RANSAC 解算共面点的PnP问题
bool PoseSolver::solveIterativePnP(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    PoseResult& result) {
    result.solver_used = "ITERATIVE (回退)";

    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
        config_.lm_max_iterations,
        config_.lm_epsilon
    );

    bool success = cv::solvePnP(
        object_pts, image_pts, camera_matrix_, dist_coeffs_,
        result.rvec, result.tvec, false, cv::SOLVEPNP_ITERATIVE
    );

    if (!success) return false;
    if (!validatePose(result.rvec, result.tvec)) return false;

    cv::Rodrigues(result.rvec, result.R);
    result.reprojection_error = computeReprojectionError(
        object_pts, image_pts, result.rvec, result.tvec
    );
    result.success = true;
    result.inlier_count = (int)object_pts.size();
    result.matched_count = (int)object_pts.size();

    std::cout << "[PnP] ITERATIVE 解算成功（回退），"
        << "重投影误差=" << result.reprojection_error << " px" << std::endl;
    return true;
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

bool PoseSolver::solvePnP(
    const std::vector<cv::Point3f>& object_pts,
    const std::vector<cv::Point2f>& image_pts,
    PoseResult& result)
{
    if (object_pts.size() < 4 || image_pts.size() < 4) {
        return false;
    }

        // 直接调用 solvePnP + IPPE
        bool success = cv::solvePnP(
            object_pts, image_pts, camera_matrix_, dist_coeffs_,
            result.rvec, result.tvec, false, cv::SOLVEPNP_IPPE
        );

        if (!success) {
            // IPPE 失败（可能是 4 点共线），回退到 ITERATIVE
            std::cerr << "[PnP] IPPE 解算失败，回退到 ITERATIVE" << std::endl;
            success = cv::solvePnP(
                object_pts, image_pts, camera_matrix_, dist_coeffs_,
                result.rvec, result.tvec, false, cv::SOLVEPNP_ITERATIVE
            );
        }

        if (!success) {
            std::cerr << "[PnP] 所有解算器均失败" << std::endl;
            return false;
        }

        // 轻量级 LM 精修（只做 5 次迭代，避免过度拟合）
        cv::TermCriteria criteria(
            cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
            5,  // ← 只迭代 5 次
            0.001
        );
        cv::solvePnP(
            object_pts, image_pts, camera_matrix_, dist_coeffs_,
            result.rvec, result.tvec, true, cv::SOLVEPNP_ITERATIVE
        );

        // 转换为旋转矩阵
        cv::Rodrigues(result.rvec, result.R);

        // 计算重投影误差
        result.reprojection_error = computeReprojectionError(
            object_pts, image_pts, result.rvec, result.tvec
        );

        result.success = true;
        result.matched_count = (int)object_pts.size();
        result.inlier_count = (int)object_pts.size();  // IPPE 没有 RANSAC，所有点都是内点

        std::cout << "[PnP] ✅ IPPE 解算成功，重投影误差="
            << result.reprojection_error << " px" << std::endl;

        return true;
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