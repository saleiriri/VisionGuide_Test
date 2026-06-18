#include "MultiFramePoseAverager.h"

bool MultiFramePoseAverager::addPose(const cv::Mat& rvec, const cv::Mat& tvec, double reprojection_error) {
    // 防呆：剔除质量明显不佳的帧（例如振动或严重遮挡）
    if (reprojection_error > 3.0) {
        std::cout << "[Frame Rejected] Reprojection error too high: " << reprojection_error << std::endl;
        return false;
    }

    rvecs_.push_back(rvec.clone());
    tvecs_.push_back(tvec.clone());
    errors_.push_back(reprojection_error);

    std::cout << "[Frame Captured] " << rvecs_.size() << "/" << max_frames_
        << " (Reproj Err: " << reprojection_error << " px)" << std::endl;

    // 如果收集了足够的帧，执行融合计算
    if ((int)rvecs_.size() >= max_frames_) {
        computeFinalPose();
        return true; // 表示最终结果已就绪
    }
    return false; // 仍需继续采集
}

void MultiFramePoseAverager::computeFinalPose() {
    // 1. 平移向量取均值
    cv::Mat t_sum = cv::Mat::zeros(3, 1, CV_64F);
    for (const auto& t : tvecs_) {
        t_sum += t;
    }
    tvec_final_ = t_sum / (double)tvecs_.size();

    // 2. 旋转矩阵取平均（使用四元数 / 旋转矩阵线性平均 + SVD 正交化）
    cv::Mat R_sum = cv::Mat::zeros(3, 3, CV_64F);
    for (const auto& rvec : rvecs_) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        R_sum += R;
    }
    cv::Mat R_avg = R_sum / (double)rvecs_.size();

    // SVD 正交化，确保结果依然是合法的旋转矩阵（防止数值漂移）
    cv::Mat U, W, Vt;
    cv::SVDecomp(R_avg, W, U, Vt);
    R_avg = U * Vt; // 强制为正交矩阵

    cv::Rodrigues(R_avg, rvec_final_);

    std::cout << "\n========== Multi-Frame Fusion Complete ==========" << std::endl;
    std::cout << "Fused " << rvecs_.size() << " valid frames." << std::endl;
    std::cout << "Final Translation (X, Y, Z): "
        << tvec_final_.at<double>(0) << ", "
        << tvec_final_.at<double>(1) << ", "
        << tvec_final_.at<double>(2) << std::endl;

    // 清空缓存，准备下一轮采集
    rvecs_.clear();
    tvecs_.clear();
    errors_.clear();
}