#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class MultiFramePoseAverager {
private:
    std::vector<cv::Mat> rvecs_;
    std::vector<cv::Mat> tvecs_;
    std::vector<double> errors_;
    int max_frames_;
    cv::Mat rvec_final_;
    cv::Mat tvec_final_;

public:
    // 构造函数：设置最大帧数（建议 8~12 帧）
    MultiFramePoseAverager(int max_frames = 10) : max_frames_(max_frames) {}

    // 向平均器添加一帧位姿计算结果
    // 返回 true 表示已集满帧并计算出最终结果
    bool addPose(const cv::Mat& rvec, const cv::Mat& tvec, double reprojection_error);

    // 获取最终融合结果（必须在 addPose 返回 true 后调用）
    cv::Mat getFinalRvec() const { return rvec_final_.clone(); }
    cv::Mat getFinalTvec() const { return tvec_final_.clone(); }

    // 清空缓存（用于重新开始测量）
    void reset() {
        rvecs_.clear();
        tvecs_.clear();
        errors_.clear();
        rvec_final_ = cv::Mat();
        tvec_final_ = cv::Mat();
    }

private:
    void computeFinalPose();
};