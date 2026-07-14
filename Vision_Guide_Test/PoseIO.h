/*
将结果保存为 JSON 文件的类 PoseIO，提供了一个静态方法 savePose，用于将旋转向量、平移向量以及其他相关信息保存到指定的 JSON 文件中。
*/
#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <ctime>
#include <vector>
#include <nlohmann/json.hpp>
#include "Logger.h"

using json = nlohmann::json;

class PoseIO {
public:
    // 保存位姿到 JSON 文件
    static bool savePose(const std::string& filename,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        const std::string& description = "",
        double reprojection_error = 0.0) {
        try {
            json j;
            j["description"] = description;
            j["reprojection_error"] = reprojection_error;

            // 保存旋转向量Rvec
            j["rvec"] = {
                rvec.at<double>(0, 0),
                rvec.at<double>(1, 0),
                rvec.at<double>(2, 0)
            };

            // 保存平移向量Tvec
            j["tvec"] = {
                tvec.at<double>(0, 0),
                tvec.at<double>(1, 0),
                tvec.at<double>(2, 0)
            };

            // 保存旋转矩阵
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            j["R"] = {
                {R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2)},
                {R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2)},
                {R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2)}
            };

            std::ofstream file(filename);
            if (!file.is_open()) {
                LOG_ERROR("无法打开文件: {}", filename);
                return false;
            }
            file << j.dump(4);   // 将 JSON 对象以缩进格式写入文件
            file.close();

            std::cout << "位姿已保存到: " << filename << std::endl;
            LOG_INFO("位姿已保存到: {}", filename);
            return true;
        }
        catch (const std::exception& e) {
            LOG_ERROR("保存失败: {}", e.what());
            return false;
        }
    }
    
    // 从JSON中加载位姿
    static bool loadPose(const std::string& filename,
        cv::Mat& rvec,
        cv::Mat& tvec,
        std::string& description,
        double& reprojection_error) {
        try {
            std::ifstream file(filename);
            if (!file.is_open()) {
                LOG_ERROR("无法打开文件: {}", filename);
                return false;
            }

            json j;
            file >> j;      // 读取 JSON 文件内容
            file.close();

            // 读取旋转向量
            std::vector<double> rvec_data = j["rvec"];
            rvec = (cv::Mat_<double>(3, 1) <<
                rvec_data[0], rvec_data[1], rvec_data[2]);

            // 读取平移向量
            std::vector<double> tvec_data = j["tvec"];
            tvec = (cv::Mat_<double>(3, 1) <<
                tvec_data[0], tvec_data[1], tvec_data[2]);

            description = j.value("description", "");
            reprojection_error = j.value("reprojection_error", 0.0);

            LOG_INFO("[PoseIO] 位姿已加载: {}", filename);
            return true;
        }
        catch (const std::exception& e) {
            LOG_ERROR("[PoseIO] 加载失败: {}", e.what());
            return false;
        }
    }

    // 检查文件是否存在
    static bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
};
#pragma once
