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
                std::cerr << "无法打开文件: " << filename << std::endl;
                return false;
            }
            file << j.dump(4);   // 将 JSON 对象以缩进格式写入文件
            file.close();

            std::cout << "位姿已保存到: " << filename << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "保存失败: " << e.what() << std::endl;
            return false;
        }
    }
};
#pragma once
