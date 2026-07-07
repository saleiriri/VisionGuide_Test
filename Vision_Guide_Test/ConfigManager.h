#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include "FeatureDetectorBase.h"
#include "YoloDetector.h"

using json = nlohmann::json;
// 相机内参
struct CameraConfig {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        8730.097601293379, 0.0, 2032.5411458089795,
        0.0, 8729.93196788731, 1590.2239152932418,
        0.0, 0.0, 1.0);
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << -0.059419770213628355, 0.6242804643577333, 0.0023748112509786178, 0.0005345488879690347, -1.5262933488921209);
    int image_width = 4024;
    int image_height = 3046;
};

// 3D 点库（工件坐标系下的特征点）
struct ObjectPoint {
    float x, y, z;
};

// ROI（每个孔独立）
struct ROIConfig {
    cv::Rect roi;
    double minRadius = 20.0;
    double maxRadius = 80.0;
    double minCircularity = 0.3;
    double minAspectRatio = 0.3;
    double maxAspectRatio = 4.0;
};

// YOLO 配置
struct YoloConfigFile {
    std::string model_path = "";
    float confidence_threshold = 0.5;
    float nms_threshold = 0.4;
    int input_width = 640;
    int input_height = 640;
    int hole_class_id = 0;
    bool enable = false;  // 是否启用
};

// PLC 配置
struct PLCConfigFile {
    std::string ip = "192.168.0.1";
    int rack = 0;
    int slot = 1;
    int db_number = 100;
    int db_start_offset = 0;
    int timeout_ms = 3000;
    bool auto_reconnect = true;
    bool enable = false;  // 是否启用
};

// 完整配置文件
struct AppConfig {
    std::string description = "视觉引导定位参数配置";
    CameraConfig camera;
    std::vector<ObjectPoint> object_points;   // 3D 点库
    std::vector<ROIConfig> rois;              // ROI 列表
    YoloConfigFile yolo;
    PLCConfigFile plc;
    std::string standard_pose_file = "standard_pose.json";
    bool save_as_standard = false;            // 是否保存标准位姿
};


// 配置
class ConfigManager {
public:
    // 加载配置文件
    static bool loadConfig(const std::string& filename, AppConfig& config);

    // 保存配置文件为 JSON文件
    static bool saveConfigTemplate(const std::string& filename);

    // 将配置转换为 ROIParams（供 RealImageFeatureDetector 使用）
    static std::vector<ROIParams> convertToROIParams(const std::vector<ROIConfig>& rois);

    // 将配置转换为 YoloConfig（供 YoloDetector 使用）
    static YoloConfig convertToYoloConfig(const YoloConfigFile& cfg);

private:
    static void from_json(const json& j, CameraConfig& c);
    static void from_json(const json& j, ObjectPoint& p);
    static void from_json(const json& j, ROIConfig& r);
    static void from_json(const json& j, YoloConfigFile& y);
    static void from_json(const json& j, PLCConfigFile& p);
    static void from_json(const json& j, AppConfig& cfg);
};
