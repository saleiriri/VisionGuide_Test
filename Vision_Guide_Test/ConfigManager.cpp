#include "ConfigManager.h"
#include <iostream>

// 读取 JSON 配置文件到结构体
void ConfigManager::from_json(const json& j, CameraConfig& c) {
    if (j.contains("fx")) c.camera_matrix.at<double>(0, 0) = j["fx"];
    if (j.contains("fy")) c.camera_matrix.at<double>(1, 1) = j["fy"];
    if (j.contains("cx")) c.camera_matrix.at<double>(0, 2) = j["cx"];
    if (j.contains("cy")) c.camera_matrix.at<double>(1, 2) = j["cy"];
    if (j.contains("image_width")) c.image_width = j["image_width"];
    if (j.contains("image_height")) c.image_height = j["image_height"];
    if (j.contains("dist_coeffs")) {
        auto dc = j["dist_coeffs"];
        if (dc.is_array() && dc.size() >= 5) {
            c.dist_coeffs = (cv::Mat_<double>(1, 5) <<
                dc[0].get<double>(),   // k1
                dc[1].get<double>(),   // k2
                dc[2].get<double>(),   // p1
                dc[3].get<double>(),   // p2
                dc[4].get<double>()    // k3
                );
        }
        else if (dc.is_array() && dc.size() == 4) {
            // 4参数畸变模型（k1, k2, p1, p2）
            c.dist_coeffs = (cv::Mat_<double>(1, 4) <<
                dc[0].get<double>(), dc[1].get<double>(),
                dc[2].get<double>(), dc[3].get<double>()
                );
        }
        else {
            std::cerr << "[ConfigManager] dist_coeffs 格式错误，使用默认畸变（全0）" << std::endl;
        }
    }

}

void ConfigManager::from_json(const json& j, ObjectPoint& p) {
    p.x = j["x"];
    p.y = j["y"];
    p.z = j["z"];
}

void ConfigManager::from_json(const json& j, ROIConfig& r) {
    r.roi = cv::Rect(j["roi_x"], j["roi_y"], j["roi_w"], j["roi_h"]);
    if (j.contains("minRadius")) r.minRadius = j["minRadius"];
    if (j.contains("maxRadius")) r.maxRadius = j["maxRadius"];
    if (j.contains("minCircularity")) r.minCircularity = j["minCircularity"];
    if (j.contains("minAspectRatio")) r.minAspectRatio = j["minAspectRatio"];
    if (j.contains("maxAspectRatio")) r.maxAspectRatio = j["maxAspectRatio"];
}

void ConfigManager::from_json(const json& j, YoloConfigFile& y) {
    if (j.contains("model_path")) y.model_path = j["model_path"];
    if (j.contains("confidence_threshold")) y.confidence_threshold = j["confidence_threshold"];
    if (j.contains("nms_threshold")) y.nms_threshold = j["nms_threshold"];
    if (j.contains("input_width")) y.input_width = j["input_width"];
    if (j.contains("input_height")) y.input_height = j["input_height"];
    if (j.contains("hole_class_id")) y.hole_class_id = j["hole_class_id"];
    if (j.contains("enable")) y.enable = j["enable"];
}

void ConfigManager::from_json(const json& j, PLCConfigFile& p) {
    if (j.contains("ip")) p.ip = j["ip"];
    if (j.contains("rack")) p.rack = j["rack"];
    if (j.contains("slot")) p.slot = j["slot"];
    if (j.contains("db_number")) p.db_number = j["db_number"];
    if (j.contains("db_start_offset")) p.db_start_offset = j["db_start_offset"];
    if (j.contains("timeout_ms")) p.timeout_ms = j["timeout_ms"];
    if (j.contains("auto_reconnect")) p.auto_reconnect = j["auto_reconnect"];
    if (j.contains("enable")) p.enable = j["enable"];
}

void ConfigManager::from_json(const json& j, AppConfig& cfg) {
    if (j.contains("description")) cfg.description = j["description"];

    // 相机参数
    if (j.contains("camera")) {
        from_json(j["camera"], cfg.camera);
    }

    // 3D 点库
    if (j.contains("object_points")) {
        cfg.object_points.clear();
        for (const auto& item : j["object_points"]) {
            ObjectPoint p;
            from_json(item, p);
            cfg.object_points.push_back(p);
        }
    }

    // ROI 列表
    if (j.contains("rois")) {
        cfg.rois.clear();
        for (const auto& item : j["rois"]) {
            ROIConfig r;
            from_json(item, r);
            cfg.rois.push_back(r);
        }
    }

    // YOLO 配置
    if (j.contains("yolo")) {
        from_json(j["yolo"], cfg.yolo);
    }

    // PLC 配置
    if (j.contains("plc")) {
        from_json(j["plc"], cfg.plc);
    }

    if (j.contains("standard_pose_file")) cfg.standard_pose_file = j["standard_pose_file"];
    if (j.contains("save_as_standard")) cfg.save_as_standard = j["save_as_standard"];
}


// 加载配置文件
bool ConfigManager::loadConfig(const std::string& filename, AppConfig& config) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ConfigManager] 无法打开配置文件: " << filename << std::endl;
            return false;
        }

        json j;
        file >> j;
        file.close();

        from_json(j, config);

        std::cout << "[ConfigManager] 配置已加载: " << filename << std::endl;
        std::cout << "[ConfigManager] 3D 点数: " << config.object_points.size() << std::endl;
        std::cout << "[ConfigManager] ROI 数: " << config.rois.size() << std::endl;
        std::cout << "[ConfigManager] YOLO: " << (config.yolo.enable ? "启用" : "禁用") << std::endl;
        std::cout << "[ConfigManager] PLC: " << (config.plc.enable ? "启用" : "禁用") << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[ConfigManager] 加载失败: " << e.what() << std::endl;
        return false;
    }
}

// 保存配置模板
bool ConfigManager::saveConfigTemplate(const std::string& filename) {
    try {
        json j;
        j["description"] = "config";
        j["standard_pose_file"] = "standard_pose.json";
        j["save_as_standard"] = false;

        // 相机内参
        j["camera"]["fx"] = 8730.097601293379;
        j["camera"]["fy"] = 8729.93196788731;
        j["camera"]["cx"] = 2032.5411458089795;
        j["camera"]["cy"] = 1590.2239152932418;
        j["camera"]["image_width"] = 4024;
        j["camera"]["image_height"] = 3046;

        // 3D点
        j["object_points"] = {
            {{"x", 2262.615}, {"y", -608.766}, {"z", 481.019}},
            {{"x", 2188.383}, {"y", -608.058}, {"z", 511.965}},
            {{"x", 2148.132}, {"y", -609.187}, {"z", 588.711}},
            {{"x", 2132.353}, {"y", -612.087}, {"z", 646.559}},
            {{"x", 2198.0}, {"y", -591.0}, {"z", 755.0}},
            {{"x", 2193.449}, {"y", -606.384}, {"z", 600.541}}
        };

        // ROI 配置
        j["rois"] = {
            {
                {"roi_x", 800}, {"roi_y", 1100}, {"roi_w", 300}, {"roi_h", 250},
                {"minRadius", 50}, {"maxRadius", 60},
                {"minCircularity", 0.1}, {"minAspectRatio", 0.5}, {"maxAspectRatio", 4.0}
            },
            {
                {"roi_x", 1000}, {"roi_y", 1840}, {"roi_w", 300}, {"roi_h", 250},
                {"minRadius", 30}, {"maxRadius", 50},
                {"minCircularity", 0.1}, {"minAspectRatio", 0.5}, {"maxAspectRatio", 4.0}
            },
            {
                {"roi_x", 1720}, {"roi_y", 2300}, {"roi_w", 300}, {"roi_h", 250},
                {"minRadius", 40}, {"maxRadius", 50},
                {"minCircularity", 0.1}, {"minAspectRatio", 0.5}, {"maxAspectRatio", 4.0}
            },
            {
                {"roi_x", 2300}, {"roi_y", 2550}, {"roi_w", 300}, {"roi_h", 250},
                {"minRadius", 30}, {"maxRadius", 40},
                {"minCircularity", 0.1}, {"minAspectRatio", 0.5}, {"maxAspectRatio", 4.0}
            },
            {
                {"roi_x", 3400}, {"roi_y", 1850}, {"roi_w", 500}, {"roi_h", 400},
                {"minRadius", 85}, {"maxRadius", 95},
                {"minCircularity", 0.1}, {"minAspectRatio", 0.5}, {"maxAspectRatio", 4.0}
            },
            {
                {"roi_x", 1860}, {"roi_y", 1890}, {"roi_w", 300}, {"roi_h", 250},
                {"minRadius", 45}, {"maxRadius", 55},
                {"minCircularity", 0.1}, {"minAspectRatio", 0.5}, {"maxAspectRatio", 4.0}
            }
        };

        // YOLO 配置
        j["yolo"]["model_path"] = "hole_detector.onnx";
        j["yolo"]["confidence_threshold"] = 0.5;
        j["yolo"]["nms_threshold"] = 0.4;
        j["yolo"]["input_width"] = 640;
        j["yolo"]["input_height"] = 640;
        j["yolo"]["hole_class_id"] = 0;
        j["yolo"]["enable"] = false;

        // PLC 配置
        j["plc"]["ip"] = "192.168.0.1";
        j["plc"]["rack"] = 0;
        j["plc"]["slot"] = 1;
        j["plc"]["db_number"] = 100;
        j["plc"]["db_start_offset"] = 0;
        j["plc"]["timeout_ms"] = 3000;
        j["plc"]["auto_reconnect"] = true;
        j["plc"]["enable"] = false;

        std::ofstream file(filename);
        file << j.dump(4);
        file.close();

        std::cout << "[ConfigManager] 配置模板已保存: " << filename << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[ConfigManager] 保存模板失败: " << e.what() << std::endl;
        return false;
    }
}

// 类型转换函数
std::vector<ROIParams> ConfigManager::convertToROIParams(const std::vector<ROIConfig>& rois) {
    std::vector<ROIParams> params;
    for (const auto& r : rois) {
        ROIParams p;
        p.roi = r.roi;
        p.minRadius = r.minRadius;
        p.maxRadius = r.maxRadius;
        p.minCircularity = r.minCircularity;
        p.minAspectRatio = r.minAspectRatio;
        p.maxAspectRatio = r.maxAspectRatio;
        params.push_back(p);
    }
    return params;
}

YoloConfig ConfigManager::convertToYoloConfig(const YoloConfigFile& cfg) {
    YoloConfig yc;
    yc.model_path = cfg.model_path;
    yc.confidence_threshold = cfg.confidence_threshold;
    yc.nms_threshold = cfg.nms_threshold;
    yc.input_width = cfg.input_width;
    yc.input_height = cfg.input_height;
    yc.hole_class_id = cfg.hole_class_id;
    return yc;
}