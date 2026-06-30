/*
YOLO在此的作用主要为粗定位（自动、智能、自适应的 ROI 生成器）
后续高精度定位特征点仍需使用传统算法（canny+椭圆拟合）
因此后续需要将其与TraditionalDetector结合
*/

#include "YoloDetector.h"
#include <opencv2/imgproc.hpp>

// 构造函数
YoloDetector::YoloDetector(const YoloConfig& config)
    : config_(config) {
    net_ = cv::dnn::readNetFromONNX(config_.model_path);     // 读取ONNX格式的神经网络模型
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);  // 使用OpenCV自带的推理引擎
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);       // 在CPU上运行（也可改为CUDA用GPU）
}

// 将图像转换为张量
cv::Mat YoloDetector::preprocess(const cv::Mat& image) {
    cv::Mat blob;      // 输出张量
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0,
        cv::Size(config_.input_width, config_.input_height),
        cv::Scalar(), true, false);
    return blob;
}

// detect()：兼容 FeatureDetectorBase 接口
bool YoloDetector::detect(const cv::Mat& image,
    std::vector<DetectedFeature>& features) {
    features.clear();                // 清空输出容器

    std::vector<cv::Rect> rois;      // ROI区域
    std::vector<float> confidences;  // 置信度

    if (!detectROIsWithConfidence(image, rois, &confidences)) {
        return false;
    }

    for (size_t i = 0; i < rois.size(); ++i) {
        DetectedFeature feat;
        feat.point = cv::Point2f(    // 计算ROI中心作为特征点（粗定位）
            rois[i].x + rois[i].width / 2.0f,
            rois[i].y + rois[i].height / 2.0f);
        feat.avgRadius = std::min(rois[i].width, rois[i].height) / 2.0f;
        feat.confidence = (i < confidences.size()) ? confidences[i] : 0.0f;
        features.push_back(feat);
    }

    return !features.empty();
}

// 返回 ROI 列表
bool YoloDetector::detectROIs(const cv::Mat& image, std::vector<cv::Rect>& rois) {
    return detectROIsWithConfidence(image, rois, nullptr);
}

// 返回 ROI、置信度
bool YoloDetector::detectROIsWithConfidence(const cv::Mat& image,
    std::vector<cv::Rect>& rois,
    std::vector<float>* confidences) {
    // 清空输出
    rois.clear();
    if (confidences) {
        confidences->clear();
    }

    // 预处理
    cv::Mat blob = preprocess(image);
    net_.setInput(blob);

    // 推理
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    // 后处理
    return postprocess(image, outputs, rois, confidences);
}

// 解析输出
bool YoloDetector::postprocess(const cv::Mat& image,
    const std::vector<cv::Mat>& outputs,
    std::vector<cv::Rect>& rois,
    std::vector<float>* confidences) {

    float scale_x = (float)image.cols / config_.input_width;   // 宽度缩放比
    float scale_y = (float)image.rows / config_.input_height;  // 高度缩放比

    std::vector<cv::Rect> raw_boxes;
    std::vector<float> raw_confidences;

    // 解析 YOLO 输出
    cv::Mat detections = outputs[0];
    for (int i = 0; i < detections.rows; ++i) {
        float confidence = detections.at<float>(i, 4);         // 置信度
        if (confidence < config_.confidence_threshold) continue;

        int class_id = (int)detections.at<float>(i, 5);        // 类别
        if (class_id != config_.hole_class_id) continue;       // 只保留孔洞

        float x = detections.at<float>(i, 0) * scale_x;
        float y = detections.at<float>(i, 1) * scale_y;
        float w = detections.at<float>(i, 2) * scale_x;
        float h = detections.at<float>(i, 3) * scale_y;

        raw_boxes.push_back(cv::Rect((int)(x - w / 2), (int)(y - h / 2), (int)w, (int)h));
        raw_confidences.push_back(confidence);
    }

    // NMS 过滤
    std::vector<int> indices;
    cv::dnn::NMSBoxes(raw_boxes, raw_confidences,
        config_.confidence_threshold,
        config_.nms_threshold, indices);

    // 输出结果
    for (int idx : indices) {
        rois.push_back(raw_boxes[idx]);
        if (confidences) {
            confidences->push_back(raw_confidences[idx]);
        }
    }
     

    // 防呆
    // 如果检测数量少于预期，用默认 ROI 补齐（防止后续崩溃）
    const int EXPECTED_COUNT = 6;
    while ((int)rois.size() < EXPECTED_COUNT) {
        rois.push_back(cv::Rect(0, 0, 100, 100));
        if (confidences) {
            confidences->push_back(0.0f);
        }
    }

    return !rois.empty();
}