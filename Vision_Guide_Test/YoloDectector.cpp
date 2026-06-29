#include "YoloDetector.h"
#include <opencv2/imgproc.hpp>
#include "FeatureDetectorBase.h"

YoloDetector::YoloDetector(const YoloConfig& config) 
    : config_(config) {
    // 加载ONNX模型
    net_ = cv::dnn::readNetFromONNX(config_.model_path);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

cv::Mat YoloDetector::preprocess(const cv::Mat& image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, 
                           cv::Size(config_.input_width, config_.input_height),
                           cv::Scalar(), true, false);
    return blob;
}

bool YoloDetector::detect(const cv::Mat& image, 
                          std::vector<DetectedFeature>& features) {
    features.clear();
    
    // 1. 预处理 + 推理
    cv::Mat blob = preprocess(image);
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    
    // 2. 后处理
    return postprocess(image, outputs, features);
}

bool YoloDetector::postprocess(const cv::Mat& image,
                               const std::vector<cv::Mat>& outputs,
                               std::vector<DetectedFeature>& features) {
    float scale_x = (float)image.cols / config_.input_width;
    float scale_y = (float)image.rows / config_.input_height;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    // 解析YOLO输出（具体格式取决于你的模型）
    cv::Mat detections = outputs[0];
    for (int i = 0; i < detections.rows; ++i) {
        float confidence = detections.at<float>(i, 4);
        if (confidence < config_.confidence_threshold) continue;
        
        int class_id = (int)detections.at<float>(i, 5);
        if (class_id != config_.hole_class_id) continue;
        
        float x = detections.at<float>(i, 0) * scale_x;
        float y = detections.at<float>(i, 1) * scale_y;
        float w = detections.at<float>(i, 2) * scale_x;
        float h = detections.at<float>(i, 3) * scale_y;
        
        boxes.push_back(cv::Rect(x - w/2, y - h/2, w, h));
        confidences.push_back(confidence);
        class_ids.push_back(class_id);
    }
    
    // NMS过滤
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 
                      config_.confidence_threshold, 
                      config_.nms_threshold, indices);
    
    // 提取最终结果
    for (int idx : indices) {
        cv::Rect bbox = boxes[idx];
        cv::Point2f refined_center;
        if (!refineCenter(image, bbox, refined_center)) {
            refined_center = cv::Point2f(bbox.x + bbox.width/2.0f,
                                        bbox.y + bbox.height/2.0f);
        }
        
        DetectedFeature feat;
        feat.point = refined_center;
        feat.avgRadius = std::min(bbox.width, bbox.height) / 2.0f;
        feat.confidence = confidences[idx];
        features.push_back(feat);
    }
    
    return !features.empty();
}

bool YoloDetector::refineCenter(const cv::Mat& image, 
                                const cv::Rect& bbox,
                                cv::Point2f& refined_center) {
    // 在bbox区域内用传统方法精修中心（复用原有逻辑）
    cv::Mat roi = image(bbox);
    cv::Mat gray, edges;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 30, 100);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        if (contour.size() < 10) continue;
        cv::RotatedRect ellipse = cv::fitEllipse(contour);
        double avg_r = (ellipse.size.width + ellipse.size.height) / 4.0;
        if (avg_r > bbox.width * 0.3 && avg_r < bbox.width * 0.6) {
            refined_center = ellipse.center + cv::Point2f(bbox.x, bbox.y);
            return true;
        }
    }
    return false;
}