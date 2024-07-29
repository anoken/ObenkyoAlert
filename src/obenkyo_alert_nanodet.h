#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include "net.h"  // NCNN library header

// Structure to represent a detected object
struct Object {
    cv::Rect_<float> rect;
    std::vector<cv::Point2f> pts;
    int left_or_right;
    int label;
    float prob;
};

// Calculate intersection area of two objects
static inline float intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// Safely crop a rectangle from an image
cv::Mat safeRectCrop(const cv::Mat& src, cv::Rect& rect) {
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, src.cols - rect.x);
    rect.height = std::min(rect.height, src.rows - rect.y);
    return (rect.width <= 0 || rect.height <= 0) ? cv::Mat() : src(rect).clone();
}

// Quick sort objects by probability (descending order)
void qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
    int i = left, j = right;
    float pivot = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > pivot) i++;
        while (objects[j].prob < pivot) j--;
        if (i <= j) std::swap(objects[i++], objects[j--]);
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

// Wrapper function for quick sort
void qsort_descent_inplace(std::vector<Object>& objects) {
    if (!objects.empty()) qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// Non-maximum suppression for object detection
static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    std::vector<float> areas(objects.size());
    for (size_t i = 0; i < objects.size(); i++) {
        areas[i] = objects[i].rect.width * objects[i].rect.height;
    }

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& a = objects[i];
        int keep = 1;
        for (int j : picked) {
            const Object& b = objects[j];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            if (inter_area / union_area > nms_threshold) {
                keep = 0;
                break;
            }
        }
        if (keep) picked.push_back(i);
    }
}

// Generate object proposals from network output
static void generate_proposals(const ncnn::Mat& cls_pred, const ncnn::Mat& dis_pred, int stride,
                               const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects) {
    const int num_grid = cls_pred.h;
    const int num_class = cls_pred.w;
    const int reg_max_1 = dis_pred.w / 4;

    int num_grid_x = in_pad.w / stride;
    int num_grid_y = num_grid / num_grid_x;

    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            const int idx = i * num_grid_x + j;
            const float* scores = cls_pred.row(idx);

            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++) {
                if (scores[k] > score) {
                    label = k;
                    score = scores[k];
                }
            }

            if (score >= prob_threshold) {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void*)dis_pred.row(idx));
                // Apply softmax to bbox_pred (code omitted for brevity)

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++) {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++) {
                        dis += l * dis_after_sm[l];
                    }
                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (j + 0.5f) * stride;
                float pb_cy = (i + 0.5f) * stride;

                Object obj;
                obj.rect.x = pb_cx - pred_ltrb[0];
                obj.rect.y = pb_cy - pred_ltrb[1];
                obj.rect.width = pred_ltrb[0] + pred_ltrb[2];
                obj.rect.height = pred_ltrb[1] + pred_ltrb[3];
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

// Convert NCNN Mat to OpenCV Mat
cv::Mat ncnnMatToCvMat(const ncnn::Mat& in) {
    if (in.empty()) return cv::Mat();

    cv::Mat out(in.h, in.w, CV_32FC3);
    float* pout = (float*)out.data;
    float scale = 255.0;

    for (int h = 0; h < in.h; h++) {
        for (int w = 0; w < in.w; w++) {
            for (int c = 0; c < 3; c++) {
                pout[(h * out.step1() + w * 3) + (2 - c)] = in.channel(c)[h * in.w + w] * scale;
            }
        }
    }

    cv::Mat out_8u;
    out.convertTo(out_8u, CV_8UC3);
    return out_8u;
}

// Convert NCNN feature to OpenCV Mat for visualization
void ncnn_conv_cvMat(const ncnn::Mat& feature, cv::Mat& image) {
    cv::Mat feature_img(feature.h, feature.w, CV_32FC1);
    memcpy((unsigned char*)feature_img.data, feature.data, feature.w * feature.h * sizeof(float));
    
    double min_val, max_val;
    cv::minMaxLoc(feature_img, &min_val, &max_val);
    
    cv::Mat adjMap;
    float scale = 255 / (max_val - min_val);
    feature_img.convertTo(adjMap, CV_8UC1, scale, -min_val * scale);
    
    cv::Mat resultMap;
    cv::applyColorMap(adjMap, resultMap, cv::COLORMAP_JET);
    cv::resize(resultMap, resultMap, cv::Size(feature.w * 10, feature.h * 10), 0, 0, cv::INTER_AREA);
    
    image = resultMap;
}


// Check if a string ends with a specific suffix
bool hasSuffix(const std::string& s, const std::string& suffix) {
    return (s.size() >= suffix.size()) && 
           equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}



// Get current timestamp as string
std::string getCurrentTimeStamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%m%d%H%M%S") << '_' << std::setfill('0') << std::setw(1) << ms.count() / 100;
    return oss.str();
}

// Calculate Root Mean Square of audio samples
double calculateRMS(const std::vector<int16_t>& samples) {
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0, 
        [](double acc, int16_t sample) { return acc + static_cast<double>(sample) * sample; });
    return std::sqrt(sum / samples.size());
}

// Generate random number
int random(int range) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, range);
    return dis(gen);
}