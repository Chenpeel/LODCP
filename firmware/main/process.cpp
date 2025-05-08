#include "../include/process.h"
#include "../include/semantic_seg.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <memory>

static const char *TAG = "Process";

// 创建和初始化LaneAreaDetector对象
static LaneAreaDetector lane_detector;

// 处理帧 - 主函数
esp_err_t process_frame(camera_fb_t* fb, frame_processing_result_t& result) {
    if (!fb) {
        ESP_LOGE(TAG, "Invalid frame buffer");
        return ESP_FAIL;
    }

    // 记录处理开始时间
    int64_t start_time = esp_timer_get_time();

    // 处理车道线检测
    esp_err_t ret = process_lane_detection(fb, result.lane);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Lane detection failed: %s", esp_err_to_name(ret));
    }

    // 处理语义分割 (使用TFLite模型)
    semantic_segmentation_t seg_result = {0};
    ret = run_semantic_segmentation(fb, &seg_result);
    
    if (ret == ESP_OK && seg_result.mask != nullptr) {
        // 使用TFLite模型的分割结果
        result.segmentation.mask = seg_result.mask;
        result.segmentation.width = seg_result.width;
        result.segmentation.height = seg_result.height;
        result.segmentation.valid = true;
        
        ESP_LOGI(TAG, "语义分割完成, 类别数: %d", seg_result.num_classes);
        for (int i = 0; i < seg_result.num_classes; i++) {
            ESP_LOGI(TAG, "  类别 %d 置信度: %.3f", i, seg_result.class_scores[i]);
        }
        
        // 注意：这里不释放seg_result.mask，它被转移到了result.segmentation
        free(seg_result.class_scores);  // 但需要释放class_scores
    } else {
        // 回退到使用车道线检测结果
        if (result.lane.valid && result.lane.area_mask != nullptr) {
            result.segmentation.mask = result.lane.area_mask;
            result.segmentation.width = result.lane.width;
            result.segmentation.height = result.lane.height;
            result.segmentation.valid = true;
            ESP_LOGW(TAG, "使用车道线检测结果作为分割掩码（TFLite模型失败）");
        } else {
            result.segmentation.mask = nullptr;
            result.segmentation.valid = false;
            ESP_LOGW(TAG, "语义分割失败，无分割掩码");
        }
    }

    // 这个版本暂时不处理目标检测和跟踪
    result.detection.valid = false;
    result.tracking.valid = false;

    // 记录处理完成时间
    int64_t end_time = esp_timer_get_time();
    float process_time = (end_time - start_time) / 1000.0; // 转换为毫秒
    
    ESP_LOGI(TAG, "Frame processed in %.2f ms", process_time);
    
    return ESP_OK;
}

// 处理车道线和区域检测
esp_err_t process_lane_detection(camera_fb_t* fb, lane_result_t& result) {
    if (!fb) {
        ESP_LOGE(TAG, "Invalid frame buffer");
        return ESP_FAIL;
    }

    // 创建输出图像缓冲区用于模型拟合区域
    processed_image_t area_output = {0};
    
    // 调用车道线检测和区域分割
    esp_err_t ret = lane_detector.model_fit_area(fb, &area_output);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Lane area detection failed: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // 保存车道线拟合结果
    result.left_lane = lane_detector.left_lane_fit;
    result.right_lane = lane_detector.right_lane_fit;
    
    // 保存区域掩码
    result.area_mask = area_output.data;
    result.width = area_output.width;
    result.height = area_output.height;
    result.valid = (result.left_lane.valid || result.right_lane.valid) && (area_output.data != nullptr);
    
    // 注意：不要释放area_output.data，因为它被转移到了result.area_mask
    
    return ESP_OK;
}

// 清理帧处理结果
void cleanup_frame_result(frame_processing_result_t& result) {
    // 清理语义分割结果
    if (result.segmentation.valid && result.segmentation.mask) {
        // 无论掩码是否共享都释放，因为现在语义分割拥有自己的内存
        free(result.segmentation.mask);
        result.segmentation.mask = nullptr;
        result.segmentation.valid = false;
    }
    
    // 清理车道线结果
    if (result.lane.valid && result.lane.area_mask) {
        free(result.lane.area_mask);
        result.lane.area_mask = nullptr;
        result.lane.valid = false;
    }
    
    // 清理检测和跟踪结果
    result.detection.boxes.clear();
    result.detection.valid = false;
    
    result.tracking.tracks.clear();
    result.tracking.valid = false;
}