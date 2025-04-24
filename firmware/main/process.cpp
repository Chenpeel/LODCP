#include "../include/process.h"
#include "dl_lib_matrix3d.h"
#include "esp_dsp.h"
#include "esp_log.h"

static const char *TAG = "Process";
// 模型加载
esp_err_t init_processing_pipeline() {}

// 传统图像处理过程
void *traditional_processing(camera_fb_t *frame) {}

// 语义分割
void *semantic_segmentation(camera_fb_t *frame) {}

// 目标检测
void *object_detection(camera_fb_t *frame) {}

// 跟踪匹配
void *tracking_matching(void *detection_results, void *tracking_data) {}

// 碰撞预测
void *collision_prediction(void *tracking_data, void *segmentation_mask) {}

frame_processing_result_t process_frame(camera_fb_t *frame) {
  frame_processing_result_t result;
  result.frame = frame;

  // 传统处理
  void *traditional_result = traditional_processing(frame);

  // 语义分割
  result.segmentation_mask = semantic_segmentation(frame);

  // 目标检测
  result.detection_results = object_detection(frame);

  // 跟踪匹配
  result.tracking_data = tracking_matching(result.detection_results, nullptr);

  // 碰撞预测
  result.collision_risk =
      collision_prediction(result.tracking_data, result.segmentation_mask);

  return result;
}
