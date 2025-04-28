#ifndef PROCESS_H
#define PROCESS_H

#include "../include/collision.h"
#include "../include/deep_sort.h"
#include "../include/semantic_seg.h"
#include "esp_camera.h"
#include "esp_err.h"

// 处理结果结构体
typedef struct {
  camera_fb_t *frame;                 // 原始帧
  segmentation_result_t segmentation; // 分割结果
  detection_result_t detection;       // 检测结果
  tracking_result_t tracking;         // 跟踪结果
  collision_prediction_t collision;   // 碰撞预测结果
} frame_processing_result_t;
extern float latest_coefficients[4];
// 初始化处理流水线
esp_err_t init_processing_pipeline();

// 处理单帧图像
frame_processing_result_t process_frame(camera_fb_t *frame);

// 传统图像处理
esp_err_t traditional_processing(camera_fb_t *frame);

// 目标检测 - 返回检测结果
detection_result_t detect_objects(camera_fb_t *frame);

#endif // PROCESS_H
