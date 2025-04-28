#ifndef SEMANTIC_SEG_H
#define SEMANTIC_SEG_H

#include "esp_camera.h"
#include "esp_err.h"
#include <mutex>
#include <vector>

// 分割结果结构体
typedef struct {
  uint8_t *mask;                       // 分割掩码
  int width;                           // 宽度
  int height;                          // 高度
  int num_classes;                     // 类别数量
  std::vector<float> class_confidence; // 每个类别的置信度
} segmentation_result_t;

// 初始化语义分割模块
esp_err_t init_semantic_segmentation();

// 运行语义分割
segmentation_result_t run_semantic_segmentation(camera_fb_t *frame);

// 预处理图像用于分割模型
uint8_t *preprocess_for_segmentation(camera_fb_t *frame, int target_width,
                                     int target_height);

// 获取最新的分割结果
segmentation_result_t get_latest_segmentation();

#endif // SEMANTIC_SEG_H
