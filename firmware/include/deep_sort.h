#ifndef DEEP_SORT_H
#define DEEP_SORT_H

#include "esp_err.h"
#include <memory>
#include <vector>

// 检测框结构体
typedef struct {
  float x;          // 中心 x 坐标
  float y;          // 中心 y 坐标
  float width;      // 宽度
  float height;     // 高度
  float confidence; // 置信度
  int class_id;     // 类别ID
} detection_box_t;

// 跟踪对象结构体
typedef struct {
  int id;                // 跟踪ID
  float x;               // 中心 x 坐标
  float y;               // 中心 y 坐标
  float width;           // 宽度
  float height;          // 高度
  float vx;              // x方向速度
  float vy;              // y方向速度
  int class_id;          // 类别ID
  int age;               // 存在的帧数
  int time_since_update; // 自上次更新以来的帧数
} track_t;

// 检测结果结构体
typedef struct {
  std::vector<detection_box_t> boxes;
  int width;
  int height;
} detection_result_t;

// 跟踪结果结构体
typedef struct {
  std::vector<track_t> tracks;
  int frame_count;
} tracking_result_t;

// 初始化DeepSORT跟踪器
esp_err_t init_deep_sort();

// 更新跟踪器并获取跟踪结果
tracking_result_t update_tracker(const detection_result_t &detections);

// 计算两个边界框的IoU
float calculate_iou(const detection_box_t &box1, const detection_box_t &box2);

// 获取最新的跟踪结果
tracking_result_t get_latest_tracking();

#endif // DEEP_SORT_H
