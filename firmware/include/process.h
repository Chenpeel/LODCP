#ifndef __PROCESS_H__
#define __PROCESS_H__
#include "./lane_area_detect.h"
#include "esp_camera.h"
#include "esp_err.h"
#include <vector>

// 检测框结构体
typedef struct {
    float x;          // 归一化x坐标 (中心)
    float y;          // 归一化y坐标 (中心)
    float width;      // 归一化宽度
    float height;     // 归一化高度
    float confidence; // 置信度
    int class_id;     // 类别ID
} detection_box_t;

// 跟踪目标结构体
typedef struct {
    int id;           // 跟踪ID
    float x;          // 归一化x坐标 (中心)
    float y;          // 归一化y坐标 (中心) 
    float width;      // 归一化宽度
    float height;     // 归一化高度
    float velocity_x; // X方向速度
    float velocity_y; // Y方向速度
} tracking_object_t;

// 分割结果结构体
typedef struct {
    uint8_t* mask;    // 分割掩码 (每个像素的类别ID)
    int width;        // 掩码宽度
    int height;       // 掩码高度
    bool valid;       // 掩码是否有效
} segmentation_result_t;

// 检测结果结构体
typedef struct {
    std::vector<detection_box_t> boxes; // 检测框
    bool valid;                        // 检测结果是否有效
} detection_result_t;

// 跟踪结果结构体
typedef struct {
    std::vector<tracking_object_t> tracks; // 跟踪对象
    bool valid;                           // 跟踪结果是否有效
} tracking_result_t;

// 车道线结果结构体
typedef struct {
    poly_fit_t left_lane;  // 左车道线拟合
    poly_fit_t right_lane; // 右车道线拟合
    uint8_t* area_mask;    // 区域掩码
    int width;             // 掩码宽度
    int height;            // 掩码高度
    bool valid;            // 结果是否有效
} lane_result_t;

// 帧处理结果结构体
typedef struct {
    detection_result_t detection;       // 目标检测结果
    tracking_result_t tracking;         // 目标跟踪结果
    segmentation_result_t segmentation; // 语义分割结果
    lane_result_t lane;                 // 车道线结果
} frame_processing_result_t;

// 处理函数
esp_err_t process_frame(camera_fb_t* fb, frame_processing_result_t& result);

// 处理车道线和区域检测
esp_err_t process_lane_detection(camera_fb_t* fb, lane_result_t& result);

// 清理帧处理结果
void cleanup_frame_result(frame_processing_result_t& result);

#endif
