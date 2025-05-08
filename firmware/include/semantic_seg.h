#ifndef __SEMANTIC_SEG_H__
#define __SEMANTIC_SEG_H__

#include "esp_camera.h"
#include "esp_err.h"

// 语义分割模型类型
typedef enum {
    SEG_MODEL_BINARY,     // 二分类模型（背景/前景）
    SEG_MODEL_MULTICLASS, // 多类别模型（道路/车辆/人等）
    SEG_MODEL_CUSTOM      // 自定义模型
} seg_model_type_t;

// 语义分割结果
typedef struct {
    uint8_t* mask;       // 分割掩码（每个像素的类别ID）
    int width;           // 掩码宽度
    int height;          // 掩码高度
    int num_classes;     // 类别数量
    float* class_scores; // 每个类别的置信度
} semantic_segmentation_t;

// 语义分割类别信息
typedef struct {
    int id;              // 类别ID
    const char* name;    // 类别名称
    uint8_t color[3];    // 类别颜色 (RGB)
} seg_class_info_t;

// 初始化语义分割模型
esp_err_t init_semantic_segmentation();

// 加载指定的语义分割模型文件
esp_err_t load_segmentation_model(const char* model_path, seg_model_type_t model_type);

// 获取模型信息
esp_err_t get_segmentation_model_info(int* input_width, int* input_height, 
                                     int* num_classes);

// 执行语义分割推理
esp_err_t run_semantic_segmentation(camera_fb_t* fb, semantic_segmentation_t* result);

// 释放语义分割结果
void free_segmentation_result(semantic_segmentation_t* result);

// 获取类别信息
const seg_class_info_t* get_class_info(int class_id);

// 设置推理参数
esp_err_t set_segmentation_params(float conf_threshold, bool enable_preprocessing);

#endif // __SEMANTIC_SEG_H__