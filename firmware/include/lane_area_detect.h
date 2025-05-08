#ifndef __LANE_AREA_DETECT_H__
#define __LANE_AREA_DETECT_H__

#include "camera_config.h"
#include "esp_camera.h"
#include "esp_err.h"

// 结构体定义：处理后的图像
typedef struct
{
  uint8_t *data; // 图像数据
  int width;     // 图像宽度
  int height;    // 图像高度
  int channels;  // 图像通道数
} processed_image_t;

// 常量定义
#define MIN_SLOPE 0.4             // 最小斜率阈值
#define BUFFER_SIZE 256           // 通用缓冲区大小
#define ROI_RATIO 0.6             // ROI区域比例
#define PI 3.14159265358979323846 // PI常量

// 颜色阈值定义 (HSV空间)
typedef struct
{
  uint8_t h_min, h_max;
  uint8_t s_min, s_max;
  uint8_t v_min, v_max;
} color_threshold_t;

// 线段结构定义
typedef struct
{
  int16_t x1, y1;
  int16_t x2, y2;
  float slope;
} line_segment_t;

// 多项式拟合结果
typedef struct
{
  float a;    // 二次项系数
  float b;    // 一次项系数
  float c;    // 常数项
  bool valid; // 标记拟合是否有效
} poly_fit_t;

// 区域类型枚举
typedef enum
{
  AREA_LEVEL_0 = 0, // 0级区域 (两车道线内侧与模型预测的重叠区)
  AREA_LEVEL_1 = 1, // 1级区域 (车道线扩展的100px红色掩膜区)
  AREA_LEVEL_2 = 2  // 2级区域 (最外侧)
} area_level_t;

// 颜色阈值
extern const color_threshold_t WHITE_THRESHOLD;
extern const color_threshold_t YELLOW_THRESHOLD;

class LaneAreaDetector
{
public:
  // 构造和析构函数
  LaneAreaDetector();
  ~LaneAreaDetector();

  // 图像处理函数
  esp_err_t color_regmentation(camera_fb_t *fb, processed_image_t *output);
  esp_err_t color_2_gray(camera_fb_t *fb, processed_image_t *output);
  esp_err_t multi_scale_DoG(camera_fb_t *fb, processed_image_t *output);
  esp_err_t enhance_edges(camera_fb_t *fb, processed_image_t *output);
  esp_err_t ROI_mask(camera_fb_t *fb, processed_image_t *output);
  esp_err_t detect_lines(camera_fb_t *fb, processed_image_t *output);
  esp_err_t fit_lane_lines(camera_fb_t *fb, processed_image_t *output);
  esp_err_t lane_line_expand(camera_fb_t *fb, processed_image_t *output);
  esp_err_t fastSCNN_load(camera_fb_t *fb, processed_image_t *output);
  esp_err_t model_fit_area(camera_fb_t *fb, processed_image_t *output);

private:
  // 辅助绘图函数
  void bresenham_line(processed_image_t *img, int x1, int y1, int x2, int y2,
                      uint8_t r, uint8_t g, uint8_t b);

  // 检测到的线段数据
  line_segment_t *left_lines = nullptr;
  line_segment_t *right_lines = nullptr;
  int left_line_count = 0;
  int right_line_count = 0;

public:
  // 车道线拟合结果
  poly_fit_t left_lane_fit = {0, 0, 0, false};
  poly_fit_t right_lane_fit = {0, 0, 0, false};

private:

  // FastSCNN模型
  bool model_loaded = false;
  void *model_interpreter = nullptr;
  int model_input_width = 0;
  int model_input_height = 0;
};

#endif
