#ifndef IMAGE_FILTERS_H
#define IMAGE_FILTERS_H

#include "esp_camera.h"
#include "esp_err.h"

// 结构体定义：处理后的图像
typedef struct {
  uint8_t *data; // 图像数据
  int width;     // 图像宽度
  int height;    // 图像高度
  int channels;  // 图像通道数
} processed_image_t;

// 各种滤波器和图像处理函数
esp_err_t apply_gaussian_filter(camera_fb_t *fb, processed_image_t *output,
                                float sigma);
esp_err_t apply_log_filter(camera_fb_t *fb, processed_image_t *output,
                           float sigma);
esp_err_t apply_dog_filter(camera_fb_t *fb, processed_image_t *output,
                           float sigma1, float sigma2);
esp_err_t apply_gabor_filter(camera_fb_t *fb, processed_image_t *output,
                             float lambda, float theta, float sigma,
                             float gamma);
esp_err_t detect_edges_canny(camera_fb_t *fb, processed_image_t *output,
                             float low_threshold, float high_threshold);
esp_err_t polynomial_curve_fitting(const processed_image_t *edge_image,
                                   int degree, float *coefficients);

// 实用函数
esp_err_t convert_to_grayscale(camera_fb_t *fb, processed_image_t *output);
void free_processed_image(processed_image_t *img);

#endif // IMAGE_FILTERS_H
