#include "../include/semantic_seg.h"
#include "../include/model_loader.h"
#include "esp_log.h"
#include "img_converters.h"
#include <algorithm>
#include <cstdlib>

static const char *TAG = "SemanticSeg";

// 分割模型参数
#define SEG_CLASS_COUNT 2

#define ROAD_CLASS_ID 0
#define NON_ROAD_CLASS_ID 1

// 分割结果保护
static std::mutex seg_mutex;
static segmentation_result_t latest_segmentation = {nullptr, 0, 0, 0};

// 初始化语义分割
esp_err_t init_semantic_segmentation() {
  ESP_LOGI(TAG, "Initializing semantic segmentation");

  // 初始化分割模型
  esp_err_t ret = g_segmentation_model.init();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize segmentation model");
    return ret;
  }

  // 获取模型输出尺寸
  int seg_width = 0, seg_height = 0, seg_channels = 0;
  g_segmentation_model.getInputDims(seg_width, seg_height, seg_channels);

  ESP_LOGI(TAG,
           "Semantic segmentation initialized with input dimensions: %dx%dx%d",
           seg_width, seg_height, seg_channels);

  return ESP_OK;
}

// 预处理图像用于分割模型
uint8_t *preprocess_for_segmentation(camera_fb_t *frame, int target_width,
                                     int target_height) {
  if (!frame) {
    ESP_LOGE(TAG, "Invalid frame");
    return nullptr;
  }

  // 分配内存用于预处理图像
  uint8_t *preprocessed = (uint8_t *)malloc(target_width * target_height * 3);
  if (!preprocessed) {
    ESP_LOGE(TAG, "Failed to allocate memory for preprocessed image");
    return nullptr;
  }

  // 分配RGB缓冲区
  uint8_t *rgb_buffer = nullptr;

  if (frame->format == PIXFORMAT_JPEG) {
    rgb_buffer = (uint8_t *)malloc(frame->width * frame->height * 3);
    if (!rgb_buffer) {
      ESP_LOGE(TAG, "Failed to allocate RGB buffer");
      free(preprocessed);
      return nullptr;
    }

    // 将JPEG转换为RGB888
    bool converted =
        fmt2rgb888(frame->buf, frame->len, PIXFORMAT_JPEG, rgb_buffer);
    if (!converted) {
      ESP_LOGE(TAG, "Failed to convert JPEG to RGB888");
      free(rgb_buffer);
      free(preprocessed);
      return nullptr;
    }
  } else if (frame->format == PIXFORMAT_RGB888) {
    rgb_buffer = frame->buf;
  } else {
    ESP_LOGE(TAG, "Unsupported image format");
    free(preprocessed);
    return nullptr;
  }

  // 将图像缩放到模型输入尺寸
  // 这里使用简单的双线性插值方法
  float scale_x = (float)frame->width / target_width;
  float scale_y = (float)frame->height / target_height;

  for (int y = 0; y < target_height; y++) {
    for (int x = 0; x < target_width; x++) {
      // 计算源坐标
      float src_x = x * scale_x;
      float src_y = y * scale_y;

      // 双线性插值的四个点
      int x1 = (int)src_x;
      int y1 = (int)src_y;
      int x2 = std::min(x1 + 1, frame->width - 1);
      int y2 = std::min(y1 + 1, frame->height - 1);

      // 计算权重
      float wx = src_x - x1;
      float wy = src_y - y1;

      // 对每个颜色通道进行插值
      for (int c = 0; c < 3; c++) {
        float top = rgb_buffer[(y1 * frame->width + x1) * 3 + c] * (1 - wx) +
                    rgb_buffer[(y1 * frame->width + x2) * 3 + c] * wx;
        float bottom = rgb_buffer[(y2 * frame->width + x1) * 3 + c] * (1 - wx) +
                       rgb_buffer[(y2 * frame->width + x2) * 3 + c] * wx;
        float pixel = top * (1 - wy) + bottom * wy;

        // 量化为uint8
        preprocessed[(y * target_width + x) * 3 + c] = (uint8_t)pixel;
      }
    }
  }

  // 如果我们分配了RGB缓冲区，释放它
  if (frame->format == PIXFORMAT_JPEG) {
    free(rgb_buffer);
  }

  return preprocessed;
}

// 运行语义分割
segmentation_result_t run_semantic_segmentation(camera_fb_t *frame) {
  ESP_LOGI(TAG, "Running semantic segmentation");

  segmentation_result_t result = {nullptr, 0, 0, 0};

  // 获取分割模型输入尺寸
  int seg_width = 0, seg_height = 0, seg_channels = 0;
  g_segmentation_model.getInputDims(seg_width, seg_height, seg_channels);

  if (seg_width == 0 || seg_height == 0 || seg_channels == 0) {
    ESP_LOGE(TAG, "Invalid segmentation model dimensions");
    return result;
  }

  // 预处理图像
  uint8_t *preprocessed =
      preprocess_for_segmentation(frame, seg_width, seg_height);
  if (!preprocessed) {
    ESP_LOGE(TAG, "Failed to preprocess image for segmentation");
    free(preprocessed);
    return result;
  }

  // 运行推理
  esp_err_t ret = g_segmentation_model.runInference(
      preprocessed, seg_width * seg_height * seg_channels * sizeof(uint8_t));

  free(preprocessed);

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Segmentation inference failed");
    return result;
  }

  // 获取输出结果
  uint8_t *output_data = g_segmentation_model.getQuantizedOutputData();
  int output_height = g_segmentation_model.getOutputHeight();
  int output_width = g_segmentation_model.getOutputWidth();
  int output_channels = g_segmentation_model.getOutputChannels();

  if (!output_data || output_height == 0 || output_width == 0) {
    ESP_LOGE(TAG, "Invalid segmentation output");
    return result;
  }

  // 为分割掩码分配内存
  uint8_t *mask = (uint8_t *)malloc(output_width * output_height);
  if (!mask) {
    ESP_LOGE(TAG, "Failed to allocate memory for segmentation mask");
    return result;
  }

  // 处理分割结果 - FastSCNN输出每个像素的类别概率
  // 这里我们需要找到每个像素的最可能类别
  std::vector<float> class_confidence(SEG_CLASS_COUNT, 0.0f);

  for (int y = 0; y < output_height; y++) {
    for (int x = 0; x < output_width; x++) {
      uint8_t max_class = 0;
      uint8_t max_val = 0;

      // 找到概率最高的类别
      for (int c = 0; c < output_channels; c++) {
        uint8_t val = output_data[(y * output_width + x) * output_channels + c];
        if (val > max_val) {
          max_val = val;
          max_class = c;
        }

        // 累加类别置信度
        class_confidence[c] += val;
      }

      // 存储像素的类别
      mask[y * output_width + x] = max_class;
    }
  }

  // 计算每个类别的平均置信度
  for (int c = 0; c < SEG_CLASS_COUNT; c++) {
    class_confidence[c] /= (output_width * output_height);
  }

  // 填充结果结构体
  result.mask = mask;
  result.width = output_width;
  result.height = output_height;
  result.num_classes = SEG_CLASS_COUNT;
  result.class_confidence = class_confidence;

  // 更新最新分割结果
  {
    std::lock_guard<std::mutex> lock(seg_mutex);

    // 释放之前的结果
    if (latest_segmentation.mask) {
      free(latest_segmentation.mask);
    }

    latest_segmentation = result;
  }

  ESP_LOGI(TAG, "Semantic segmentation completed: %dx%d with %d classes",
           result.width, result.height, result.num_classes);

  return result;
}

// 获取最新的分割结果
segmentation_result_t get_latest_segmentation() {
  std::lock_guard<std::mutex> lock(seg_mutex);
  return latest_segmentation;
}
