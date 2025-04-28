#include "../include/process.h"
#include "../include/model_loader.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "img_converters.h"
#include <functional>
#include <pthread.h>
#include <vector>

static const char *TAG = "Process";
float latest_coefficients[4] = {0.0f, 0.0f, 0.0f, 0.0f};
// 目标检测模型参数
#define DET_INPUT_WIDTH 640
#define DET_INPUT_HEIGHT 640
#define DET_INPUT_CHANNELS 3
#define DET_CLASS_COUNT 2

// 线程参数
typedef struct {
  camera_fb_t *frame;
  void *result;
  bool done;
  std::function<void *()> task;
} thread_args_t;

static thread_args_t seg_thread_args = {nullptr, nullptr, false, nullptr};
static thread_args_t det_thread_args = {nullptr, nullptr, false, nullptr};
static pthread_t seg_thread_id, det_thread_id;
static pthread_mutex_t frame_mutex = PTHREAD_MUTEX_INITIALIZER;

// 分割任务线程函数
static void *segmentation_thread_func(void *arg) {
  thread_args_t *args = (thread_args_t *)arg;

  while (true) {
    // 等待新的任务
    pthread_mutex_lock(&frame_mutex);
    if (!args->frame) {
      pthread_mutex_unlock(&frame_mutex);
      usleep(10000); // 10ms
      continue;
    }

    // 获取任务并解锁
    camera_fb_t *frame = args->frame;
    std::function<void *()> task = args->task;
    args->done = false;
    pthread_mutex_unlock(&frame_mutex);

    // 执行任务
    void *result = task();

    // 完成任务
    pthread_mutex_lock(&frame_mutex);
    args->result = result;
    args->done = true;
    args->frame = nullptr; // 清除任务
    pthread_mutex_unlock(&frame_mutex);
  }

  return nullptr;
}

// 检测任务线程函数
static void *detection_thread_func(void *arg) {
  thread_args_t *args = (thread_args_t *)arg;

  while (true) {
    // 等待新的任务
    pthread_mutex_lock(&frame_mutex);
    if (!args->frame) {
      pthread_mutex_unlock(&frame_mutex);
      usleep(10000); // 10ms
      continue;
    }

    // 获取任务并解锁
    camera_fb_t *frame = args->frame;
    std::function<void *()> task = args->task;
    args->done = false;
    pthread_mutex_unlock(&frame_mutex);

    // 执行任务
    void *result = task();

    // 完成任务
    pthread_mutex_lock(&frame_mutex);
    args->result = result;
    args->done = true;
    args->frame = nullptr; // 清除任务
    pthread_mutex_unlock(&frame_mutex);
  }

  return nullptr;
}

// 初始化处理线程
static esp_err_t init_processing_threads() {
  ESP_LOGI(TAG, "Initializing processing threads");

  pthread_attr_t attr;
  pthread_attr_init(&attr);

  // 创建分割线程
  if (pthread_create(&seg_thread_id, &attr, segmentation_thread_func,
                     &seg_thread_args) != 0) {
    ESP_LOGE(TAG, "Failed to create segmentation thread");
    return ESP_FAIL;
  }

  // 创建检测线程
  if (pthread_create(&det_thread_id, &attr, detection_thread_func,
                     &det_thread_args) != 0) {
    ESP_LOGE(TAG, "Failed to create detection thread");
    return ESP_FAIL;
  }

  ESP_LOGI(TAG, "Processing threads initialized successfully");
  return ESP_OK;
}

// 预处理图像用于目标检测
uint8_t *preprocess_for_detection(camera_fb_t *frame, int target_width,
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
  float scale_x = (float)frame->width / target_width;
  float scale_y = (float)frame->height / target_height;

  for (int y = 0; y < target_height; y++) {
    for (int x = 0; x < target_width; x++) {
      // 计算源坐标
      int src_x = (int)(x * scale_x);
      int src_y = (int)(y * scale_y);

      if (src_x >= frame->width)
        src_x = frame->width - 1;
      if (src_y >= frame->height)
        src_y = frame->height - 1;

      int src_idx = (src_y * frame->width + src_x) * 3;
      int dst_idx = (y * target_width + x) * 3;

      // 复制RGB通道
      preprocessed[dst_idx] = rgb_buffer[src_idx];         // R
      preprocessed[dst_idx + 1] = rgb_buffer[src_idx + 1]; // G
      preprocessed[dst_idx + 2] = rgb_buffer[src_idx + 2]; // B
    }
  }

  // 如果我们分配了RGB缓冲区，释放它
  if (frame->format == PIXFORMAT_JPEG) {
    free(rgb_buffer);
  }

  return preprocessed;
}

// 目标检测处理
detection_result_t detect_objects(camera_fb_t *frame) {
  ESP_LOGI(TAG, "Detecting objects in frame");
  detection_result_t result = {std::vector<detection_box_t>(), 0, 0};

  if (!frame) {
    ESP_LOGE(TAG, "Null frame");
    return result;
  }

  result.width = frame->width;
  result.height = frame->height;

  // 获取检测模型输入尺寸
  int det_width = DET_INPUT_WIDTH;
  int det_height = DET_INPUT_HEIGHT;
  int det_channels = DET_INPUT_CHANNELS;

  // 预处理图像
  uint8_t *preprocessed =
      preprocess_for_detection(frame, det_width, det_height);
  if (!preprocessed) {
    ESP_LOGE(TAG, "Failed to preprocess image");
    return result;
  }

  // 运行推理
  esp_err_t ret = g_detection_model.runInference(
      preprocessed, det_width * det_height * det_channels);

  free(preprocessed);

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Detection inference failed");
    return result;
  }

  // 获取模型输出
  float *output_data = g_detection_model.getOutputData();
  if (!output_data) {
    ESP_LOGE(TAG, "Failed to get detection output data");
    return result;
  }

  // 根据模型输出格式解析检测结果
  // 假设输出格式为 [num_detections, 7] 其中每个检测包含:
  // [batch_id, class_id, confidence, x_min, y_min, x_max, y_max]
  size_t output_size = g_detection_model.getOutputSize();

  // 处理检测结果
  const int values_per_detection = 6;
  const int max_detections = output_size / values_per_detection;
  const float confidence_threshold = 0.5f; // 调整检测阈值

  for (int i = 0; i < max_detections; i++) {
    float *detection = output_data + i * values_per_detection;
    float confidence = detection[2];

    // 过滤低置信度检测
    if (confidence < confidence_threshold) {
      continue;
    }

    int class_id = (int)detection[1];
    float x_min = detection[3] * frame->width;
    float y_min = detection[4] * frame->height;
    float x_max = detection[5] * frame->width;
    float y_max = detection[6] * frame->height;

    // 计算中心点坐标和宽高
    float center_x = (x_min + x_max) / 2.0f;
    float center_y = (y_min + y_max) / 2.0f;
    float width = x_max - x_min;
    float height = y_max - y_min;

    // 创建检测框
    detection_box_t box;
    box.x = center_x / frame->width; // 归一化坐标
    box.y = center_y / frame->height;
    box.width = width / frame->width;
    box.height = height / frame->height;
    box.confidence = confidence;
    box.class_id = class_id;

    result.boxes.push_back(box);
  }

  ESP_LOGI(TAG, "Detected %d objects", result.boxes.size());
  return result;
}

// 传统图像处理
esp_err_t traditional_processing(camera_fb_t *frame) {
  ESP_LOGI(TAG, "Applying traditional image processing");

  if (!frame) {
    return ESP_FAIL;
  }

  // 如果是JPEG格式，我们需要先解码
  bool is_jpeg = (frame->format == PIXFORMAT_JPEG);
  uint8_t *rgb_buffer = NULL;
  size_t rgb_len = 0;
  if (is_jpeg) {
    // 分配RGB缓冲区
    rgb_buffer = (uint8_t *)malloc(frame->width * frame->height * 3);
    if (!rgb_buffer) {
      ESP_LOGE(TAG, "Failed to allocate memory for RGB buffer");
      return ESP_FAIL;
    }

    // 将JPEG转换为RGB888
    bool converted =
        fmt2rgb888(frame->buf, frame->len, PIXFORMAT_JPEG, rgb_buffer);
    if (!converted) {
      ESP_LOGE(TAG, "Failed to convert JPEG to RGB888");
      free(rgb_buffer);
      return ESP_FAIL;
    }

    rgb_len = frame->width * frame->height * 3;
  } else if (frame->format == PIXFORMAT_RGB888) {
    // 直接使用RGB数据
    rgb_buffer = frame->buf;
    rgb_len = frame->width * frame->height * 3;
  } else {
    ESP_LOGE(TAG, "Unsupported image format for processing");
    return ESP_FAIL;
  }

  // 计算亮度均值和标准差
  float mean = 0.0f;
  float sum_squared = 0.0f;

  for (size_t i = 0; i < rgb_len; i += 3) {
    float luminance = 0.299f * rgb_buffer[i] + 0.587f * rgb_buffer[i + 1] +
                      0.114f * rgb_buffer[i + 2];
    mean += luminance;
    sum_squared += luminance * luminance;
  }

  int pixel_count = rgb_len / 3;
  mean /= pixel_count;
  float variance = (sum_squared / pixel_count) - (mean * mean);
  float std_dev = sqrtf(variance);

  ESP_LOGI(TAG, "Image statistics: mean=%.2f, std_dev=%.2f", mean, std_dev);

  // 根据图像统计选择合适的处理方法
  processed_image_t output = {NULL, 0, 0, 0};
  esp_err_t ret = ESP_OK;

  // 低照度情况 - 使用高斯滤波减少噪声
  if (mean < 80.0f) {
    ESP_LOGI(TAG, "Applying Gaussian filter for low light conditions");
    ret = apply_gaussian_filter(frame, &output, 1.5f);
  }
  // 高对比度情况 - 使用DoG滤波增强边缘
  else if (std_dev > 60.0f) {
    ESP_LOGI(TAG, "Applying DoG filter for high contrast conditions");
    ret = apply_dog_filter(frame, &output, 1.0f, 2.0f);
  }
  // 检测道路边缘 - 使用Gabor滤波器
  else {
    ESP_LOGI(TAG, "Applying Gabor filter for edge detection");
    // 使用不同方向的Gabor滤波器以检测不同方向的边缘
    ret = apply_gabor_filter(frame, &output, 8.0f, 0.0f, 2.0f, 0.5f);
  }

  // 如果处理失败，记录错误但继续执行
  if (ret != ESP_OK) {
    ESP_LOGW(TAG, "Image filter application failed: %d", ret);
  }
  // 如果处理成功，进行边缘检测并尝试多项式拟合
  else if (output.data) {
    // 检测边缘
    processed_image_t edges = {NULL, 0, 0, 0};
    ret = detect_edges_canny(frame, &edges, 0.1f, 0.3f);

    if (ret == ESP_OK && edges.data) {
      // 拟合道路边缘的多项式曲线 (3次多项式)
      float coefficients[4] = {0.0f};
      ret = polynomial_curve_fitting(&edges, 3, coefficients);

      if (ret == ESP_OK) {
        ESP_LOGI(
            TAG,
            "Road curve fitted as: y = %.3f*x^3 + %.3f*x^2 + %.3f*x + %.3f",
            coefficients[3], coefficients[2], coefficients[1], coefficients[0]);

        // 存储系数供后续使用
        memcpy(latest_coefficients, coefficients, sizeof(latest_coefficients));
      }

      // 释放边缘图像
      free_processed_image(&edges);
    }

    // 将处理后的图像复制回原始帧（如果不是JPEG格式）
    if (frame->format == PIXFORMAT_RGB888 && output.channels == 3) {
      memcpy(frame->buf, output.data, frame->width * frame->height * 3);
    }

    // 释放处理后的图像
    free_processed_image(&output);
  }

  // RGB888格式的简单处理 (如果前面没有应用其他滤波器)
  if (frame->format == PIXFORMAT_RGB888 && !output.data) {
    uint8_t *buffer = frame->buf;
    size_t len = frame->width * frame->height * 3;

    // 简单增强对比度
    for (size_t i = 0; i < len; i++) {
      // 将像素值映射到0-255范围
      buffer[i] = (uint8_t)(buffer[i] * 1.2);
    }

    ESP_LOGI(TAG, "Applied traditional processing to RGB frame");
  }

  // 释放RGB缓冲区（如果是我们分配的）
  if (is_jpeg && rgb_buffer) {
    free(rgb_buffer);
  }

  return ret;
}

void use_polynomial_fitting_results(const float *coefficients,
                                    frame_processing_result_t &result) {
  if (!coefficients) {
    return;
  }

  float image_center_x = 0.5f; // 归一化坐标

  // 使用多项式计算在中心x位置的曲线y值
  float curve_y =
      coefficients[0] + coefficients[1] * image_center_x +
      coefficients[2] * image_center_x * image_center_x +
      coefficients[3] * image_center_x * image_center_x * image_center_x;

  // 计算图像中心点到曲线的垂直距离
  float distance_to_curve = fabsf(0.5f - curve_y); // 归一化坐标

  // 估计曲率（二阶导数）
  float curvature =
      2.0f * coefficients[2] + 6.0f * coefficients[3] * image_center_x;

  // 如果我们有碰撞结果，根据车道线信息更新碰撞风险
  if (result.collision.risks.size() > 0) {
    // 如果偏离车道线太远，增加碰撞风险
    if (distance_to_curve > 0.15f) { // 阈值可以调整
      for (auto &risk : result.collision.risks) {
        risk.risk_score = std::min(1.0f, risk.risk_score + 0.2f);
      }

      // 更新总体风险
      result.collision.overall_risk =
          std::min(1.0f, result.collision.overall_risk + 0.2f);
      ESP_LOGW(TAG, "Lane departure detected! Adjusted collision risk to %.2f",
               result.collision.overall_risk);
    }
  }
}

// 初始化处理流水线
esp_err_t init_processing_pipeline() {
  ESP_LOGI(TAG, "Initializing processing pipeline");

  // 初始化检测模型
  if (g_detection_model.init() != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize detection model");
    return ESP_FAIL;
  }

  // 初始化分割模型
  if (init_semantic_segmentation() != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize semantic segmentation");
    return ESP_FAIL;
  }

  // 初始化DeepSORT跟踪器
  if (init_deep_sort() != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize DeepSORT tracker");
    return ESP_FAIL;
  }

  // 初始化碰撞预测
  init_collision_prediction();

  // 初始化处理线程
  if (init_processing_threads() != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize processing threads");
    return ESP_FAIL;
  }

  ESP_LOGI(TAG, "Processing pipeline initialized successfully");
  return ESP_OK;
}

// 处理单帧图像 - 主要处理函数
frame_processing_result_t process_frame(camera_fb_t *frame) {
  int64_t start_time = esp_timer_get_time();

  ESP_LOGI(TAG, "Processing frame %dx%d, format %d", frame->width,
           frame->height, frame->format);

  frame_processing_result_t result;
  result.frame = frame;

  // 应用传统图像处理
  esp_err_t traditional_result = traditional_processing(frame);

  // 使用多项式拟合结果（只有当处理成功时）
  if (traditional_result == ESP_OK) {
    use_polynomial_fitting_results(latest_coefficients, result);
  }

  // 启动两个并行任务
  // 1. 语义分割任务
  pthread_mutex_lock(&frame_mutex);
  bool seg_task_set = false;
  if (seg_thread_args.done || seg_thread_args.frame == nullptr) {
    // 只有当前一个任务完成或没有任务时才设置新任务
    seg_thread_args.frame = frame;
    seg_thread_args.task = [frame]() -> void * {
      segmentation_result_t *seg_result = new segmentation_result_t();
      *seg_result = run_semantic_segmentation(frame);
      return seg_result;
    };
    seg_thread_args.done = false;
    seg_task_set = true;
  }
  pthread_mutex_unlock(&frame_mutex);

  // 如果无法设置任务，等待当前任务完成
  if (!seg_task_set) {
    for (int i = 0; i < 10; i++) {
      usleep(10000); // 10ms
      pthread_mutex_lock(&frame_mutex);
      if (seg_thread_args.done) {
        seg_task_set = true;
        pthread_mutex_unlock(&frame_mutex);
        break;
      }
      pthread_mutex_unlock(&frame_mutex);
    }
  }

  // 2. 目标检测任务
  pthread_mutex_lock(&frame_mutex);
  bool det_task_set = false;
  if (det_thread_args.done || det_thread_args.frame == nullptr) {
    det_thread_args.frame = frame;
    det_thread_args.task = [frame]() -> void * {
      detection_result_t *det_result = new detection_result_t();
      *det_result = detect_objects(frame);
      return det_result;
    };
    det_thread_args.done = false;
    det_task_set = true;
  }
  pthread_mutex_unlock(&frame_mutex);

  if (!det_task_set) {
    for (int i = 0; i < 10; i++) {
      usleep(10000);
      pthread_mutex_lock(&frame_mutex);
      if (det_thread_args.done) {
        det_task_set = true;
        pthread_mutex_unlock(&frame_mutex);
        break;
      }
      pthread_mutex_unlock(&frame_mutex);
    }
  }

  // 等待分割和检测任务完成
  bool seg_done = false;
  bool det_done = false;

  // 最多等待100ms
  for (int i = 0; i < 10 && (!seg_done || !det_done); i++) {
    pthread_mutex_lock(&frame_mutex);
    seg_done = seg_thread_args.done;
    det_done = det_thread_args.done;
    pthread_mutex_unlock(&frame_mutex);

    if (!seg_done || !det_done) {
      usleep(10000); // 10ms
    }
  }

  // 收集结果
  pthread_mutex_lock(&frame_mutex);
  if (seg_done && seg_thread_args.result) {
    result.segmentation = *(segmentation_result_t *)seg_thread_args.result;
    delete (segmentation_result_t *)seg_thread_args.result;
    seg_thread_args.result = nullptr;
  } else {
    // 使用最新缓存的分割结果
    result.segmentation = get_latest_segmentation();
    ESP_LOGW(TAG, "Using cached segmentation result");
  }

  if (det_done && det_thread_args.result) {
    result.detection = *(detection_result_t *)det_thread_args.result;
    delete (detection_result_t *)det_thread_args.result;
    det_thread_args.result = nullptr;
  } else {
    // 创建空的检测结果
    result.detection.width = frame->width;
    result.detection.height = frame->height;
    ESP_LOGW(TAG, "Detection task not completed");
  }
  pthread_mutex_unlock(&frame_mutex);

  // 使用检测结果更新跟踪器
  result.tracking = update_tracker(result.detection);

  // 使用分割和跟踪结果进行碰撞预测
  result.collision = predict_collisions(result.tracking, result.segmentation);

  int64_t end_time = esp_timer_get_time();
  float process_time = (end_time - start_time) / 1000.0f;

  ESP_LOGI(TAG,
           "Frame processed in %.2f ms with %d detections, %d tracks, overall "
           "risk %.2f",
           process_time, result.detection.boxes.size(),
           result.tracking.tracks.size(), result.collision.overall_risk);

  return result;
}
