#include "../include/camera_config.h"
#include "../include/collision.h"
#include "../include/deep_sort.h"
#include "../include/init.h"
#include "../include/model_loader.h"
#include "../include/process.h"
#include "../include/semantic_seg.h"
#include "../include/video_recorder.h"

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "img_converters.h"

static const char *TAG = "Main";

// 添加启动参数：是否自动开始录制
#define AUTO_START_RECORDING true
#define MAX_RECORDING_SECONDS 60 // 最大录制时间（秒）

extern "C" void app_main() {
  // 初始化所有组件
  if (initialize_all() != ESP_OK) {
    ESP_LOGE(TAG, "Initialization failed");
    return;
  }

  // 初始化处理流水线(加载模型和创建线程)
  if (init_processing_pipeline() != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize processing pipeline");
    return;
  }

  // 初始化视频录制器
  video_config_t video_config = {
      .format = VIDEO_FRAMES,         // 保存为单独的帧序列
      .filename = "driving_sequence", // 基础文件名
      .max_frames = 1800,             // 最大帧数 (60秒 * 30fps)
      .fps = 30,                      // 目标帧率
      .include_timestamp = true,      // 在文件名中包含时间戳
      .quality = 10,                  // JPEG质量
      .draw_detections = true,        // 在处理后的帧上绘制检测结果
      .draw_segmentation = true       // 在处理后的帧上绘制分割结果
  };

  if (init_video_recorder(video_config) != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize video recorder");
    return;
  }

  ESP_LOGI(TAG, "System initialized successfully, entering main loop");

  // 自动开始录制
  if (AUTO_START_RECORDING) {
    start_recording();
  }

  // 记录录制开始时间
  int64_t recording_start_time = esp_timer_get_time();
  // 启动视频服务器
  if (start_video_server() != ESP_OK) {
    ESP_LOGW(TAG, "Failed to start video server");
  } else {
    ESP_LOGI(TAG,
             "Video server started, access via http://[ESP32-IP-Address]/");
  }
  // 主循环
  int frame_count = 0;
  int64_t total_time = 0;

  while (true) {
    int64_t frame_start = esp_timer_get_time();

    // 获取一帧图像
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGE(TAG, "Camera capture failed");
      vTaskDelay(1000 / portTICK_PERIOD_MS);
      continue;
    }

    ESP_LOGI(TAG, "Frame %d captured: %dx%d", frame_count, fb->width,
             fb->height);

    // 保存原始帧
    if (is_recording()) {
      save_raw_frame(fb);
    }

    // 处理帧
    frame_processing_result_t result = process_frame(fb);

    // 保存处理后的帧
    if (is_recording()) {
      save_processed_frame(fb, result);

      // 检查是否超过最大录制时间
      int64_t elapsed_seconds =
          (esp_timer_get_time() - recording_start_time) / 1000000;
      if (elapsed_seconds >= MAX_RECORDING_SECONDS) {
        ESP_LOGI(
            TAG,
            "Reached maximum recording time (%d seconds), stopping recording",
            MAX_RECORDING_SECONDS);
        stop_recording();
      }
    }

    // 显示处理结果摘要
    ESP_LOGI(TAG, "Frame %d processed:", frame_count);
    ESP_LOGI(TAG, "- Detections: %d objects", result.detection.boxes.size());
    ESP_LOGI(TAG, "- Tracking: %d active tracks",
             result.tracking.tracks.size());
    ESP_LOGI(TAG, "- Collision risk: %.2f", result.collision.overall_risk);

    if (result.collision.overall_risk > 0.7f) {
      ESP_LOGW(TAG, "HIGH COLLISION RISK DETECTED!");

      // 如果检测到高风险且未录制，开始录制
      if (!is_recording()) {
        ESP_LOGI(TAG, "Starting recording due to high collision risk");
        start_recording();
        recording_start_time = esp_timer_get_time();
      }
    }

    // 释放帧缓冲区
    esp_camera_fb_return(fb);

    // 计算处理时间
    int64_t frame_end = esp_timer_get_time();
    int64_t frame_time = frame_end - frame_start;
    total_time += frame_time;

    ESP_LOGI(TAG, "Frame processing time: %lld ms", frame_time / 1000);

    frame_count++;
    if (frame_count % 10 == 0) {
      float avg_time = total_time / (frame_count * 1000.0f);
      ESP_LOGI(TAG, "Average processing time: %.2f ms (%.2f FPS)", avg_time,
               1000.0f / avg_time);

      // 检查剩余存储空间
      float remaining_storage = get_remaining_storage();
      if (remaining_storage > 0) {
        ESP_LOGI(TAG, "Remaining storage: %.1f MB", remaining_storage);

        // 如果存储空间不足10MB且正在录制，停止录制
        if (remaining_storage < 10.0f && is_recording()) {
          ESP_LOGW(TAG, "Low storage space, stopping recording");
          stop_recording();
        }
      }
    }

    // 控制帧率
    int64_t remaining_time =
        33333 - (esp_timer_get_time() - frame_start); // 约30fps
    if (remaining_time > 0) {
      usleep(remaining_time);
    }
  }
}
