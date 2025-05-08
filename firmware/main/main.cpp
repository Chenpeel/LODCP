#include "../include/camera_config.h"
#include "../include/init.h"
#include "../include/lane_area_detect.h"
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
#define FRAME_DELAY_MS 100       // 帧间延迟，约10 FPS

extern "C" void app_main() {
  // 初始化所有组件
  init initializer;
  if (initializer.initialize_all() != ESP_OK) {
    ESP_LOGE(TAG, "初始化失败");
    return;
  }

  if (init_video_recorder(video_config) != ESP_OK) {
    ESP_LOGE(TAG, "视频录制初始化失败");
    return;
  }

  ESP_LOGI(TAG, "系统初始化成功");

  // 自动开始录制
  if (AUTO_START_RECORDING) {
    start_recording();
  }

  // 记录录制开始时间
  int64_t recording_start_time = esp_timer_get_time();
  // 初始化语义分割模型
  if (init_semantic_segmentation() == ESP_OK) {
    ESP_LOGI(TAG, "语义分割模型初始化成功");
  } else {
    ESP_LOGW(TAG, "语义分割模型初始化失败，将使用传统方法");
  }
  
  // 启动视频服务器
  if (init_wifi() != ESP_OK) {
    ESP_LOGE(TAG, "WiFi初始化失败，无法启动视频服务器");
    return;
  } else {
    if (start_video_server() != ESP_OK) {
      ESP_LOGW(TAG, "视频服务初始化失败");
    } else {
      ESP_LOGI(TAG, "视频服务地址：http://[ESP32-IP-Address]/");
    }
  }
  // 主循环
  int frame_count = 0;
  int64_t total_time = 0;
  int64_t fps_check_time = esp_timer_get_time();
  int fps_frame_count = 0;

  // 创建帧处理结果结构体
  frame_processing_result_t result = {0};

  while (true) {
    // 1. 获取帧
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGE(TAG, "获取相机帧失败");
      vTaskDelay(FRAME_DELAY_MS / portTICK_PERIOD_MS);
      continue;
    }

    // 2. 记录开始时间
    int64_t frame_start = esp_timer_get_time();

    // 3. 处理帧 (车道线检测和语义分割)
    esp_err_t ret = process_frame(fb, result);
    if (ret != ESP_OK) {
      ESP_LOGW(TAG, "帧处理失败: %s", esp_err_to_name(ret));
    }

    // 4. 保存原始帧和处理后的帧
    if (is_recording()) {
      save_raw_frame(fb);
      save_processed_frame(fb, result);
    }

    // 5. 计算处理时间
    int64_t frame_time = esp_timer_get_time() - frame_start;
    total_time += frame_time;
    frame_count++;
    fps_frame_count++;

    // 6. 释放帧和清理资源
    esp_camera_fb_return(fb);
    cleanup_frame_result(result);

    // 7. 每秒输出FPS信息
    int64_t now = esp_timer_get_time();
    if (now - fps_check_time > 1000000) { // 每秒
      float fps = fps_frame_count * 1000000.0f / (now - fps_check_time);
      float avg_time = (fps_frame_count > 0) ? (now - fps_check_time) / (fps_frame_count * 1000.0f) : 0;
      ESP_LOGI(TAG, "FPS: %.2f, 平均帧处理时间: %.2f ms", fps, avg_time);
      fps_check_time = now;
      fps_frame_count = 0;
    }

    // 8. 检查录制时间是否超过限制
    if (AUTO_START_RECORDING && is_recording() && 
        (now - recording_start_time) > MAX_RECORDING_SECONDS * 1000000) {
      ESP_LOGI(TAG, "达到最大录制时间，停止录制");
      stop_recording();
    }

    // 9. 延迟以控制帧率
    vTaskDelay(FRAME_DELAY_MS / portTICK_PERIOD_MS);
  }
}
