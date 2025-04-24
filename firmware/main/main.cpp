#include "../include/camera_config.h"
#include "../include/init.h"
#include "../include/process.h"
#include "esp_camera.h"
#include "esp_dsp.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "img_converters.h"
static const char *TAG = "Main";

extern "C" void app_main() {
  ESP_LOGI(TAG, "Initializing NVS");
  if (!init_nvs()) {
    ESP_LOGE(TAG, "NVS initialization failed");
    return;
  }
  // 初始化摄像头
  ESP_LOGI(TAG, "Initializing camera");
  if (init_camera() != ESP_OK) {
    ESP_LOGE(TAG, "Camera initialization failed");
    return;
  }

  // 主循环
  while (true) {
    // 获取一帧图像
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGE(TAG, "Camera capture failed");
      continue;
    }

    // 处理帧
    frame_processing_result_t result = process_frame(fb);

    // 在这里可以根据result做进一步处理或发送结果
    ESP_LOGI(TAG, "Frame processed. Collision risk: %.2f",
             result.collision_risk);

    // 释放帧缓冲区
    esp_camera_fb_return(fb);

    // 适当延迟
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}
