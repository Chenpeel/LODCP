#include "../include/init.h"
#include "SD_MMC.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
static const char *TAG = "Init";

static esp_err_t init_camera() {
  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Camera Init Failed");
    return err;
  }
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV2640_PID) {
    // 针对OV2640进行特殊配置
    s->set_vflip(s, 1);   // 垂直翻转
    s->set_hmirror(s, 1); // 水平镜像
  }
  ESP_LOGI(TAG, "Camera Init Success");
  // SD卡初始化
  if (!SD_MMC.begin()) {
    ESP_LOGE(TAG, "SD Card Mount Failed");
    return ESP_FAIL;
  }
  return ESP_OK;
}

// 初始化NVS
static esp_err_t init_nvas() {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  return ret;
}
