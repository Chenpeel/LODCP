#include "../include/init.h"
#include "driver/sdmmc_defs.h"
#include "driver/sdmmc_host.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "nvs_flash.h"
#include "sdmmc_cmd.h"

static const char *TAG = "Init";

// 初始化NVS
esp_err_t init_nvs() {
  ESP_LOGI(TAG, "Initializing NVS");
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_LOGW(TAG, "NVS needs to be erased");
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  return ret;
}

// 初始化摄像头
esp_err_t init_camera() {
  ESP_LOGI(TAG, "Initializing camera");
  esp_err_t ret = esp_camera_init(&camera_config);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Camera init failed with error 0x%x", ret);
    return ret;
  }

  // 设置帧大小
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_framesize(s, FRAMESIZE_SVGA);
    s->set_quality(s, 10); // 降低质量以提高FPS
    s->set_hmirror(s, 0);  // 可根据需要设置水平镜像
    s->set_vflip(s, 0);    // 可根据需要设置垂直翻转
  }

  return ESP_OK;
}

// 初始化SD卡
esp_err_t init_sd_card() {
  ESP_LOGI(TAG, "Initializing SD card");

  // SD卡挂载配置
  esp_vfs_fat_sdmmc_mount_config_t mount_config = {
      .format_if_mount_failed = false,
      .max_files = 5,
      .allocation_unit_size = 16 * 1024};

  // 使用SPI模式
  sdmmc_host_t host = SDMMC_HOST_DEFAULT();
  host.flags = SDMMC_HOST_FLAG_1BIT;        // 使用1位数据线
  host.max_freq_khz = SDMMC_FREQ_HIGHSPEED; // 使用高速模式

  sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
  slot_config.width = 1; // 1位数据线

  // SD卡GPIO配置
  gpio_set_pull_mode(GPIO_NUM_2, GPIO_PULLUP_ONLY); // CMD线上拉
  gpio_set_pull_mode(GPIO_NUM_4, GPIO_PULLUP_ONLY); // D0线上拉

  // 初始化SD卡
  sdmmc_card_t *card;
  esp_err_t ret = esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot_config,
                                          &mount_config, &card);

  if (ret != ESP_OK) {
    if (ret == ESP_FAIL) {
      ESP_LOGE(TAG,
               "Failed to mount the SD card. "
               "If you want to format it, set format_if_mount_failed = true.");
    } else {
      ESP_LOGE(TAG, "Failed to initialize the SD card (%s).",
               esp_err_to_name(ret));
    }
    return ret;
  }

  // 打印SD卡信息
  sdmmc_card_print_info(stdout, card);
  return ESP_OK;
}

esp_err_t initialize_all() {
  // 初始化NVS
  if (init_nvs() != ESP_OK) {
    ESP_LOGE(TAG, "NVS initialization failed");
    return ESP_FAIL;
  }

  // 初始化SD卡
  if (init_sd_card() != ESP_OK) {
    ESP_LOGE(TAG, "SD card initialization failed");
    return ESP_FAIL;
  }

  // 初始化摄像头
  if (init_camera() != ESP_OK) {
    ESP_LOGE(TAG, "Camera initialization failed");
    return ESP_FAIL;
  }

  // 初始化WiFi
  if (init_wifi() != ESP_OK) {
    ESP_LOGW(TAG, "WiFi initialization failed, continuing without WiFi");
    // 不返回失败，可以继续使用其他功能
  }

  ESP_LOGI(TAG, "All hardware initialized successfully");
  return ESP_OK;
}
