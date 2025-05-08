#include "../include/init.h"
#include "driver/sdmmc_defs.h"
#include "driver/sdmmc_host.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_vfs_fat.h"
#include "nvs_flash.h"
#include "sdmmc_cmd.h"

static const char *TAG = "Init";

// 初始化Non-Volatile Storage
esp_err_t init::init_nvs() {
  ESP_LOGI(TAG, "Initializing NVS");
  esp_err_t ret = nvs_flash_init();

  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_LOGW(TAG, "NVS needs to be erased");

    // Erase NVS and retry initialization
    ret = nvs_flash_erase();
    if (ret != ESP_OK) {
      ESP_LOGE(TAG, "Failed to erase NVS: %s", esp_err_to_name(ret));
      return ret;
    }

    ret = nvs_flash_init();
    if (ret != ESP_OK) {
      ESP_LOGE(TAG, "Failed to initialize NVS after erase: %s",
               esp_err_to_name(ret));
    }
  }

  return ret;
}

// Initialize camera module
esp_err_t init::init_camera() {
  ESP_LOGI(TAG, "Initializing camera");
  esp_err_t ret = esp_camera_init(&camera_config);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Camera init failed with error %s (0x%x)",
             esp_err_to_name(ret), ret);
    return ret;
  }

  // Configure frame settings
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_framesize(s, FRAMESIZE_SVGA);
    s->set_quality(s, 10); // Lower quality to increase FPS
    s->set_hmirror(s, 0);  // Horizontal mirror (can be adjusted as needed)
    s->set_vflip(s, 0);    // Vertical flip (can be adjusted as needed)

    ESP_LOGI(TAG, "Camera settings configured: SVGA, quality 10");
  } else {
    ESP_LOGW(TAG, "Failed to get camera sensor, using default settings");
  }

  return ESP_OK;
}

// Initialize SD card
esp_err_t init::init_sd_card() {
  ESP_LOGI(TAG, "Initializing SD card");

  // SD card mounting configuration
  esp_vfs_fat_sdmmc_mount_config_t mount_config = {
      .format_if_mount_failed = false,
      .max_files = 5,
      .allocation_unit_size = 16 * 1024};

  // Use SPI mode for SD card
  sdmmc_host_t host = SDMMC_HOST_DEFAULT();
  host.flags = SDMMC_HOST_FLAG_1BIT;        // Use 1-bit data line
  host.max_freq_khz = SDMMC_FREQ_HIGHSPEED; // Use high-speed mode

  sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
  slot_config.width = 1; // 1-bit data line

  // SD card GPIO configuration
  gpio_set_pull_mode(GPIO_NUM_2, GPIO_PULLUP_ONLY); // Pull up CMD line
  gpio_set_pull_mode(GPIO_NUM_4, GPIO_PULLUP_ONLY); // Pull up D0 line

  // Initialize SD card
  sdmmc_card_t *card;
  esp_err_t ret = esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot_config,
                                          &mount_config, &card);

  if (ret != ESP_OK) {
    if (ret == ESP_FAIL) {
      ESP_LOGE(TAG,
               "Failed to mount the SD card. "
               "If you want to format it, set format_if_mount_failed = true");
    } else {
      ESP_LOGE(TAG, "Failed to initialize the SD card: %s",
               esp_err_to_name(ret));
    }
    return ret;
  }

  // Print SD card info
  ESP_LOGI(TAG, "SD card mounted successfully");
  sdmmc_card_print_info(stdout, card);

  return ESP_OK;
}

// 初始化SPIFFS文件系统
esp_err_t init::init_spiffs() {
  ESP_LOGI(TAG, "Initializing SPIFFS");

  esp_vfs_spiffs_conf_t conf = {
      .base_path = "/spiffs",
      .partition_label = "storage",
      .max_files = 5,
      .format_if_mount_failed = true
  };

  esp_err_t ret = esp_vfs_spiffs_register(&conf);
  if (ret != ESP_OK) {
    if (ret == ESP_FAIL) {
      ESP_LOGE(TAG, "Failed to mount or format SPIFFS");
    } else if (ret == ESP_ERR_NOT_FOUND) {
      ESP_LOGE(TAG, "Failed to find SPIFFS partition");
    } else {
      ESP_LOGE(TAG, "Failed to initialize SPIFFS: %s", esp_err_to_name(ret));
    }
    return ret;
  }

  size_t total = 0, used = 0;
  ret = esp_spiffs_info(conf.partition_label, &total, &used);
  if (ret == ESP_OK) {
    ESP_LOGI(TAG, "SPIFFS Partition: total: %d, used: %d", total, used);
  } else {
    ESP_LOGE(TAG, "Failed to get SPIFFS partition info");
  }

  return ESP_OK;
}

esp_err_t init::initialize_all() {
  esp_err_t ret;
  
  // Initialize NVS
  ret = init_nvs();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "NVS initialization failed: %s", esp_err_to_name(ret));
    return ret;
  }

  // Initialize SD card
  ret = init_sd_card();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "SD card initialization failed: %s", esp_err_to_name(ret));
    return ret;
  }

  // Initialize SPIFFS
  ret = init_spiffs();
  if (ret != ESP_OK) {
    ESP_LOGW(TAG, "SPIFFS initialization failed: %s, continuing without SPIFFS", 
             esp_err_to_name(ret));
    // Not returning error, can continue using other functions
  }

  // Initialize camera
  ret = init_camera();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Camera initialization failed: %s", esp_err_to_name(ret));
    return ret;
  }

  // Initialize WiFi
  ret = init_wifi();
  if (ret != ESP_OK) {
    ESP_LOGW(TAG, "WiFi initialization failed: %s, continuing without WiFi", 
             esp_err_to_name(ret));
    // Not returning error, can continue using other functions
  }

  ESP_LOGI(TAG, "All hardware initialized successfully");
  return ESP_OK;
}
