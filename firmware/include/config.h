#ifndef __CONFIG_H__
#define __CONFIG_H__
#include "esp_all.h"
// 定义摄像头模块
#define CAMERA_MODEL_AI_THINKER
#if defined(CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#else
#error "Camera model not selected"
#endif

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sccb_sda = SIOD_GPIO_NUM,
    .pin_sccb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 10000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 20,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

// SPACE
#ifdef CONFIG_PTHREAD_TASK_STACK_SIZE_DEFAULT
#undef CONFIG_PTHREAD_TASK_STACK_SIZE_DEFAULT
#define CONFIG_PTHREAD_TASK_STACK_SIZE_DEFAULT 98304
#endif
#ifdef CONFIG_ESP_SYSTEM_EVENT_TASK_STACK_SIZE
#undef CONFIG_ESP_SYSTEM_EVENT_TASK_STACK_SIZE
#define CONFIG_ESP_SYSTEM_EVENT_TASK_STACK_SIZE 16384
#endif
#ifdef CONFIG_ARDUINO_LOOP_STACK_SIZE
#undef CONFIG_ARDUINO_LOOP_STACK_SIZE
#define CONFIG_ARDUINO_LOOP_STACK_SIZE 32768
#endif
#ifdef CAMERA_DMA_BUFFER_SIZE_MAX
#undef CAMERA_DMA_BUFFER_SIZE_MAX
#define CAMERA_DMA_BUFFER_SIZE_MAX 32768
#endif // SPACE

static sdmmc_host_t sdmmc = SDMMC_HOST_DEFAULT();
static sdmmc_card_t *card = NULL;
static sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
static esp_vfs_fat_sdmmc_mount_config_t mount_config = {
    .format_if_mount_failed = false,
    .max_files = 5,
    .allocation_unit_size = 16 * 1024};

static esp_vfs_spiffs_conf_t spiffs = {.base_path = "/spiffs",
                                       .partition_label = NULL,
                                       .max_files = 5,
                                       .format_if_mount_failed = false};

// wifi
static wifi_init_config_t wifi_init_config = WIFI_INIT_CONFIG_DEFAULT();
#define WIFI_SSID "X"
#define WIFI_PASSWORD "QWE999@@"
#define WIFI_CONNECT_TIMEOUT 10000
// 蓝牙
#define BLE_DEVICE_NAME "ESP32-CAM"

// 日志
#define LOG_LOCAL_LEVEL ESP_LOG_INFO

#define SEGMENTATION_MODEL "/sdcard/models/segmentation.tflite"
#define DETECTION_MODEL "/sdcard/models/detection.tflite"

#define USE_DYNAMIC_MEMORY_ALLOCATION    true // 是否使用动态内存分配

#endif // __CONFIG_H__
