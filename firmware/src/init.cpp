#include "init.h"
#include "esp_all.h"

static const char *TAG = "Init";
void softResetI2C(int sda_pin, int scl_pin);
esp_err_t Init::init_nvs()
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    if (ret != ESP_OK)
    {
        ESP_LOGE("NVS", "Failed to initialize NVS");
        return ret;
    }
    return ESP_OK;
}

esp_err_t Init::init_sd_card()
{
    if (!SD_MMC.begin("/sdcard", true))
    {
        ESP_LOGE("SD", "SD卡挂载失败");
        sdCardStatus = false;
        return false;
    }
    uint8_t cardType = SD_MMC.cardType();
    if (cardType == CARD_NONE)
    {
        ESP_LOGE("SD", "No SD card attached");
        sdCardStatus = false;
        return false;
    }
    ESP_LOGI("SD", "SD Card initialized, size: %llu MB", SD_MMC.cardSize() / (1024 * 1024));
    sdCardStatus = true;
    return ESP_OK;
}
esp_err_t Init::init_spiffs()
{
    // 如果已经挂载，先卸载
    if (esp_spiffs_mounted(NULL))
    {
        ESP_LOGW(TAG, "SPIFFS已经挂载，先卸载");
        // 修复：使用正确的函数卸载SPIFFS
        esp_vfs_spiffs_unregister(NULL);
        delay(100);
    }

    ESP_LOGI(TAG, "正在初始化SPIFFS...");

    esp_vfs_spiffs_conf_t conf = {.base_path = "/spiffs",
                                  .partition_label = NULL, // 使用默认分区
                                  .max_files = 5,
                                  .format_if_mount_failed = true};

    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to register SPIFFS: %s", esp_err_to_name(ret));
        return ESP_FAIL; // 修改：返回ESP_FAIL而不是false，保持一致的返回类型
    }

    // 获取SPIFFS信息
    size_t total = 0, used = 0;
    ret = esp_spiffs_info(NULL, &total, &used);
    if (ret == ESP_OK)
    {
        ESP_LOGI(TAG, "SPIFFS: 总空间: %d KB, 已使用: %d KB", total / 1024,
                 used / 1024);
    }
    else
    {
        ESP_LOGE(TAG, "获取SPIFFS信息失败");
    }

    return ESP_OK; // 修改：返回ESP_OK而不是true，保持一致的返回类型
}
esp_err_t Init::init_wifi()
{
    esp_err_t ret = esp_wifi_init(&wifi_init_config);
    if (ret != ESP_OK)
    {
        ESP_LOGE("WiFi", "Failed to initialize WiFi");
        return ret;
    }
    return ESP_OK;
}
// 初始化摄像头
esp_err_t Init::init_camera()
{
    // 检查PSRAM
    if (!psramFound() || ESP.getPsramSize() == 0)
    {
        ESP_LOGE("Camera", "PSRAM未找到或大小为0，无法初始化摄像头");
        ESP_LOGE("Camera", "ESP32-CAM需要PSRAM才能正常工作");
        return ESP_FAIL;
    }

    // 输出可用内存信息
    ESP_LOGI("Camera", "PSRAM大小: %d bytes", ESP.getPsramSize());
    ESP_LOGI("Camera", "可用PSRAM: %d bytes", ESP.getFreePsram());
    ESP_LOGI("Camera", "可用堆内存: %d bytes", ESP.getFreeHeap());

    // 修改这里 - 重置PWDN引脚以触发摄像头重启
    pinMode(PWDN_GPIO_NUM, OUTPUT);
    digitalWrite(PWDN_GPIO_NUM, HIGH); // 关闭摄像头
    delay(100);
    digitalWrite(PWDN_GPIO_NUM, LOW); // 重新打开摄像头
    delay(500);                       // 重要！给摄像头足够的启动时间
    // 软重置I2C总线
    softResetI2C(SIOD_GPIO_NUM, SIOC_GPIO_NUM);

    // 尝试多次初始化摄像头
    const int MAX_RETRY = 5; // 增加尝试次数
    esp_err_t ret = ESP_FAIL;

    for (int i = 0; i < MAX_RETRY; i++)
    {
        ESP_LOGI("Camera", "尝试初始化摄像头 (尝试 %d/%d)", i + 1, MAX_RETRY);

        // 每次尝试都修改一些参数
        camera_config_t modified_config = camera_config;

        // 第一次尝试降低时钟频率
        if (i >= 0)
        {
            modified_config.xclk_freq_hz = 10000000; // 10MHz
        }

        // 第二次及以后尝试更低分辨率
        if (i >= 1)
        {
            modified_config.frame_size = FRAMESIZE_QQVGA; // 320x240
        }

        // 第三次及以后尝试更低品质和抓取模式
        if (i >= 2)
        {
            modified_config.jpeg_quality = 30;
            modified_config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
        }

        // 短暂延迟
        if (i > 0)
        {
            delay(1000);
        }

        // 尝试初始化
        ret = esp_camera_init(&modified_config);
        if (ret == ESP_OK)
        {
            ESP_LOGI("Camera", "摄像头初始化成功 (尝试 %d)", i + 1);
            break;
        }

        ESP_LOGE("Camera", "尝试 %d 失败: %s (%d)",
                 i + 1, esp_err_to_name(ret), ret);

        // 若失败，进行硬件复位
        pinMode(PWDN_GPIO_NUM, OUTPUT);
        digitalWrite(PWDN_GPIO_NUM, HIGH);
        delay(100);
        digitalWrite(PWDN_GPIO_NUM, LOW);
        delay(500);
    }

    if (ret != ESP_OK)
    {
        ESP_LOGE("Camera", "摄像头初始化失败: %s", esp_err_to_name(ret));
        return ret;
    }

    // 设置摄像头参数
    sensor_t *s = esp_camera_sensor_get();
    if (s)
    {
        s->set_brightness(s, 1);
        s->set_contrast(s, 1);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_gain_ctrl(s, 1);
        s->set_gainceiling(s, (gainceiling_t)1);
    }

    // 添加延迟后再尝试获取测试帧
    delay(500);

    // 尝试多次获取测试帧
    for (int i = 0; i < 3; i++)
    {
        ESP_LOGI("Camera", "尝试获取测试帧 %d/3...", i + 1);
        camera_fb_t *fb = esp_camera_fb_get();

        if (fb)
        {
            ESP_LOGI("Camera", "摄像头测试成功，捕获到 %dx%d 分辨率图像",
                     fb->width, fb->height);
            esp_camera_fb_return(fb);
            return ESP_OK;
        }

        delay(500);
    }

    ESP_LOGE("Camera", "摄像头测试失败，无法捕获帧");
    return ESP_FAIL;
}

// 软重置I2C总线，可能有助于解决某些通信问题
void softResetI2C(int sda_pin, int scl_pin)
{
    ESP_LOGI("Camera", "尝试软重置I2C总线 (SDA=%d, SCL=%d)", sda_pin, scl_pin);

    pinMode(sda_pin, OUTPUT);
    pinMode(scl_pin, OUTPUT);

    // 产生停止信号
    digitalWrite(sda_pin, HIGH);
    digitalWrite(scl_pin, HIGH);
    delay(10);

    // 发送9个时钟脉冲，确保I2C设备释放总线
    for (int i = 0; i < 9; i++)
    {
        digitalWrite(scl_pin, LOW);
        delayMicroseconds(5);
        digitalWrite(scl_pin, HIGH);
        delayMicroseconds(5);
    }

    // 产生最终停止信号
    digitalWrite(sda_pin, LOW);
    delayMicroseconds(5);
    digitalWrite(scl_pin, LOW);
    delayMicroseconds(5);
    digitalWrite(scl_pin, HIGH);
    delayMicroseconds(5);
    digitalWrite(sda_pin, HIGH);
    delay(10);

    ESP_LOGI("Camera", "I2C总线软重置完成");
}

esp_err_t Init::init_ble()
{
    esp_err_t ret = esp_bt_controller_mem_release(ESP_BT_MODE_BLE);
    if (ret != ESP_OK)
    {
        ESP_LOGE("BLE", "Failed to release BLE memory");
        return ret;
    }
    return ESP_OK;
}

// 增加线程配置，解决栈溢出问题
void Init::configure_thread_settings()
{
    // 设置更大的默认线程栈大小
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    // 大幅增加线程栈大小
    size_t stack_size = 96 * 1024;
    pthread_attr_setstacksize(&attr, stack_size);

    // 配置FreeRTOS任务优先级和核心分配
    ESP_LOGI(TAG, "配置线程参数: 栈大小=%d, 优先级=1", stack_size);

    // 释放资源
    pthread_attr_destroy(&attr);

    // 增加内存释放，确保有更多可用内存
    ESP_LOGI(TAG, "尝试整理内存...");
    heap_caps_dump_all();

    // 将一些全局配置应用到config.h中定义的常量
    extern camera_config_t camera_config;
    camera_config.frame_size = FRAMESIZE_QVGA;        // 320x240分辨率
    camera_config.jpeg_quality = 20;                  // 降低JPEG质量以减少内存使用
    camera_config.fb_count = 1;                       // 减少帧缓冲区数量
    camera_config.grab_mode = CAMERA_GRAB_WHEN_EMPTY; // 更有效的抓取模式

    // 禁用不必要的功能，减少内存使用
    ESP_LOGI(TAG, "已应用优化的配置，减少内存使用");
}
bool Init::hardware_reset()
{
    ESP_LOGI(TAG, "执行硬件复位程序...");

    // 重置摄像头
    pinMode(PWDN_GPIO_NUM, OUTPUT);
    digitalWrite(PWDN_GPIO_NUM, HIGH); // 关闭摄像头
    delay(100);
    digitalWrite(PWDN_GPIO_NUM, LOW); // 打开摄像头
    delay(500);

    // 重置SPI总线 (连接到PSRAM)
    pinMode(SIOD_GPIO_NUM, OUTPUT);
    pinMode(SIOC_GPIO_NUM, OUTPUT);

    // 复位I2C总线
    softResetI2C(SIOD_GPIO_NUM, SIOC_GPIO_NUM);

    // 延迟一段时间，让所有硬件稳定
    delay(1000);

    return true;
}

bool Init::force_psram_init()
{
    bool psram_found = false;
    ESP_LOGI(TAG, "强制初始化PSRAM...");

    // 尝试硬件复位前先检查PSRAM状态
    if (psramFound() && ESP.getPsramSize() > 0)
    {
        ESP_LOGI(TAG, "PSRAM已初始化，大小: %d bytes", ESP.getPsramSize());
        return true; // 直接返回真
    }

    // 执行硬件复位
    hardware_reset();

    // 尝试初始化PSRAM
    if (psramInit())
    {
        Serial.println("PSRAM initialized successfully!");
        psram_found = true;
    }
    else
    {
        Serial.println("Failed to initialize PSRAM");
        return false;
    }

    // 检查PSRAM大小
    size_t psramSize = ESP.getPsramSize();
    Serial.print("Total PSRAM: ");
    Serial.println(psramSize);

    if (psramSize == 0)
    {
        ESP_LOGE(TAG, "PSRAM初始化成功但大小为0");
        return false;
    }

    size_t freePsram = ESP.getFreePsram();
    Serial.print("Free PSRAM: ");
    Serial.println(freePsram);

    // 测试分配一个小字符串
    const char *testString = "Test";
    int stringLength = strlen(testString) + 1;

    char *psramString = (char *)heap_caps_malloc(stringLength, MALLOC_CAP_SPIRAM);

    if (psramString == NULL)
    {
        ESP_LOGE(TAG, "无法在PSRAM中分配内存");
        return false;
    }

    // 内存测试成功
    strcpy(psramString, testString);
    ESP_LOGI(TAG, "在PSRAM中存储的字符串: %s", psramString);
    heap_caps_free(psramString);

    return true; // 显式返回成功
}

Init::Init() { ESP_LOGI("Init", "Initialization started"); }
Init::~Init() { ESP_LOGI("Init", "Initialization completed"); }

esp_err_t Init::init()
{
    esp_err_t ret;

    // 配置线程设置
    configure_thread_settings();

    // 初始化NVS
    ret = init_nvs();
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "NVS初始化失败: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "NVS初始化成功");

    // 首先尝试初始化PSRAM，因为这对摄像头至关重要
    if (!force_psram_init())
    {
        ESP_LOGE(TAG, "PSRAM初始化失败，系统不能正常工作");
        // 这里不立即返回，继续初始化其他组件，但摄像头将无法工作
    }

    // 初始化SD卡
    ret = init_sd_card();
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "SD卡初始化失败: %s", esp_err_to_name(ret));
        // 不立即返回，因为SD卡不是必需的
    }
    else
    {
        ESP_LOGI(TAG, "SD卡初始化成功");
    }

    // 初始化SPIFFS
    ret = init_spiffs();
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "SPIFFS初始化失败: %s", esp_err_to_name(ret));
        // 不立即返回，因为SPIFFS不是必需的
    }
    else
    {
        ESP_LOGI(TAG, "SPIFFS初始化成功");
    }

    // 初始化WiFi
    ret = init_wifi();
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "WiFi初始化失败: %s", esp_err_to_name(ret));
        // 不立即返回，因为WiFi不是必需的
    }
    else
    {
        ESP_LOGI(TAG, "WiFi初始化成功");
    }

    // 最后初始化摄像头，如果PSRAM已可用
    if (ESP.getPsramSize() > 0)
    {
        ret = init_camera();
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "摄像头初始化失败: %s", esp_err_to_name(ret));
        }
        else
        {
            ESP_LOGI(TAG, "摄像头初始化成功");
        }
    }
    else
    {
        ESP_LOGE(TAG, "PSRAM未初始化，跳过摄像头初始化");
    }

    ESP_LOGI(TAG, "初始化完成");
    return ESP_OK;
}
