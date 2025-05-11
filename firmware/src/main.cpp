#include "common_types.h"
#include "config.h"
#include "esp_all.h"
#include "init.h"
#include "process_stream.h"
#include <atomic>
#include <memory>

// 图像帧缓存管理
#define MAX_CACHED_FRAMES 30
bool frameInUse[MAX_CACHED_FRAMES] = {false};         // 用于跟踪缓存帧是否被使用
int currentFrameIndex = 0;                            // 当前帧索引，用于循环缓冲区
bool frameBufferInitialized = false;                  // 标记帧缓冲是否已初始化
FrameCacheMode frameCacheMode = SD_CARD;              // 默认使用SD卡
bool sdCardAvailable = false;                         // SD卡是否可用
const int RAM_FRAME_BUFFER_SIZE = 6;                  // RAM中最多保存的帧数量
std::vector<std::shared_ptr<uint8_t>> ramFrameBuffer; // RAM帧缓冲区
static const char *TAG = "Main";

// 图像缓冲区
const int MAX_WIDTH = 1280;
const int MAX_HEIGHT = 720;
const int MAX_CHANNELS = 3;
uint8_t *imageBuffer = nullptr;

// 帧率计算相关变量
int frameCount = 0;
unsigned long startTime = 0;

// 全局处理流实例
std::unique_ptr<ProcessStream> processor;
static Init sysInit;

// 全局停止标志
std::atomic<bool> g_running(true);

// 录制控制变量
const int LED_RECORD_PIN = 33; // 录制状态LED引脚，通常ESP32-CAM板上有GPIO33的LED
bool isRecording = true;

// 自动录制控制
bool hasHighRiskEvent = false;
unsigned long highRiskEventTime = 0;
const unsigned long AUTO_RECORD_DURATION = 30 * 1000; // 高风险后自动录制30秒
const unsigned long AUTO_RECORD_COOLDOWN = 5 * 1000;  // 冷却期5秒，避免频繁启停

// 初始化帧缓冲区
void initializeFrameBuffer()
{
    ESP_LOGI(TAG, "初始化帧缓冲区...");

    // 创建临时目录（如果不存在）
    if (!SD_MMC.exists("/temp"))
    {
        ESP_LOGI(TAG, "创建临时目录: /temp");
        SD_MMC.mkdir("/temp");
    }

    // 创建固定数量的空文件用于循环缓冲
    for (int i = 0; i < MAX_CACHED_FRAMES; i++)
    {
        char framePath[64];
        snprintf(framePath, sizeof(framePath), "/temp/frame_%d.dat", i);

        // 如果文件已存在，先删除
        if (SD_MMC.exists(framePath))
        {
            SD_MMC.remove(framePath);
        }

        // 创建空文件
        File f = SD_MMC.open(framePath, FILE_WRITE);
        if (f)
        {
            f.close();
            ESP_LOGI(TAG, "创建帧缓冲文件: %s", framePath);
        }
        else
        {
            ESP_LOGE(TAG, "无法创建帧缓冲文件: %s", framePath);
        }

        // 初始化帧使用状态
        frameInUse[i] = false;
    }

    frameBufferInitialized = true;
    ESP_LOGI(TAG, "帧缓冲区初始化完成");
}

// 帧处理完成回调函数
void frameProcessedCallback(int frameId)
{
    // 标记帧已处理完成
    if (frameId >= 0 && frameId < MAX_CACHED_FRAMES)
    {
        ESP_LOGI(TAG, "帧 #%d 处理完成，标记为可用", frameId);
        frameInUse[frameId] = false;
    }
}

// 清理未使用的帧
void cleanupUnusedFrames()
{
    ESP_LOGI(TAG, "开始清理未使用的帧...");
    int cleanedCount = 0;

    for (int i = 0; i < MAX_CACHED_FRAMES; i++)
    {
        if (!frameInUse[i])
        {
            char framePath[64];
            snprintf(framePath, sizeof(framePath), "/temp/frame_%d.dat", i);

            // 检查文件是否存在
            if (SD_MMC.exists(framePath))
            {
                // 重置文件（保留文件但清空内容）
                SD_MMC.remove(framePath);
                File f = SD_MMC.open(framePath, FILE_WRITE);
                if (f)
                {
                    f.close();
                    cleanedCount++;
                }
            }
        }
    }

    ESP_LOGI(TAG, "帧清理完成，已重置 %d 个帧文件", cleanedCount);
}

// 确保录制状态
void ensureRecording()
{
    // 仅在未录制状态时启动录制
    if (!isRecording)
    {
        // 获取当前相机分辨率
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb)
        {
            // 开始录制
            if (processor->startVideoRecording(fb->width, fb->height, 15))
            {
                isRecording = true;
                // 点亮LED指示灯
                digitalWrite(LED_RECORD_PIN, HIGH);
                ESP_LOGI(TAG, "自动开始视频录制，分辨率: %dx%d", fb->width, fb->height);
            }
            esp_camera_fb_return(fb);
        }
        else
        {
            ESP_LOGE(TAG, "无法获取相机帧，录制失败");
        }
    }

    // 如果有高风险事件，只记录日志，不改变录制状态
    if (hasHighRiskEvent)
    {
        ESP_LOGI(TAG, "检测到高风险事件，继续录制中");
        hasHighRiskEvent = false; // 重置标志
    }
}

// 结果回调函数
void processResultCallback(const std::shared_ptr<FrameData> &frameData)
{
    // 在这里处理最终结果
    if (!frameData->ttcResults.empty())
    {
        // 筛选高风险目标
        bool hasHighRisk = false;
        for (const auto &ttc : frameData->ttcResults)
        {
            if (ttc.risk > 0.7f)
            {
                hasHighRisk = true;
                ESP_LOGI(TAG, "高风险警告 - 目标ID:%d 类型:%d TTC:%.2fs 风险:%.2f",
                         ttc.trackId, ttc.classId, ttc.ttc, ttc.risk);
            }
        }

        // 如果有高风险目标，更新高风险事件
        if (hasHighRisk)
        {
            ESP_LOGI(TAG, "触发警报：检测到高风险目标");
            hasHighRiskEvent = true;
            highRiskEventTime = millis();
        }
    }

    // 标记帧已完成处理
    if (frameData->frameId >= 0)
    {
        frameProcessedCallback(frameData->frameId);
    }
}

bool addFrameToProcess(uint8_t *buffer, int width, int height, int channels, double timestamp)
{
    if (!processor)
    {
        return false;
    }

    // 根据帧缓存模式选择不同处理方式
    switch (frameCacheMode)
    {
    case SD_CARD:
        if (sdCardAvailable)
        {
            // 原有SD卡模式
            return processor->addFrame(buffer, width, height, channels, timestamp);
        }
        else
        {
            frameCacheMode = RAM_ONLY;
            ESP_LOGW(TAG, "SD卡不可用，切换到RAM模式");
            return addFrameToProcess(buffer, width, height, channels, timestamp);
        }
        break;

    case RAM_ONLY:
    {
        // 使用RAM存储帧
        size_t frameSize = width * height * channels;

        // 检查当前队列大小
        if (ramFrameBuffer.size() >= RAM_FRAME_BUFFER_SIZE)
        {
            // 队列已满，移除最旧的帧
            ESP_LOGD(TAG, "RAM帧队列已满，移除最旧的帧");
            ramFrameBuffer.erase(ramFrameBuffer.begin());
        }

        // 分配新帧内存并复制数据
        std::shared_ptr<uint8_t> frameData(new uint8_t[frameSize], std::default_delete<uint8_t[]>());
        if (!frameData)
        {
            ESP_LOGE(TAG, "无法为帧分配RAM");
            return false;
        }

        memcpy(frameData.get(), buffer, frameSize);
        ramFrameBuffer.push_back(frameData);

        // 使用直接内存方式添加帧
        return processor->addFrameDirect(frameData.get(), width, height, channels, timestamp, ramFrameBuffer.size() - 1);
    }
    break;

    case HYBRID:
        // 混合模式暂未实现
        return false;
    }

    return false;
}

// 从摄像头获取图像帧
bool getNextFrame(uint8_t *buffer, int &width, int &height, int &channels)
{
    // 定义静态变量
    static unsigned long lastCaptureTime = 0;
    static int failCount = 0;
    static unsigned long lastResetTime = 0;
    const unsigned long resetCooldown = 5000; // 重置冷却时间(毫秒)

    // 动态计算捕获间隔
    unsigned long captureInterval = 30; // 默认基础间隔

    // 根据处理队列状态动态调整间隔
    if (processor)
    {
        size_t queueSize = processor->getRawQueueSize();
        // 队列越满，间隔越长
        captureInterval = 30 + (queueSize * 15); // 每增加一个队列项，增加15ms延迟

        // 设置上限，避免间隔过长
        if (captureInterval > 500)
            captureInterval = 500; // 最大500ms
    }

    // 检查buffer是否为NULL
    if (buffer == NULL)
    {
        ESP_LOGE(TAG, "图像缓冲区为NULL，无法获取帧");
        return false;
    }

    // 防抖处理，避免过快请求摄像头
    unsigned long currentTime = millis();
    if (currentTime - lastCaptureTime < captureInterval)
    {
        delay(5);
        return false;
    }

    // 输出调试信息
    static unsigned long last_debug_time = 0;
    if (currentTime - last_debug_time > 5000)
    { // 每5秒输出一次
        ESP_LOGD(TAG, "获取帧 - 缓冲区地址: %p, PSRAM: %d bytes, 堆: %d bytes",
                 buffer, ESP.getFreePsram(), ESP.getFreeHeap());
        last_debug_time = currentTime;
    }

    // 记录捕获时间
    lastCaptureTime = currentTime;

    // 记录缓冲区指针用于调试
    ESP_LOGD(TAG, "获取帧，缓冲区地址: %p", buffer);

    // 确保全局常量已定义
    if (MAX_WIDTH <= 0 || MAX_HEIGHT <= 0 || MAX_CHANNELS <= 0)
    {
        ESP_LOGE(TAG, "无效的图像参数: MAX_WIDTH=%d, MAX_HEIGHT=%d, MAX_CHANNELS=%d",
                 MAX_WIDTH, MAX_HEIGHT, MAX_CHANNELS);
        return false;
    }

    // 在获取新帧前检查内存状态
    size_t freePsram = ESP.getFreePsram();
    size_t freeHeap = ESP.getFreeHeap();

    // 如果内存不足，尝试进行内存清理
    if (freePsram < 100000 || freeHeap < 20000)
    {
        ESP_LOGW(TAG, "内存不足，尝试释放资源: PSRAM=%d, 堆=%d", freePsram, freeHeap);

        // 触发一些内存整理操作
        ESP.getPsramSize();

        // 触发清理未使用的帧
        cleanupUnusedFrames();

        // 给系统一些恢复时间
        delay(50);
    }

    // 获取摄像头帧缓冲
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
        failCount++;

        // 如果摄像头连续失败，尝试重新初始化
        if (failCount > 3)
        {
            ESP_LOGW(TAG, "摄像头连续失败");
            // 每隔30帧尝试重新初始化摄像头
            if (failCount % 30 == 0 && currentTime - lastResetTime > resetCooldown)
            {
                ESP_LOGI(TAG, "尝试重新初始化摄像头");
                esp_camera_deinit();
                delay(500);
                esp_err_t err = esp_camera_init(&camera_config);
                if (err == ESP_OK)
                {
                    ESP_LOGI(TAG, "摄像头已重新初始化");
                    failCount = 0;
                }

                lastResetTime = currentTime;
            }

            return false;
        }

        ESP_LOGE(TAG, "摄像头捕获失败");
        delay(100);
        return false;
    }

    // 重置失败计数
    failCount = 0;

    // 打印帧信息
    ESP_LOGD(TAG, "捕获成功: %dx%d, 格式=%d, 大小=%d bytes", fb->width,
             fb->height, fb->format, fb->len);

    // 设置输出参数
    width = fb->width;
    height = fb->height;

    bool result = false;

    // 确定通道数并处理不同格式
    if (fb->format == PIXFORMAT_JPEG)
    {
        // 对于JPEG格式，我们可以直接处理
        channels = 1; // JPEG视为单通道数据

        // 安全检查缓冲区大小
        if (fb->len <= MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS)
        {
            // 直接复制整个JPEG数据
            memcpy(buffer, fb->buf, fb->len);
            result = true;
        }
        else
        {
            ESP_LOGE(TAG, "JPEG数据太大: %d bytes，缓冲区大小: %d bytes", fb->len,
                     MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
            result = false;
        }
    }
    else if (fb->format == PIXFORMAT_RGB565)
    {
        // 处理RGB565格式转RGB888
        channels = 3;

        // 计算需要的缓冲区大小
        size_t pixelCount = fb->width * fb->height;
        size_t requiredSize = pixelCount * 3;

        // 安全检查
        if (requiredSize <= MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS)
        {
            // 将RGB565转换为RGB888
            uint16_t *rgb565 = (uint16_t *)fb->buf;

#ifdef CONFIG_IDF_TARGET_ESP32
            // 使用循环展开优化性能 (每次处理4个像素)
            size_t i = 0;
            size_t destIdx = 0; // 使用单独的目标索引
            size_t blocks = pixelCount / 4;

            for (size_t b = 0; b < blocks; b++)
            {
                uint16_t pixel0 = rgb565[i];
                uint16_t pixel1 = rgb565[i + 1];
                uint16_t pixel2 = rgb565[i + 2];
                uint16_t pixel3 = rgb565[i + 3];

                // 像素0
                uint8_t r0 = (pixel0 >> 11) & 0x1F;
                uint8_t g0 = (pixel0 >> 5) & 0x3F;
                uint8_t b0 = pixel0 & 0x1F;
                buffer[destIdx++] = (r0 << 3) | (r0 >> 2); // 扩展到8位
                buffer[destIdx++] = (g0 << 2) | (g0 >> 4);
                buffer[destIdx++] = (b0 << 3) | (b0 >> 2);

                // 像素1
                uint8_t r1 = (pixel1 >> 11) & 0x1F;
                uint8_t g1 = (pixel1 >> 5) & 0x3F;
                uint8_t b1 = pixel1 & 0x1F;
                buffer[destIdx++] = (r1 << 3) | (r1 >> 2);
                buffer[destIdx++] = (g1 << 2) | (g1 >> 4);
                buffer[destIdx++] = (b1 << 3) | (b1 >> 2);

                // 像素2
                uint8_t r2 = (pixel2 >> 11) & 0x1F;
                uint8_t g2 = (pixel2 >> 5) & 0x3F;
                uint8_t b2 = pixel2 & 0x1F;
                buffer[destIdx++] = (r2 << 3) | (r2 >> 2);
                buffer[destIdx++] = (g2 << 2) | (g2 >> 4);
                buffer[destIdx++] = (b2 << 3) | (b2 >> 2);

                // 像素3
                uint8_t r3 = (pixel3 >> 11) & 0x1F;
                uint8_t g3 = (pixel3 >> 5) & 0x3F;
                uint8_t b3 = pixel3 & 0x1F;
                buffer[destIdx++] = (r3 << 3) | (r3 >> 2);
                buffer[destIdx++] = (g3 << 2) | (g3 >> 4);
                buffer[destIdx++] = (b3 << 3) | (b3 >> 2);

                i += 4;
            }

            // 处理剩余像素
            for (; i < pixelCount; i++)
            {
                uint16_t pixel = rgb565[i];
                uint8_t r = (pixel >> 11) & 0x1F;
                uint8_t g = (pixel >> 5) & 0x3F;
                uint8_t b = pixel & 0x1F;

                buffer[destIdx++] = (r << 3) | (r >> 2); // 扩展5位到8位
                buffer[destIdx++] = (g << 2) | (g >> 4); // 扩展6位到8位
                buffer[destIdx++] = (b << 3) | (b >> 2); // 扩展5位到8位
            }
#else
            // 标准转换方法
            for (size_t i = 0; i < pixelCount; i++)
            {
                uint16_t pixel = rgb565[i];

                // 提取RGB565的各个分量
                uint8_t r = (pixel >> 11) & 0x1F;
                uint8_t g = (pixel >> 5) & 0x3F;
                uint8_t b = pixel & 0x1F;

                // 将5位和6位值扩展到8位，以保持最大色彩精度
                buffer[i * 3] = (r << 3) | (r >> 2);     // 扩展5位到8位
                buffer[i * 3 + 1] = (g << 2) | (g >> 4); // 扩展6位到8位
                buffer[i * 3 + 2] = (b << 3) | (b >> 2); // 扩展5位到8位
            }
#endif

            result = true;
        }
        else
        {
            ESP_LOGE(TAG, "转换后图像太大: %d bytes，超过缓冲区: %d bytes",
                     requiredSize, MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
            result = false;
        }
    }
    else if (fb->format == PIXFORMAT_GRAYSCALE)
    {
        channels = 1;

        // 安全检查
        if (fb->len <= MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS)
        {
            // 复制灰度数据
            memcpy(buffer, fb->buf, fb->len);
            result = true;
        }
        else
        {
            ESP_LOGE(TAG, "灰度数据太大: %d bytes", fb->len);
            result = false;
        }
    }
    else if (fb->format == PIXFORMAT_RGB888)
    {
        // 原生RGB888格式
        channels = 3;

        // 计算实际大小
        size_t imageSize = width * height * channels;

        // 安全检查缓冲区大小
        if (imageSize <= MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS)
        {
            memcpy(buffer, fb->buf, imageSize);
            result = true;
        }
        else
        {
            ESP_LOGE(TAG, "RGB888图像太大: %d bytes", imageSize);
            result = false;
        }
    }
    else
    {
        // 其他未知格式，尝试作为普通缓冲区复制
        channels = 3; // 假设是3通道

        ESP_LOGW(TAG, "未知格式: %d，尝试直接复制", fb->format);

        // 安全检查
        if (fb->len <= MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS)
        {
            memcpy(buffer, fb->buf, fb->len);
            result = true;
        }
        else
        {
            ESP_LOGE(TAG, "未知格式数据太大: %d bytes", fb->len);
            result = false;
        }
    }

    // 返回帧缓冲
    esp_camera_fb_return(fb);

    // 定期报告内存状态(每100帧)
    static int frameCounter = 0;
    frameCounter++;
    if (frameCounter % 100 == 0)
    {
        ESP_LOGI(TAG, "内存状态 - PSRAM剩余: %d bytes, 堆内存: %d bytes",
                 ESP.getFreePsram(), ESP.getFreeHeap());
    }

    return result;
}

void setup()
{
    // 初始化串口通信
    Serial.begin(115200);
    delay(1000);             // 短暂延迟确保串口初始化
    setCpuFrequencyMhz(240); // 设置CPU频率为240MHz
    pinMode(4, OUTPUT);      // GPIO4 控制闪光灯
    digitalWrite(4, LOW);    // 关闭闪光灯节省电力
    delay(2000);             // 增加延迟，等待电源稳定

    // 设置LED引脚
    pinMode(LED_RECORD_PIN, OUTPUT);   // LED输出
    digitalWrite(LED_RECORD_PIN, LOW); // 初始状态LED关闭

    // PSRAM初始化
    bool psram_initialized = false;

    if (!psramInit())
    {
        ESP_LOGE(TAG, "PSRAM初始化失败! 系统可能无法正常工作");
        delay(1000); // 延迟以便查看错误信息

        if (sysInit.force_psram_init())
        {
            ESP_LOGI(TAG, "PSRAM强制初始化成功");
            psram_initialized = true;
        }
        else
        {
            ESP_LOGE(TAG, "PSRAM强制初始化失败，系统将重启");
            delay(1000);
            ESP.restart(); // 重启设备以尝试重新初始化
            return;
        }
    }
    else
    {
        ESP_LOGI(TAG, "PSRAM初始化成功: %d bytes", ESP.getPsramSize());
        psram_initialized = true;
    }

    // 全局系统初始化 - 即使失败也继续执行
    bool init_success = sysInit.init() == ESP_OK;
    sdCardAvailable = sysInit.getSDCardStatus();
    if (sdCardAvailable)
    {
        // 创建必要的目录
        if (!SD_MMC.exists("/save"))
        {
            SD_MMC.mkdir("/save");
        }
        if (!SD_MMC.exists("/temp"))
        {
            SD_MMC.mkdir("/temp");
        }

        frameCacheMode = SD_CARD;
        ESP_LOGI(TAG, "使用SD卡模式存储帧");
    }
    else
    {
        frameCacheMode = RAM_ONLY;
        ESP_LOGI(TAG, "使用RAM模式存储帧");
        // 预分配RAM帧缓冲区
        ramFrameBuffer.reserve(RAM_FRAME_BUFFER_SIZE);
    }
    if (!init_success)
    {
        ESP_LOGW(TAG, "系统初始化部分失败，将尝试继续运行");
    }

    // 初始化SD卡
    if (!SD_MMC.begin("/sdcard", true))
    {
        Serial.println("SD Card Mount Failed");
        sdCardAvailable = false;
    }
    else
    {
        uint8_t cardType = SD_MMC.cardType();
        if (cardType == CARD_NONE)
        {
            Serial.println("No SD card attached");
            sdCardAvailable = false;
        }
        else
        {
            Serial.print("SD Card Size: ");
            Serial.print(SD_MMC.cardSize() / (1024 * 1024));
            Serial.println("MB");
            sdCardAvailable = true;
        }
    }

    if (sdCardAvailable)
    {
        if (!SD_MMC.exists("/save"))
            SD_MMC.mkdir("/save");
        if (!SD_MMC.exists("/temp"))
            SD_MMC.mkdir("/temp");
    }

    // 初始化帧缓冲区
    initializeFrameBuffer();

    // 创建并启动处理流
    bool processor_started = false;
    try
    {
        processor = std::make_unique<ProcessStream>();
        if (processor)
        {
            processor->setResultCallback(processResultCallback);
            processor->setAutoSaveHighRiskFrames(sdCardAvailable);

            size_t freePsram = ESP.getFreePsram();
            bool enableTradition = (freePsram < 3000000);
            processor->setUseTraditionalLaneDetection(enableTradition);
            processor_started = processor->start();
            if (!processor_started)
            {
                ESP_LOGW(TAG, "处理流启动失败: %s", processor->getLastError().c_str());
            }
        }
    }
    catch (const std::exception &e)
    {
        ESP_LOGE(TAG, "创建处理流时出错: %s", e.what());
        // 继续执行，但不使用处理流
    }

    // 为图像缓冲区分配内存 - 这是最关键的部分，必须成功
    bool buffer_allocated = false;
    for (int attempt = 0; attempt < 3 && !buffer_allocated; attempt++)
    {
        ESP_LOGI(TAG, "尝试分配图像缓冲区 (尝试 %d/3)", attempt + 1);

        if (ESP.getPsramSize() > 0)
        {
            // 尝试在PSRAM中分配
            ESP_LOGI(TAG, "尝试在PSRAM中分配 %d KB",
                     (MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS) / 1024);

            imageBuffer = (uint8_t *)ps_malloc(MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
        }

        if (!imageBuffer && ESP.getFreeHeap() >= MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS)
        {
            // 如果PSRAM分配失败但堆内存足够，尝试在堆上分配
            ESP_LOGW(TAG, "PSRAM分配失败，尝试在堆上分配");
            imageBuffer = (uint8_t *)malloc(MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
        }

        if (imageBuffer)
        {
            ESP_LOGI(TAG, "图像缓冲区分配成功，地址: %p", imageBuffer);
            // 初始化缓冲区内存为0
            memset(imageBuffer, 0, MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
            buffer_allocated = true;
        }
        else
        {
            // 分配失败，休息一下后重试
            ESP_LOGE(TAG, "内存分配失败，可用PSRAM: %d bytes, 堆: %d bytes",
                     ESP.getFreePsram(), ESP.getFreeHeap());
            delay(1000);
        }
    }

    if (!buffer_allocated)
    {
        // 如果所有尝试都失败，尝试分配一个小得多的缓冲区
        ESP_LOGE(TAG, "无法分配完整图像缓冲区，尝试最小化缓冲区");

        // 尝试分配小缓冲区 (160x120x3 = 约57KB)
        int small_width = 160;
        int small_height = 120;
        int small_channels = 3;
        size_t small_size = small_width * small_height * small_channels;

        imageBuffer = (uint8_t *)malloc(small_size);
        if (imageBuffer)
        {
            ESP_LOGW(TAG, "已分配小缓冲区: %d KB", small_size / 1024);
            memset(imageBuffer, 0, small_size);

            // 相机配置也需要调整以匹配
            sensor_t *s = esp_camera_sensor_get();
            if (s)
            {
                s->set_framesize(s, FRAMESIZE_QQVGA); // 设置为160x120
            }

            buffer_allocated = true;
        }
        else
        {
            ESP_LOGE(TAG, "所有内存分配尝试都失败，系统将重启");
            delay(3000);
            ESP.restart();
            return;
        }
    }

    // 初始化计时和其他变量
    frameCount = 0;
    startTime = millis();

    // 记录启动状态
    ESP_LOGI(TAG, "系统启动状态:");
    ESP_LOGI(TAG, "  PSRAM初始化: %s", psram_initialized ? "成功" : "失败");
    ESP_LOGI(TAG, "  系统初始化: %s", init_success ? "成功" : "部分失败");
    ESP_LOGI(TAG, "  处理流启动: %s", processor_started ? "成功" : "失败");
    ESP_LOGI(TAG, "  缓冲区分配: %s", buffer_allocated ? "成功" : "失败");

    ESP_LOGI(TAG, "开始主处理循环");
}

void loop()
{
    // 检查全局运行标志
    if (!g_running)
    {
        // 清理资源
        ESP_LOGI(TAG, "收到停止信号，清理资源");

        if (imageBuffer != nullptr)
        {
            ESP_LOGI(TAG, "释放图像缓冲区: %p", imageBuffer);
            free(imageBuffer); // 使用free替代delete[]，因为我们改用了malloc/ps_malloc
            imageBuffer = nullptr;
        }

        if (processor)
        {
            ESP_LOGI(TAG, "停止处理流");
            processor->stop();
            processor.reset();
        }

        ESP_LOGI(TAG, "程序退出，共处理 %d 帧", frameCount);

        // 进入空闲状态
        while (true)
        {
            delay(1000);
        }
    }

    // 确保录制状态
    ensureRecording();

    // 检查图像缓冲区是否可用
    if (imageBuffer == nullptr)
    {
        ESP_LOGE(TAG, "图像缓冲区为NULL，尝试重新分配");

        // 尝试重新分配
        imageBuffer = (uint8_t *)ps_malloc(MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
        if (!imageBuffer)
        {
            imageBuffer = (uint8_t *)malloc(MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
        }

        if (!imageBuffer)
        {
            ESP_LOGE(TAG, "重新分配失败，稍后重试");
            delay(1000);
            return; // 跳过这个循环迭代
        }

        // 初始化新分配的缓冲区
        memset(imageBuffer, 0, MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS);
        ESP_LOGI(TAG, "图像缓冲区重新分配成功: %p", imageBuffer);
    }

    unsigned long frameStart = millis();
    int width = 0, height = 0, channels = 0;
    unsigned long captureDelay = 33;
    if (processor)
    {
        size_t queueSize = processor->getRawQueueSize();
        if (queueSize > 5)
            captureDelay = 100;
        if (queueSize > 10)
            captureDelay = 150;
        if (queueSize > 15)
            captureDelay = 200;
    }
    delay(captureDelay);
    try
    {
        // 获取下一帧图像
        if (getNextFrame(imageBuffer, width, height, channels))
        {
            // 记录当前时间戳
            double timestamp = millis() / 1000.0;

            // 提交到处理流程 (如果处理流可用)
            if (processor)
            {
                processor->addFrame(imageBuffer, width, height, channels, timestamp);
            }

            frameCount++;
            // 计算帧率
            if (frameCount % 30 == 0)
            {
                unsigned long currentTime = millis();
                double elapsed = (currentTime - startTime) / 1000.0;

                double fps = frameCount / elapsed;
                ESP_LOGI(TAG, "已处理 %d 帧，当前帧率: %.2f FPS", frameCount, fps);

                // 检查队列状态
                ESP_LOGI(TAG, "队列状态 - 待处理: %zu，已完成: %zu",
                         processor->getRawQueueSize(),
                         processor->getProcessedQueueSize());

                // 输出录制状态
                if (isRecording)
                {
                    ESP_LOGI(TAG, "当前正在录制视频，保存到: %s",
                             processor->getCurrentVideoPath().c_str());
                }
            }

            // 控制处理速度（如果必要的话）
            unsigned long frameEnd = millis();
            unsigned long frameDuration = frameEnd - frameStart;

            if (frameDuration < 33)
            { // 33ms约等于30FPS
                delay(33 - frameDuration);
            }
        }
        else
        {
            ESP_LOGW(TAG, "获取帧失败，跳过");
            delay(10);
        }
    }
    catch (const std::exception &e)
    {
        ESP_LOGE(TAG, "处理过程中发生错误: %s", e.what());
        delay(1000); // 错误发生后等待一段时间
    }
}
