#include "process_stream.h"
#include <pthread.h>
#include <queue>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>

static const char *TAG = "ProcessStream";

// 简单的计时器类，用于性能测量
class Timer
{
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string operation_name;

public:
    Timer(const std::string &name) : operation_name(name)
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        ESP_LOGI(TAG, "%s took %lld ms", operation_name.c_str(), duration);
    }

    double elapsed()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        return static_cast<double>(duration) / 1000.0; // 转换为秒
    }
};

ProcessStream::ProcessStream()
{
    ESP_LOGI(TAG, "初始化处理流程");

    // 初始化临时目录
    ensureTempDir();

    // 创建对象检测器
    objectDetector = std::make_unique<ObjectDetect>();

    // 创建DeepSORT跟踪器
    tracker = std::make_unique<DeepSORT>();

    // 创建车道线检测器
    laneDetector = std::make_unique<LaneAreaDetect>();

    // 创建视频保存器
    videoSaver = std::make_unique<VideoSave>();

    ESP_LOGI(TAG, "所有处理模块初始化完成");
}

ProcessStream::~ProcessStream()
{
    // 停止所有线程
    if (running)
    {
        stop();
    }

    // 清理临时目录
    cleanupTempDir();
}

bool ProcessStream::start()
{
    if (running)
    {
        ESP_LOGW(TAG, "处理流程已经在运行");
        return false;
    }

    ESP_LOGI(TAG, "启动处理流程");
    running = true;

    try
    {
        // 清理临时目录
        cleanupTempDir();

        // 创建新的保存会话
        if (videoSaver)
        {
            videoSaver->startSession();
        }

        // 使用ESP-IDF任务API替代C++线程

        // 创建车道线检测任务 - 需要更多栈空间
        const uint32_t LANE_DETECTION_STACK = 8192; // 32KB
        ESP_LOGI(TAG, "创建车道线检测任务，栈大小: %d words (%d bytes)",
                 LANE_DETECTION_STACK, LANE_DETECTION_STACK * 4);

        BaseType_t lane_res = xTaskCreatePinnedToCore(
            [](void *arg)
            {
                ProcessStream *self = static_cast<ProcessStream *>(arg);
                self->laneDetectionWorker();
                vTaskDelete(NULL);
            },
            "lane_detect",        // 任务名称
            LANE_DETECTION_STACK, // 栈大小
            this,                 // 任务参数
            1,                    // 优先级
            &laneDetectionTask,   // 任务句柄
            1                     // 在核心1上运行
        );

        if (lane_res != pdPASS)
        {
            ESP_LOGE(TAG, "创建车道线检测任务失败: %d", lane_res);
            setLastError("创建车道线检测任务失败");
            running = false;
            return false;
        }

        // 创建目标检测任务
        const uint32_t OBJECT_DETECTION_STACK = 6144; // 24KB
        BaseType_t obj_res = xTaskCreatePinnedToCore(
            [](void *arg)
            {
                ProcessStream *self = static_cast<ProcessStream *>(arg);
                self->objectDetectionWorker();
                vTaskDelete(NULL);
            },
            "obj_detect",           // 任务名称
            OBJECT_DETECTION_STACK, // 栈大小
            this,                   // 任务参数
            1,                      // 优先级
            &objectDetectionTask,   // 任务句柄
            1                       // 在核心1上运行
        );

        if (obj_res != pdPASS)
        {
            ESP_LOGE(TAG, "创建目标检测任务失败: %d", obj_res);
            // 清理已创建的任务
            if (laneDetectionTask)
            {
                vTaskDelete(laneDetectionTask);
                laneDetectionTask = NULL;
            }
            setLastError("创建目标检测任务失败");
            running = false;
            return false;
        }

        // 创建跟踪任务
        const uint32_t TRACKING_STACK = 4096; // 16KB
        BaseType_t track_res = xTaskCreatePinnedToCore(
            [](void *arg)
            {
                ProcessStream *self = static_cast<ProcessStream *>(arg);
                self->trackingWorker();
                vTaskDelete(NULL);
            },
            "tracking",     // 任务名称
            TRACKING_STACK, // 栈大小
            this,           // 任务参数
            1,              // 优先级
            &trackingTask,  // 任务句柄
            1               // 在核心1上运行
        );

        if (track_res != pdPASS)
        {
            ESP_LOGE(TAG, "创建跟踪任务失败: %d", track_res);
            // 清理已创建的任务
            if (laneDetectionTask)
                vTaskDelete(laneDetectionTask);
            if (objectDetectionTask)
                vTaskDelete(objectDetectionTask);
            setLastError("创建跟踪任务失败");
            running = false;
            return false;
        }

        // 创建TTC计算任务
        const uint32_t TTC_STACK = 4096; // 16KB
        BaseType_t ttc_res = xTaskCreatePinnedToCore(
            [](void *arg)
            {
                ProcessStream *self = static_cast<ProcessStream *>(arg);
                self->ttcCalculationWorker();
                vTaskDelete(NULL);
            },
            "ttc_calc",          // 任务名称
            TTC_STACK,           // 栈大小
            this,                // 任务参数
            1,                   // 优先级
            &ttcCalculationTask, // 任务句柄
            1                    // 在核心1上运行
        );

        if (ttc_res != pdPASS)
        {
            ESP_LOGE(TAG, "创建TTC计算任务失败: %d", ttc_res);
            // 清理已创建的任务
            if (laneDetectionTask)
                vTaskDelete(laneDetectionTask);
            if (objectDetectionTask)
                vTaskDelete(objectDetectionTask);
            if (trackingTask)
                vTaskDelete(trackingTask);
            setLastError("创建TTC计算任务失败");
            running = false;
            return false;
        }

        // 创建结果处理任务
        const uint32_t RESULT_STACK = 4096; // 16KB
        BaseType_t result_res = xTaskCreatePinnedToCore(
            [](void *arg)
            {
                ProcessStream *self = static_cast<ProcessStream *>(arg);
                self->resultProcessingWorker();
                vTaskDelete(NULL);
            },
            "result_proc",         // 任务名称
            RESULT_STACK,          // 栈大小
            this,                  // 任务参数
            1,                     // 优先级
            &resultProcessingTask, // 任务句柄
            1                      // 在核心1上运行
        );

        if (result_res != pdPASS)
        {
            ESP_LOGE(TAG, "创建结果处理任务失败: %d", result_res);
            // 清理已创建的任务
            if (laneDetectionTask)
                vTaskDelete(laneDetectionTask);
            if (objectDetectionTask)
                vTaskDelete(objectDetectionTask);
            if (trackingTask)
                vTaskDelete(trackingTask);
            if (ttcCalculationTask)
                vTaskDelete(ttcCalculationTask);
            setLastError("创建结果处理任务失败");
            running = false;
            return false;
        }

        ESP_LOGI(TAG, "所有工作任务已启动");
        return true;
    }
    catch (const std::exception &e)
    {
        setLastError(std::string("启动任务失败: ") + e.what());
        ESP_LOGE(TAG, "启动任务失败: %s", e.what());
        running = false;
        return false;
    }
}

void ProcessStream::stop()
{
    if (!running)
    {
        ESP_LOGW(TAG, "处理流程已经停止");
        return;
    }

    ESP_LOGI(TAG, "停止处理流程");
    running = false;

    // 向所有队列发送空帧信号告知线程退出
    rawFrameQueue.push(nullptr);
    laneDetectedQueue.push(nullptr);
    objectDetectedQueue.push(nullptr);
    trackedQueue.push(nullptr);
    ttcProcessedQueue.push(nullptr);

    // 等待任务自行退出（最长等待1秒）
    vTaskDelay(pdMS_TO_TICKS(1000));

    // 然后强制删除任务
    if (laneDetectionTask != NULL)
    {
        vTaskDelete(laneDetectionTask);
        laneDetectionTask = NULL;
    }

    if (objectDetectionTask != NULL)
    {
        vTaskDelete(objectDetectionTask);
        objectDetectionTask = NULL;
    }

    if (trackingTask != NULL)
    {
        vTaskDelete(trackingTask);
        trackingTask = NULL;
    }

    if (ttcCalculationTask != NULL)
    {
        vTaskDelete(ttcCalculationTask);
        ttcCalculationTask = NULL;
    }

    if (resultProcessingTask != NULL)
    {
        vTaskDelete(resultProcessingTask);
        resultProcessingTask = NULL;
    }

    // 如果在录制，停止录制 - 使用与stopVideoRecording()相同的锁机制
    {
        std::lock_guard<std::mutex> lock(videoMutex);
        if (videoRecordingEnabled && videoSaver)
        {
            videoSaver->finishVideoCapture();
            videoRecordingEnabled = false;
            ESP_LOGI(TAG, "视频录制已停止，文件保存在: %s",
                     videoSaver->getCurrentVideoPath().c_str());
        }
    }

    // 清理所有队列
    clearAllQueues();

    // 清理临时文件
    cleanupTempDir();

    ESP_LOGI(TAG, "处理流程已停止");
}

bool ProcessStream::addFrame(uint8_t *imageData, int width, int height, int channels, double timestamp)
{
    if (!running)
    {
        ESP_LOGW(TAG, "处理流程已停止，帧将被丢弃");
        return false;
    }

    // 检查队列大小，如果过大则丢弃最旧的帧
    if (rawFrameQueue.size() > 5)
    {
        ESP_LOGW(TAG, "原始帧队列已满，丢弃最旧的帧");
        std::shared_ptr<FrameData> oldFrame;
        if (rawFrameQueue.try_pop(oldFrame))
        {
            ESP_LOGD(TAG, "丢弃帧 #%d", oldFrame->frameId);
        }
    }

    try
    {
        // 确保临时目录存在
        if (!ensureTempDir())
        {
            throw std::runtime_error("无法确保临时目录存在");
        }

        // 创建一个唯一的临时文件名
        static int frameCounter = 0;
        char tempFilePath[64];
        snprintf(tempFilePath, sizeof(tempFilePath), "%s/frame_%d.dat",
                 tempDir.c_str(), frameCounter);

        // 将帧数据写入SD卡
        FILE *fp = fopen(tempFilePath, "wb");
        if (!fp)
        {
            ESP_LOGE(TAG, "无法创建临时文件: %s (%s)", tempFilePath, strerror(errno));
            throw std::runtime_error("文件创建失败");
        }

        // 写入帧元数据
        if (fwrite(&width, sizeof(width), 1, fp) != 1 ||
            fwrite(&height, sizeof(height), 1, fp) != 1 ||
            fwrite(&channels, sizeof(channels), 1, fp) != 1)
        {
            fclose(fp);
            remove(tempFilePath);
            ESP_LOGE(TAG, "写入帧元数据失败");
            throw std::runtime_error("写入元数据失败");
        }

        // 写入图像数据
        size_t dataSize = width * height * channels;
        if (fwrite(imageData, 1, dataSize, fp) != dataSize)
        {
            fclose(fp);
            remove(tempFilePath);
            ESP_LOGE(TAG, "写入图像数据失败");
            throw std::runtime_error("写入图像数据失败");
        }

        fclose(fp);

        // 创建帧数据对象，但不复制图像数据
        std::shared_ptr<FrameData> frameData = std::make_shared<FrameData>();
        frameData->frameId = frameCounter++;
        frameData->timestamp = timestamp;
        frameData->width = width;
        frameData->height = height;
        frameData->channels = channels;
        frameData->imageData = nullptr;         // 不保存在内存中
        frameData->tempFilePath = tempFilePath; // 保存文件路径
        frameData->dataOnSD = true;

        ESP_LOGD(TAG, "添加新帧 #%d 到处理队列 (数据保存在: %s)",
                 frameData->frameId, tempFilePath);
        rawFrameQueue.push(frameData);
    }
    catch (const std::exception &e)
    {
        ESP_LOGE(TAG, "添加帧失败: %s", e.what());
    }
    return true;
}

void ProcessStream::laneDetectionWorker()
{
    ESP_LOGI(TAG, "车道线检测工作线程开始");

    while (true)
    {
        std::shared_ptr<FrameData> frameData;
        rawFrameQueue.wait_and_pop(frameData);

        // 检查是否应该退出
        if (!running || !frameData)
        {
            if (!running)
            {
                ESP_LOGI(TAG, "车道线检测工作线程退出");
            }
            break;
        }

        // 执行车道线检测
        try
        {
            Timer timer("车道线检测");

            // 如果数据在SD卡上，先加载到内存
            if (frameData->dataOnSD)
            {
                ESP_LOGD(TAG, "从SD卡加载帧 #%d 数据", frameData->frameId);
                if (!frameData->loadImageFromSD())
                {
                    ESP_LOGE(TAG, "无法从SD卡加载帧 #%d 数据", frameData->frameId);
                    laneDetectedQueue.push(frameData); // 即使失败也推送到下一个队列
                    continue;
                }
            }

            // 确保图像数据可用
            if (!frameData->imageData)
            {
                ESP_LOGE(TAG, "帧 #%d 无图像数据", frameData->frameId);
                laneDetectedQueue.push(frameData);
                continue;
            }

            // 使用LaneAreaDetect模型处理原始图像数据
            frameData->laneInfo = laneDetector->detect(
                frameData->imageData,
                frameData->width,
                frameData->height,
                frameData->channels);

            // 记录性能指标
            {
                std::lock_guard<std::mutex> lock(metricsMutex);
                updateMetric(performanceMetrics.laneDetectionTime, timer.elapsed());
            }

            // 修改：不再在此阶段删除图像数据或SD文件
            // 因为其他流程仍然需要原始图像数据

            // 将处理后的帧推送到下一个队列
            laneDetectedQueue.push(frameData);
            ESP_LOGD(TAG, "帧 #%d 完成车道线检测", frameData->frameId);
        }
        catch (const std::exception &e)
        {
            setLastError(std::string("车道线检测错误: ") + e.what());
            ESP_LOGE(TAG, "帧 #%d 车道线检测错误: %s", frameData->frameId, e.what());

            // 即使发生错误，也将帧推送到下一步
            laneDetectedQueue.push(frameData);
        }
    }
}

void ProcessStream::objectDetectionWorker()
{
    ESP_LOGI(TAG, "物体检测工作线程开始");

    while (true)
    {
        std::shared_ptr<FrameData> frameData;
        laneDetectedQueue.wait_and_pop(frameData);

        // 检查是否应该退出
        if (!running || !frameData)
        {
            if (!running)
            {
                ESP_LOGI(TAG, "物体检测工作线程退出");
            }
            break;
        }

        // 执行物体检测
        try
        {
            Timer timer("物体检测");

            // 如果数据在SD卡上但内存中没有，先加载到内存
            if (frameData->dataOnSD && !frameData->imageData)
            {
                ESP_LOGD(TAG, "从SD卡加载帧 #%d 数据用于物体检测", frameData->frameId);
                if (!frameData->loadImageFromSD())
                {
                    ESP_LOGE(TAG, "无法从SD卡加载帧 #%d 数据用于物体检测", frameData->frameId);
                    objectDetectedQueue.push(frameData); // 即使失败也推送到下一个队列
                    continue;
                }
            }

            // 确保图像数据可用
            if (!frameData->imageData)
            {
                ESP_LOGE(TAG, "帧 #%d 无图像数据用于物体检测", frameData->frameId);
                objectDetectedQueue.push(frameData);
                continue;
            }

            // 使用ObjectDetect模型处理图像数据
            frameData->detections = objectDetector->detect(
                frameData->imageData,
                frameData->width,
                frameData->height,
                frameData->channels);

            // 记录性能指标
            {
                std::lock_guard<std::mutex> lock(metricsMutex);
                updateMetric(performanceMetrics.objectDetectionTime, timer.elapsed());
            }

            // 修改：不删除图像数据，跟踪阶段还需要它

            // 将处理后的帧推送到下一个队列
            objectDetectedQueue.push(frameData);
            ESP_LOGD(TAG, "帧 #%d 完成物体检测，检测到 %zu 个目标",
                     frameData->frameId, frameData->detections.size());
        }
        catch (const std::exception &e)
        {
            setLastError(std::string("物体检测错误: ") + e.what());
            ESP_LOGE(TAG, "帧 #%d 物体检测错误: %s", frameData->frameId, e.what());

            // 即使发生错误，也将帧推送到下一步
            objectDetectedQueue.push(frameData);
        }
    }
}

void ProcessStream::trackingWorker()
{
    ESP_LOGI(TAG, "目标跟踪工作线程开始");

    while (true)
    {
        std::shared_ptr<FrameData> frameData;
        objectDetectedQueue.wait_and_pop(frameData);

        // 检查是否应该退出
        if (!running || !frameData)
        {
            if (!running)
            {
                ESP_LOGI(TAG, "目标跟踪工作线程退出");
            }
            break;
        }

        // 执行目标跟踪
        try
        {
            Timer timer("目标跟踪");

            // 如果数据在SD卡上且未加载，先加载到内存
            if (frameData->dataOnSD && !frameData->imageData)
            {
                ESP_LOGD(TAG, "从SD卡加载帧 #%d 数据用于目标跟踪", frameData->frameId);
                if (!frameData->loadImageFromSD())
                {
                    ESP_LOGE(TAG, "无法从SD卡加载帧 #%d 数据用于目标跟踪", frameData->frameId);
                    trackedQueue.push(frameData); // 即使失败也推送到下一个队列
                    continue;
                }
            }

            // 确保图像数据可用（如果有检测结果）
            if (!frameData->imageData && !frameData->detections.empty())
            {
                ESP_LOGE(TAG, "帧 #%d 无图像数据用于目标跟踪", frameData->frameId);
                trackedQueue.push(frameData);
                continue;
            }

            // 使用DeepSORT处理检测结果
            if (!frameData->detections.empty() && frameData->imageData)
            {
                frameData->tracks = tracker->update(
                    frameData->detections,
                    frameData->imageData,
                    frameData->width,
                    frameData->height);
            }

            // 记录性能指标
            {
                std::lock_guard<std::mutex> lock(metricsMutex);
                updateMetric(performanceMetrics.trackingTime, timer.elapsed());
            }

            // 修改：仍然保留图像数据，TTC计算完成后再删除

            // 将处理后的帧推送到下一个队列
            trackedQueue.push(frameData);
            ESP_LOGD(TAG, "帧 #%d 完成目标跟踪，跟踪 %zu 个目标",
                     frameData->frameId, frameData->tracks.size());
        }
        catch (const std::exception &e)
        {
            setLastError(std::string("目标跟踪错误: ") + e.what());
            ESP_LOGE(TAG, "帧 #%d 目标跟踪错误: %s", frameData->frameId, e.what());

            // 即使发生错误，也将帧推送到下一步
            trackedQueue.push(frameData);
        }
    }
}

void ProcessStream::ttcCalculationWorker()
{
    ESP_LOGI(TAG, "TTC计算工作线程开始");

    while (true)
    {
        std::shared_ptr<FrameData> frameData;
        trackedQueue.wait_and_pop(frameData);

        // 检查是否应该退出
        if (!running || !frameData)
        {
            if (!running)
            {
                ESP_LOGI(TAG, "TTC计算工作线程退出");
            }
            break;
        }

        try
        {
            Timer timer("TTC计算");

            // TTC计算需要跟踪结果和车道线信息，但不直接需要原始图像
            if (frameData->tracks.empty() || frameData->laneInfo.empty())
            {
                ESP_LOGW(TAG, "帧 #%d 缺少跟踪结果或车道线信息，跳过TTC计算",
                         frameData->frameId);
                ttcProcessedQueue.push(frameData);
                continue;
            }

            // 使用TTC计算器计算碰撞风险
            frameData->ttcResults = ttcCalculator->calculate(
                frameData->tracks,
                frameData->laneInfo);

            // 记录警告信息
            for (const auto &ttc : frameData->ttcResults)
            {
                if (ttc.risk > 0.8f)
                {
                    ESP_LOGW(TAG, "危险警告：极高碰撞风险！目标ID: %d，TTC: %.2f秒，风险值: %.2f",
                             ttc.trackId, ttc.ttc, ttc.risk);
                }
                else if (ttc.risk > 0.5f)
                {
                    ESP_LOGW(TAG, "中等警告：潜在碰撞风险。目标ID: %d，TTC: %.2f秒，风险值: %.2f",
                             ttc.trackId, ttc.ttc, ttc.risk);
                }
            }

            // 记录性能指标
            {
                std::lock_guard<std::mutex> lock(metricsMutex);
                updateMetric(performanceMetrics.ttcTime, timer.elapsed());
                performanceMetrics.frameCount++;

                // 计算总处理时间（从帧添加到处理完成）
                if (frameData->timestamp > 0)
                {
                    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::high_resolution_clock::now().time_since_epoch())
                                               .count() /
                                           1000.0 -
                                       frameData->timestamp;

                    updateMetric(performanceMetrics.totalFrameTime, totalTime);
                }
            }

            // 现在所有计算阶段都完成了，可以释放图像数据，但是需要考虑最终的结果处理阶段可能还需要图像
            // 我们在此阶段只做评估，决定是否需要保留图像数据用于结果处理

            // 检查是否有高风险情况需要保存，以决定是否保留图像数据
            bool hasHighRisk = false;
            for (const auto &ttc : frameData->ttcResults)
            {
                if (ttc.risk > 0.8f)
                {
                    hasHighRisk = true;
                    break;
                }
            }

            // 如果没有录制视频且没有高风险情况，可以安全释放图像数据
            if (!videoRecordingEnabled && !autoSaveHighRiskFrames && !hasHighRisk)
            {
                // 现在可以释放图像数据和删除SD卡文件
                if (frameData->needToFreeImageData && frameData->imageData)
                {
                    free(frameData->imageData);
                    frameData->imageData = nullptr;
                    frameData->needToFreeImageData = false;
                }
                frameData->deleteSDFile();

                ESP_LOGD(TAG, "帧 #%d 无需保存，释放图像数据", frameData->frameId);
            }

            ttcProcessedQueue.push(frameData);
            ESP_LOGD(TAG, "帧 #%d 完成TTC计算，发现 %zu 个风险目标",
                     frameData->frameId, frameData->ttcResults.size());
        }
        catch (const std::exception &e)
        {
            setLastError(std::string("TTC计算错误: ") + e.what());
            ESP_LOGE(TAG, "帧 #%d TTC计算错误: %s", frameData->frameId, e.what());

            // 即使发生错误，也将帧推送到下一步
            ttcProcessedQueue.push(frameData);
        }
    }
}

void ProcessStream::resultProcessingWorker()
{
    static int frameSkipCounter = 0;
    ESP_LOGI(TAG, "结果处理工作线程开始");

    while (true)
    {
        std::shared_ptr<FrameData> frameData;
        ttcProcessedQueue.wait_and_pop(frameData);

        // 检查是否应该退出
        if (!running || !frameData)
        {
            if (!running)
            {
                ESP_LOGI(TAG, "结果处理工作线程退出");
            }
            break;
        }

        // 处理最终结果
        try
        {
            // 如果需要保存视频或帧，需要确保有图像数据
            bool needImageForSaving = false;

            // 检查是否有高风险情况需要保存
            bool hasHighRisk = false;
            for (const auto &ttc : frameData->ttcResults)
            {
                if (ttc.risk > 0.8f)
                {
                    hasHighRisk = true;
                    break;
                }
            }

            // 判断是否需要图像数据
            if (videoSaver &&
                ((videoRecordingEnabled && frameSkipCounter % 3 == 0) ||
                 (autoSaveHighRiskFrames && hasHighRisk)))
            {
                needImageForSaving = true;
            }

            // 如果需要图像数据但当前没有，则从SD卡加载
            if (needImageForSaving && !frameData->imageData && !frameData->tempFilePath.empty())
            {
                ESP_LOGD(TAG, "从SD卡加载帧 #%d 数据用于视频保存", frameData->frameId);
                if (!frameData->loadImageFromSD())
                {
                    ESP_LOGE(TAG, "无法从SD卡加载帧 #%d 数据用于视频保存", frameData->frameId);
                    needImageForSaving = false;
                }
            }

            if (videoSaver && frameData)
            {
                std::lock_guard<std::mutex> lock(videoMutex);

                if (videoRecordingEnabled)
                {
                    frameSkipCounter++;
                    if ((frameSkipCounter - 1) % 3 == 0 && needImageForSaving && frameData->imageData)
                    {
                        videoSaver->addFrameToVideo(frameData, true);
                    }
                }
                else if (autoSaveHighRiskFrames && hasHighRisk && needImageForSaving && frameData->imageData)
                {
                    ESP_LOGI(TAG, "检测到高风险情况，自动保存帧 #%d", frameData->frameId);
                    videoSaver->saveFrame(frameData, true);
                }
            }

            // 如果设置了回调函数，调用它
            if (resultCallback)
            {
                resultCallback(frameData);
            }

            // 最后，确保清理图像资源和临时文件
            if (frameData->needToFreeImageData && frameData->imageData)
            {
                free(frameData->imageData);
                frameData->imageData = nullptr;
                frameData->needToFreeImageData = false;
            }

            frameData->deleteSDFile();

            ESP_LOGD(TAG, "帧 #%d 处理完成", frameData->frameId);
        }
        catch (const std::exception &e)
        {
            setLastError(std::string("结果处理错误: ") + e.what());
            ESP_LOGE(TAG, "帧 #%d 结果处理错误: %s", frameData->frameId, e.what());

            // 确保清理资源
            if (frameData->needToFreeImageData && frameData->imageData)
            {
                free(frameData->imageData);
                frameData->imageData = nullptr;
                frameData->needToFreeImageData = false;
            }
            frameData->deleteSDFile();
        }

        // frameData将在此处被释放（如果没有其他引用）
    }
}

void ProcessStream::setLastError(const std::string &error)
{
    std::lock_guard<std::mutex> lock(errorMutex);
    lastError = error;
}

std::string ProcessStream::getLastError()
{
    std::lock_guard<std::mutex> lock(errorMutex);
    return lastError;
}
void ProcessStream::setResultCallback(ProcessResultCallback callback)
{
    std::lock_guard<std::mutex> lock(errorMutex); // 使用现有的互斥锁保护
    resultCallback = callback;
    ESP_LOGI(TAG, "已设置结果回调函数");
}
void ProcessStream::clearAllQueues()
{
    rawFrameQueue.clear();
    laneDetectedQueue.clear();
    objectDetectedQueue.clear();
    trackedQueue.clear();
    ttcProcessedQueue.clear();
    ESP_LOGI(TAG, "所有处理队列已清空");
}

void ProcessStream::updateMetric(double &metric, double value)
{
    // 指数移动平均更新
    const double alpha = 0.2;
    if (metric == 0)
    {
        metric = value;
    }
    else
    {
        metric = alpha * value + (1 - alpha) * metric;
    }
}

void ProcessStream::logPerformanceMetrics()
{
    std::lock_guard<std::mutex> lock(metricsMutex);

    if (performanceMetrics.frameCount == 0)
    {
        ESP_LOGI(TAG, "无性能数据可报告");
        return;
    }

    ESP_LOGI(TAG, "性能指标 (处理 %d 帧):", performanceMetrics.frameCount);
    ESP_LOGI(TAG, "  平均车道线检测时间: %.2f 秒", performanceMetrics.laneDetectionTime);
    ESP_LOGI(TAG, "  平均物体检测时间: %.2f 秒", performanceMetrics.objectDetectionTime);
    ESP_LOGI(TAG, "  平均目标跟踪时间: %.2f 秒", performanceMetrics.trackingTime);
    ESP_LOGI(TAG, "  平均TTC计算时间: %.2f 秒", performanceMetrics.ttcTime);
    ESP_LOGI(TAG, "  平均总处理时间: %.2f 秒", performanceMetrics.totalFrameTime);

    double totalStepTime = performanceMetrics.laneDetectionTime +
                           performanceMetrics.objectDetectionTime +
                           performanceMetrics.trackingTime +
                           performanceMetrics.ttcTime;

    ESP_LOGI(TAG, "  估计每秒处理帧数: %.2f FPS", 1.0 / performanceMetrics.totalFrameTime);
    ESP_LOGI(TAG, "  流水线效率: %.2f%%", 100.0 * totalStepTime / performanceMetrics.totalFrameTime);
}

double ProcessStream::getAverageProcessingTime() const
{
    std::lock_guard<std::mutex> lock(metricsMutex);
    return performanceMetrics.totalFrameTime;
}

// 视频录制
bool ProcessStream::startVideoRecording(int width, int height, int fps)
{
    const size_t WRITE_BUFFER_SIZE = 32 * 1024; // 32KB

    std::lock_guard<std::mutex> lock(videoMutex);

    if (!videoSaver)
    {
        setLastError("视频保存模块未初始化");
        return false;
    }

    if (videoRecordingEnabled)
    {
        ESP_LOGW(TAG, "视频已经在录制中");
        return true;
    }

    recordingFPS = fps;
    bool result = videoSaver->startVideoCapture(width, height, fps);
    if (result)
    {
        videoRecordingEnabled = true;
        ESP_LOGI(TAG, "开始视频录制，分辨率: %dx%d, FPS: %d", width, height, fps);
    }
    else
    {
        setLastError("无法启动视频录制");
        ESP_LOGE(TAG, "无法启动视频录制");
    }

    return result;
}

bool ProcessStream::stopVideoRecording()
{
    std::lock_guard<std::mutex> lock(videoMutex);

    if (!videoSaver || !videoRecordingEnabled)
    {
        ESP_LOGW(TAG, "没有正在进行的视频录制");
        return false;
    }

    bool result = videoSaver->finishVideoCapture();
    videoRecordingEnabled = false;

    ESP_LOGI(TAG, "视频录制已停止，文件保存在: %s",
             videoSaver->getCurrentVideoPath().c_str());

    return result;
}

void ProcessStream::setAutoSaveHighRiskFrames(bool enable)
{
    std::lock_guard<std::mutex> lock(videoMutex);
    autoSaveHighRiskFrames = enable;
    ESP_LOGI(TAG, "自动保存高风险帧: %s", enable ? "已启用" : "已禁用");
}

std::string ProcessStream::getCurrentVideoPath() const
{
    if (videoSaver)
    {
        return videoSaver->getCurrentVideoPath();
    }
    return "";
}

std::string ProcessStream::getCurrentSessionDir() const
{
    if (videoSaver)
    {
        return videoSaver->getCurrentSessionDir();
    }
    return "";
}

// 确保临时目录存在
bool ProcessStream::ensureTempDir()
{
    if (tempDir.empty())
    {
        tempDir = "/sdcard/temp";
    }

    // 检查目录是否存在
    struct stat st;
    if (stat(tempDir.c_str(), &st) == 0)
    {
        if (S_ISDIR(st.st_mode))
        {
            return true; // 目录已存在
        }
        else
        {
            ESP_LOGE(TAG, "路径 %s 存在但不是目录", tempDir.c_str());
            return false;
        }
    }

    // 创建目录
    ESP_LOGI(TAG, "创建临时目录: %s", tempDir.c_str());
    if (mkdir(tempDir.c_str(), 0755) != 0)
    {
        ESP_LOGE(TAG, "创建目录 %s 失败: %s", tempDir.c_str(), strerror(errno));
        return false;
    }

    return true;
}

// 清理临时目录中的所有文件
void ProcessStream::cleanupTempDir()
{
    if (tempDir.empty())
        return;

    ESP_LOGI(TAG, "清理临时目录: %s", tempDir.c_str());

    DIR *dir = opendir(tempDir.c_str());
    if (!dir)
    {
        ESP_LOGE(TAG, "无法打开目录: %s", tempDir.c_str());
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG)
        { // 常规文件
            std::string filePath = tempDir + "/" + entry->d_name;
            ESP_LOGD(TAG, "删除临时文件: %s", filePath.c_str());
            remove(filePath.c_str());
        }
    }

    closedir(dir);
}
bool ProcessStream::addFrameDirect(uint8_t *buffer, int width, int height, int channels, double timestamp, int frameId)
{
    if (!running)
    {
        ESP_LOGW(TAG, "处理流程已停止，帧将被丢弃");
        return false;
    }

    // 检查队列大小，如果过大则丢弃最旧的帧
    if (rawFrameQueue.size() > 5)
    {
        ESP_LOGW(TAG, "原始帧队列已满，丢弃最旧的帧");
        std::shared_ptr<FrameData> oldFrame;
        if (rawFrameQueue.try_pop(oldFrame))
        {
            ESP_LOGD(TAG, "丢弃帧 #%d", oldFrame->frameId);
        }
    }

    try
    {
        // 创建帧数据对象，直接使用内存中的图像数据
        std::shared_ptr<FrameData> frameData = std::make_shared<FrameData>();
        frameData->frameId = frameId;
        frameData->timestamp = timestamp;
        frameData->width = width;
        frameData->height = height;
        frameData->channels = channels;

        // 分配内存并复制图像数据
        size_t dataSize = width * height * channels;
        frameData->imageData = (uint8_t *)malloc(dataSize);

        if (!frameData->imageData)
        {
            ESP_LOGE(TAG, "无法为帧 #%d 分配内存", frameId);
            return false;
        }

        // 复制图像数据
        memcpy(frameData->imageData, buffer, dataSize);
        frameData->needToFreeImageData = true; // 标记需要释放内存
        frameData->dataOnSD = false;           // 数据不在SD卡上

        ESP_LOGD(TAG, "直接添加新帧 #%d 到处理队列 (数据保存在内存中)", frameData->frameId);
        rawFrameQueue.push(frameData);
        return true;
    }
    catch (const std::exception &e)
    {
        ESP_LOGE(TAG, "直接添加帧失败: %s", e.what());
        return false;
    }
}


void ProcessStream::setUseTraditionalLaneDetection(bool useTraditional)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_useTraditionalLaneDetection = useTraditional;

    // 同时更新车道线检测器的设置，确保laneDetector存在
    if (laneDetector)
    {
        laneDetector->setUseTraditionalMethod(useTraditional);
    }

    ESP_LOGI(TAG, "设置传统车道线检测: %s", useTraditional ? "启用" : "禁用");
}
