#ifndef __PROCESS_STREAM_H__
#define __PROCESS_STREAM_H__
#include "esp_all.h"
#include "config.h"
#include "esp_tflite.h"
#include "common_types.h"

#include "lane_area_detect.h"
#include "object_detect.h"
#include "video_save.h"
#include "deep_sort.h"
#include "ttc.h"
#include <pthread.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <functional>
// 处理结果回调函数类型
typedef std::function<void(const std::shared_ptr<FrameData> &)> ProcessResultCallback;
class ProcessStream
{

private:
    // 线程安全的队列实现
    template <typename T>
    class ThreadSafeQueue
    {
    private:
        std::queue<T> queue;
        mutable std::mutex mutex;
        std::condition_variable cond;
        std::atomic<bool> done{false};

    public:
        void push(T item)
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(std::move(item));
            cond.notify_one();
        }

        bool try_pop(T &item)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (queue.empty())
            {
                return false;
            }
            item = std::move(queue.front());
            queue.pop();
            return true;
        }

        void wait_and_pop(T &item)
        {
            std::unique_lock<std::mutex> lock(mutex);
            cond.wait(lock, [this]
                      { return !queue.empty() || done; });
            if (done && queue.empty())
            {
                return;
            }
            item = std::move(queue.front());
            queue.pop();
        }

        bool empty() const
        {
            std::lock_guard<std::mutex> lock(mutex);
            return queue.empty();
        }

        void set_done()
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                done = true;
            }
            cond.notify_all();
        }

        bool is_done() const
        {
            return done;
        }

        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex);
            return queue.size();
        }

        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex);
            std::queue<T> empty;
            std::swap(queue, empty);
        }
    };

    // 各个处理阶段的队列
    ThreadSafeQueue<std::shared_ptr<FrameData>> rawFrameQueue;       // 原始帧
    ThreadSafeQueue<std::shared_ptr<FrameData>> laneDetectedQueue;   // 车道线检测完成
    ThreadSafeQueue<std::shared_ptr<FrameData>> objectDetectedQueue; // 物体检测完成
    ThreadSafeQueue<std::shared_ptr<FrameData>> trackedQueue;        // 目标跟踪完成
    ThreadSafeQueue<std::shared_ptr<FrameData>> ttcProcessedQueue;   // TTC计算完成

    // 处理模块实例
    std::unique_ptr<LaneAreaDetect> laneDetector;
    std::unique_ptr<ObjectDetect> objectDetector;
    std::unique_ptr<DeepSORT> tracker;
    std::unique_ptr<TTC> ttcCalculator;

    // 处理线程
    std::vector<pthread_t> threadHandles;
    std::atomic<bool> running{false};

    // 回调函数
    ProcessResultCallback resultCallback;

    // 线程错误处理
    std::mutex errorMutex;
    std::string lastError;

    // 性能监控
    struct PerformanceMetrics
    {
        double totalFrameTime;
        double laneDetectionTime;
        double objectDetectionTime;
        double trackingTime;
        double ttcTime;
        int frameCount;

        PerformanceMetrics() : totalFrameTime(0), laneDetectionTime(0),
                               objectDetectionTime(0), trackingTime(0), ttcTime(0),
                               frameCount(0) {}
    };
    PerformanceMetrics performanceMetrics;
    mutable std::mutex metricsMutex;

    // 处理函数
    void laneDetectionWorker();
    void objectDetectionWorker();
    void trackingWorker();
    void ttcCalculationWorker();
    void resultProcessingWorker();

    // 记录错误
    void setLastError(const std::string &error);

    // 性能测量相关方法
    void updateMetric(double &metric, double value);
    void logPerformanceMetrics();

    // 保存视频
    std::unique_ptr<VideoSave> videoSaver;
    bool videoRecordingEnabled{true};
    bool autoSaveHighRiskFrames{true};
    int recordingFPS{30};
    mutable std::mutex videoMutex;

public:
    ProcessStream();
    ~ProcessStream();

    // 启动和停止处理流程
    bool start();
    void stop();
    // 添加新帧到处理队列
    bool addFrame(uint8_t *imageData, int width, int height, int channels, double timestamp);
    bool addFrameDirect(uint8_t *imageData, int width, int height, int channels, double timestamp, int frameId);

    // 设置结果回调
    void setResultCallback(ProcessResultCallback callback);
    // 获取队列状态
    size_t getRawQueueSize() const { return rawFrameQueue.size(); }
    size_t getProcessedQueueSize() const { return ttcProcessedQueue.size(); }
    // 获取最近错误
    std::string getLastError();
    // 重置所有队列
    void clearAllQueues();
    // 获取性能指标
    double getAverageProcessingTime() const;
    // 视频录制
    bool startVideoRecording(int width, int height, int fps = 30);
    bool stopVideoRecording();
    void setAutoSaveHighRiskFrames(bool enable);
    std::string getCurrentVideoPath() const;
    std::string getCurrentSessionDir() const;
    void setUseTraditionalLaneDetection(bool useTraditional);

private:
    // SD卡临时目录管理
    std::string tempDir;
    bool ensureTempDir();
    void cleanupTempDir();
    bool m_useTraditionalLaneDetection = true; // 默认使用传统车道线检测
    mutable std::mutex m_mutex;
    // 任务句柄
    TaskHandle_t laneDetectionTask = NULL;
    TaskHandle_t objectDetectionTask = NULL;
    TaskHandle_t trackingTask = NULL;
    TaskHandle_t ttcCalculationTask = NULL;
    TaskHandle_t resultProcessingTask = NULL;
};

#endif // __PROCESS_STREAM_H__