#ifndef __DEEP_SORT_H__
#define __DEEP_SORT_H__
#include "esp_all.h"
#include "config.h"
#include "common_types.h"
#include "esp_tflite.h"
#include "object_detect.h"
#include "kalman.h"

// KalmanTracker类，表示每个目标的跟踪器
class KalmanTracker {
public:
    KalmanTracker(const Detection& detection, int track_id, int max_age = 30, int min_hits = 3);
    void predict();
    void update(const Detection& detection);
    
    // 获取当前预测的位置和大小
    Detection getPredictedState() const;
    
    // 状态管理
    bool isConfirmed() const { return state == CONFIRMED; }
    bool isDeleted() const { return state == DELETED; }
    void markMissed() { time_since_update++; }
    
    // 将检查删除状态的函数设为公有
    void checkForDeletion();
    
    // 属性
    int trackId;              // 跟踪ID
    int hits;                 // 连续命中次数
    int age;                  // 跟踪器生命期
    int time_since_update;    // 自上次更新后帧数
    TrackState state;         // 跟踪状态
    int classId;              // 类别ID
    float confidence;         // 置信度
    
    // 卡尔曼滤波器
    SimpleKalmanFilter kf;    // 使用自定义卡尔曼滤波器
    
private:
    int max_age;              // 最大跟踪帧数
    int min_hits;             // 最小命中次数
    
    // 状态转换
    void tentativeToConfirmed();
};

// DeepSORT类定义
class DeepSORT {
public:
    DeepSORT(float max_iou_distance = 0.7f, int max_age = 30, int n_init = 3);
    ~DeepSORT();
    
    // 主要接口
    std::vector<Track> update(const std::vector<Detection>& detections, uint8_t* imageData, int width, int height);
    std::vector<Track> track(uint8_t *imageData, int width, int height, int channels);
    
    // 重置跟踪器
    void reset();
    
private:
    // IoU匹配相关
    float calculateIoU(const Detection& det1, const Detection& det2);
    std::vector<std::vector<float>> computeIouMatrix(
        const std::vector<Detection>& detections,
        const std::vector<KalmanTracker>& trackers);
    std::vector<std::pair<int, int>> hungrarianMatching(const std::vector<std::vector<float>>& cost_matrix);
    
    // 内部管理方法
    void initTracker(const Detection& detection);
    void updateTracker(int track_idx, const Detection& detection);
    void predictTrackers();
    void deleteOldTrackers();
    
    // 转换跟踪结果
    std::vector<Track> generateTrackResults();
    
    // 跟踪器成员变量
    std::vector<KalmanTracker> trackers;
    int next_track_id;
    
    // 配置参数
    float max_iou_distance;
    int max_age;
    int n_init;
    
    // 对象检测器（用于track方法）
    ObjectDetect detector;
    
    // 日志标签
    const char* TAG = "DeepSORT";
};

#endif // __DEEP_SORT_H__