#ifndef __TTC_H__
#define __TTC_H__
#include "esp_all.h"
#include "config.h"
#include "common_types.h"
#include "deep_sort.h"
#include <vector>

// TTC配置结构
struct TTCConfig {
    float areaDangerCoeff[3];     // 区域危险系数
    float objectDangerCoeff[10];  // 目标类型危险系数
    float pixelToMeterRatio;      // 像素到米的转换比例
    int minTrackAge;              // 最小跟踪帧数
    float ttcThresholdCritical;   // 临界TTC阈值
    float ttcThresholdWarning;    // 警告TTC阈值
    float frameRate;              // 帧率
    
    TTCConfig() : 
        pixelToMeterRatio(0.05f), 
        minTrackAge(5),
        ttcThresholdCritical(0.5f),
        ttcThresholdWarning(3.0f),
        frameRate(30.0f)
    {
        // 默认区域危险系数
        areaDangerCoeff[0] = 1.0f;  // 高风险区域
        areaDangerCoeff[1] = 0.7f;  // 中等风险区域 
        areaDangerCoeff[2] = 0.3f;  // 低风险区域
        
        // 默认目标危险系数
        objectDangerCoeff[0] = 0.95f;  // active
        objectDangerCoeff[1] = 0.10f;  // traffic sign
    }
};

class TTC
{
public:
    TTC();
    ~TTC();
    
    // 主要接口
    std::vector<TTCResult> calculate(const std::vector<Track>& tracks, const std::vector<LaneInfo>& laneInfo);
    
    // 参数配置
    void updateParameters(const TTCConfig& newConfig);
    
    // 获取当前配置
    TTCConfig getConfig() const;

private:
    // 确定目标所在的区域级别
    area_level_t determineAreaLevel(float x, float y, const std::vector<LaneInfo>& laneInfo);
    
    // 估计与目标的距离
    float estimateDistance(const Track& track);
    
    // 计算碰撞时间
    float calculateTTC(const Track& track, float distance);
    
    // 计算风险评分
    float calculateRisk(const Track& track, float ttc, area_level_t areaLevel, float distance);
    
    // 基于IoU计算碰撞风险
    float calculateCollisionRiskByIoU(const Track& track, const std::vector<LaneInfo>& laneInfo, int predictFrames = 30);

    // 成员变量
    static const int MAX_OBJECT_CLASSES = 10;  // 最大支持的目标类别数
    float areaDangerCoeff[3];                  // 不同区域的危险系数
    float objectDangerCoeff[MAX_OBJECT_CLASSES]; // 不同目标类型的危险系数
    float pixelToMeterRatio;                   // 像素到实际距离的转换比例
    int minTrackAge;                           // 最小跟踪帧数，用于过滤不稳定检测
    float ttcThresholdCritical;                // 临界TTC阈值
    float ttcThresholdWarning;                 // 警告TTC阈值
    float frameRate;                           // 帧率
};
#endif // __TTC_H__