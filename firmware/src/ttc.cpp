#include "ttc.h"
#include <cmath>
#include <algorithm>
#include "esp_log.h"

static const char* TAG = "TTC";

// 构造函数
TTC::TTC() {
    // 设置默认值
    TTCConfig defaultConfig;
    updateParameters(defaultConfig);
}

// 析构函数
TTC::~TTC() {
    // 不需要特殊清理
}

// 更新参数
void TTC::updateParameters(const TTCConfig& newConfig) {
    // 复制配置参数
    for (int i = 0; i < 3; i++) {
        areaDangerCoeff[i] = newConfig.areaDangerCoeff[i];
    }
    
    for (int i = 0; i < MAX_OBJECT_CLASSES; i++) {
        if (i < 10) { // 确保不越界
            objectDangerCoeff[i] = newConfig.objectDangerCoeff[i];
        } else {
            objectDangerCoeff[i] = 0.5f; // 默认中等危险系数
        }
    }
    
    pixelToMeterRatio = newConfig.pixelToMeterRatio;
    minTrackAge = newConfig.minTrackAge;
    ttcThresholdCritical = newConfig.ttcThresholdCritical;
    ttcThresholdWarning = newConfig.ttcThresholdWarning;
    frameRate = newConfig.frameRate;
}

// 主要计算函数
std::vector<TTCResult> TTC::calculate(const std::vector<Track>& tracks, const std::vector<LaneInfo>& laneInfo) {
    std::vector<TTCResult> results;
    
    for (const auto& track : tracks) {
        // 过滤掉刚开始跟踪的不稳定目标
        if (track.age < minTrackAge) {
            continue;
        }
        
        // 估计与目标的距离(米)
        float distance = estimateDistance(track);
        
        // 计算TTC(秒)
        float ttc = calculateTTC(track, distance);
        
        // 确定目标所在区域
        area_level_t areaLevel = determineAreaLevel(track.x, track.y, laneInfo);
        
        // 计算风险评分
        float risk = calculateRisk(track, ttc, areaLevel, distance);
        
        // 如果风险较高或TTC值较小，添加到结果中
        if (risk > 0.1f || ttc < 10.0f) {
            TTCResult result;
            result.trackId = track.trackId;
            result.classId = track.classId;
            result.ttc = ttc;
            result.distance = distance;
            result.risk = risk;
            
            results.push_back(result);
            
            // 记录警告日志
            if (risk > 0.7f) {
                ESP_LOGW(TAG, "高风险! 目标ID: %d, TTC: %.2f秒, 风险: %.2f", 
                         result.trackId, result.ttc, result.risk);
            }
        }
    }
    
    return results;
}

// 确定目标所在的区域级别
area_level_t TTC::determineAreaLevel(float x, float y, const std::vector<LaneInfo>& laneInfo) {
    // 如果没有车道线信息，默认为最高风险区域
    if (laneInfo.empty() || !laneInfo[0].valid) {
        return AREA_LEVEL_0; // 高风险区域
    }
    
    const LaneInfo& lane = laneInfo[0];
    
    // 如果车道线检测不可靠，返回默认值
    if (!lane.left_lane.valid || !lane.right_lane.valid) {
        return AREA_LEVEL_0;
    }
    
    // 获取图像高度和宽度
    int height = lane.height;
    int width = lane.width;
    
    // 使用目标底部中心点来判断位置
    float bottomCenterX = x + width / 2.0f;
    float bottomY = y + height;
    
    // 计算在此y坐标下的左右车道线位置
    float leftLaneX = lane.left_lane.a * bottomY * bottomY + 
                      lane.left_lane.b * bottomY + 
                      lane.left_lane.c;
                      
    float rightLaneX = lane.right_lane.a * bottomY * bottomY + 
                       lane.right_lane.b * bottomY + 
                       lane.right_lane.c;
    
    // 确保左右车道线顺序正确
    if (leftLaneX > rightLaneX) {
        std::swap(leftLaneX, rightLaneX);
    }
    
    // 计算车道宽度
    float laneWidth = rightLaneX - leftLaneX;
    
    // 计算中心线
    float centerLaneX = (leftLaneX + rightLaneX) / 2.0f;
    
    // 计算目标到中心线的距离
    float distanceToCenterLine = fabs(bottomCenterX - centerLaneX);
    
    // 根据距离车道中心线的远近划分区域
    // 核心行驶区域 (中心1/3)
    if (distanceToCenterLine < laneWidth / 6.0f) {
        return AREA_LEVEL_0; // 高风险区域
    }
    // 一般行驶区域 (中间1/3)
    else if (distanceToCenterLine < laneWidth / 3.0f) {
        return AREA_LEVEL_1; // 中等风险区域
    }
    // 外围区域
    else {
        return AREA_LEVEL_2; // 低风险区域
    }
}

// 估计与目标的距离
float TTC::estimateDistance(const Track& track) {
    // 采用简单逆比例关系，物体实际高度与像素高度成反比
    // 这里假设典型物体(汽车)的高度约为1.5米
    float objectRealHeight = 1.5f; // 单位：米
    if (track.classId == 1) {  // 假设类别1是行人
        objectRealHeight = 1.7f;
    } else if (track.classId == 2) { // 假设类别2是自行车
        objectRealHeight = 1.2f;
    }
    
    // 距离 = 物体实际高度 / (物体像素高度 * 每像素代表的米数)
    float distance = objectRealHeight / (track.height * pixelToMeterRatio);
    
    return distance;
}

// 计算碰撞时间
float TTC::calculateTTC(const Track& track, float distance) {
    // 如果没有速度或速度太小，返回一个大值表示无碰撞风险
    float velocityMagnitude = sqrtf(track.vx * track.vx + track.vy * track.vy);
    if (velocityMagnitude < 0.1f) {
        return 1000.0f; // 非常大的值，表示实际上不会碰撞
    }
    
    // 计算速度在z方向(深度)的分量
    // 这里采用一个启发式方法，对于向上或向下移动的物体，其z方向速度较小
    float vyAbs = fabsf(track.vy);
    float vxAbs = fabsf(track.vx);
    
    // 假设主要是沿x轴(水平)运动时，有更大可能是向我们接近/远离
    float vz = vxAbs > vyAbs ? vxAbs * 0.5f : vxAbs * 0.1f;
    
    // 如果物体变大(高度增加)，说明在接近
    if (track.age > 5 && track.vy < -0.5f) { // 物体高度增加(向上移动)
        vz = fabs(track.vy) * 2.0f; // 较大的接近速度
    }
    
    // 如果速度非常小，设置一个最小值避免除零
    if (vz < 0.1f) {
        vz = 0.1f;
    }
    
    // 将像素/帧的速度转换为米/秒
    // 假设帧率为30fps
    float vzMeterPerSec = vz * pixelToMeterRatio * frameRate;
    
    // 计算TTC = 距离 / 速度
    float ttc = distance / vzMeterPerSec;
    
    // 限制TTC范围，避免极端值
    ttc = std::max(0.0f, std::min(ttc, 1000.0f));
    
    return ttc;
}

// 使用IoU和轨迹预测计算碰撞风险
float TTC::calculateCollisionRiskByIoU(const Track& track, const std::vector<LaneInfo>& laneInfo, int predictFrames) {
    // 如果没有速度或速度太小，风险低
    if (std::abs(track.vx) < 0.1f && std::abs(track.vy) < 0.1f) {
        return 0.0f;
    }
    
    // 预测未来位置
    float futureX = track.x + track.vx * predictFrames;
    float futureY = track.y + track.vy * predictFrames;
    float futureWidth = track.width;  // 假设大小不变
    float futureHeight = track.height;
    
    // 检查是否与车道线相交
    if (laneInfo.empty() || !laneInfo[0].valid) {
        return 0.0f;  // 无车道信息，无法评估
    }
    
    // 获取车道线
    const poly_fit_t& left_lane = laneInfo[0].left_lane;
    const poly_fit_t& right_lane = laneInfo[0].right_lane;
    
    if (!left_lane.valid || !right_lane.valid) {
        return 0.0f;
    }
    
    // 计算车辆中心点
    float centerX = futureX + futureWidth / 2.0f;
    float centerY = futureY + futureHeight / 2.0f;
    
    // 计算该点处左右车道线的x坐标
    float leftX = left_lane.a * centerY * centerY + left_lane.b * centerY + left_lane.c;
    float rightX = right_lane.a * centerY * centerY + right_lane.b * centerY + right_lane.c;
    
    // 确保左右顺序正确
    if (leftX > rightX) {
        std::swap(leftX, rightX);
    }
    
    // 判断车辆是否会与车道线区域重叠
    float vehicleLeft = futureX;
    float vehicleRight = futureX + futureWidth;
    
    // 计算重叠程度（简化版IoU）
    float overlap = 0.0f;
    
    // 如果车辆完全在车道线之间，没有碰撞风险
    if (vehicleLeft >= leftX && vehicleRight <= rightX) {
        return 0.0f;
    }
    
    // 如果车辆与左车道线重叠
    if (vehicleLeft < leftX && vehicleRight > leftX) {
        float overlapWidth = vehicleRight - leftX;
        overlap = std::max(overlap, overlapWidth / futureWidth);
    }
    
    // 如果车辆与右车道线重叠
    if (vehicleLeft < rightX && vehicleRight > rightX) {
        float overlapWidth = rightX - vehicleLeft;
        overlap = std::max(overlap, overlapWidth / futureWidth);
    }
    
    // 根据重叠程度计算风险
    return overlap;
}

// 计算风险评分
float TTC::calculateRisk(const Track& track, float ttc, area_level_t areaLevel, float distance) {
    // 如果TTC很大，表示无碰撞风险
    if (ttc > 10.0f) {
        return 0.0f;
    }
    
    // 基础风险分数(TTC越小风险越高)
    float riskBase = 0.0f;
    if (ttc < ttcThresholdCritical) {
        riskBase = 1.0f; // 极高风险
    } else if (ttc < ttcThresholdWarning) {
        // 在警告阈值和临界阈值之间，线性插值
        riskBase = 0.7f - 0.3f * (ttc - ttcThresholdCritical) / 
                          (ttcThresholdWarning - ttcThresholdCritical);
    } else {
        // 较低风险
        riskBase = 0.4f * (10.0f - ttc) / (10.0f - ttcThresholdWarning);
    }
    
    // 考虑区域因素
    float areaFactor = areaDangerCoeff[areaLevel];
    
    // 考虑目标类型(确保类别ID有效)
    int classId = track.classId;
    if (classId < 0 || classId >= MAX_OBJECT_CLASSES) {
        classId = 0; // 默认类别
    }
    float typeFactor = objectDangerCoeff[classId];
    
    // 考虑距离因素(距离越近风险越高)
    float distanceFactor = 1.0f;
    if (distance > 0) {
        distanceFactor = 10.0f / (10.0f + distance); // 距离因素映射到0-1之间
    }
    
    // 加权计算最终风险评分
    float risk = riskBase * 0.4f + areaFactor * 0.2f + 
                 typeFactor * 0.1f + distanceFactor * 0.3f;
    
    // 确保风险评分在0-1范围内
    risk = std::max(0.0f, std::min(1.0f, risk));
    
    return risk;
}

// 获取当前配置
TTCConfig TTC::getConfig() const {
    TTCConfig config;
    
    for (int i = 0; i < 3; i++) {
        config.areaDangerCoeff[i] = areaDangerCoeff[i];
    }
    
    for (int i = 0; i < 10; i++) {
        if (i < MAX_OBJECT_CLASSES) {
            config.objectDangerCoeff[i] = objectDangerCoeff[i];
        }
    }
    
    config.pixelToMeterRatio = pixelToMeterRatio;
    config.minTrackAge = minTrackAge;
    config.ttcThresholdCritical = ttcThresholdCritical;
    config.ttcThresholdWarning = ttcThresholdWarning;
    config.frameRate = frameRate;
    
    return config;
}