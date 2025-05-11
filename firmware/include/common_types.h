#ifndef __COMMON_TYPES_H__
#define __COMMON_TYPES_H__
#include "esp_all.h"
#include <stdint.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <set>

// 点结构
struct Point {
    int x;
    int y;
    Point() : x(0), y(0) {}
    Point(int _x, int _y) : x(_x), y(_y) {}
};

// 图像处理结构
typedef struct {
    uint8_t* data; // 图像数据
    int width;    // 图像宽度
    int height;   // 图像高度
    int channels; // 通道数
} processed_image_t;

// 颜色阈值定义 (HSV空间)
typedef struct {
    uint8_t h_min, h_max;
    uint8_t s_min, s_max;
    uint8_t v_min, v_max;
} color_threshold_t;

// 线段结构定义
typedef struct {
    int16_t x1, y1;
    int16_t x2, y2;
    float slope;
} line_segment_t;

// 常量定义
#define MIN_SLOPE 0.4             // 最小斜率阈值
#define BUFFER_SIZE 256           // 通用缓冲区大小
#define ROI_RATIO 0.6             // ROI区域比例
#ifndef PI
#define PI 3.14159265358979323846 // PI常量
#endif
// 区域类型枚举
typedef enum {
    AREA_LEVEL_0 = 0,   // 0级区域 (两车道线内侧与模型预测的重叠区)
    AREA_LEVEL_1 = 1,   // 1级区域 (车道线扩展的100px红色掩膜区)
    AREA_LEVEL_2 = 2    // 2级区域 (最外侧)
} area_level_t;

// 多项式拟合结果
struct poly_fit_t {
    float a;    // 二次项系数
    float b;    // 一次项系数
    float c;    // 常数项
    bool valid; // 标记拟合是否有效
    
    poly_fit_t() : a(0.0f), b(0.0f), c(0.0f), valid(false) {}
};

// LaneInfo结构，用于返回车道检测结果
typedef struct {
    poly_fit_t left_lane;    // 左车道线拟合参数
    poly_fit_t right_lane;   // 右车道线拟合参数
    uint8_t* area_mask;      // 区域掩码（内存需要由调用者释放）
    int width;               // 掩码宽度
    int height;              // 掩码高度
    bool valid;              // 结果是否有效
} LaneInfo;

// 目标检测结果
struct Detection {
    int classId;                // 类别ID
    float confidence;           // 置信度
    float x, y, width, height;  // 边界框 (归一化坐标0-1)
    
    Detection() : classId(0), confidence(0.0f), x(0.0f), y(0.0f), width(0.0f), height(0.0f) {}
    
    Detection(int class_id, float conf, float x_pos, float y_pos, float w, float h) 
        : classId(class_id), confidence(conf), x(x_pos), y(y_pos), width(w), height(h) {}
};
// 定义卡尔曼滤波跟踪状态
typedef enum {
    TENTATIVE = 0,    // 临时状态，刚开始跟踪
    CONFIRMED = 1,    // 确认状态，稳定跟踪
    DELETED = 2       // 已删除状态
} TrackState;

// 跟踪结果结构
struct Track {
    int trackId;          // 跟踪ID
    int classId;          // 类别ID
    float confidence;     // 置信度
    float x, y;           // 位置
    float width, height;  // 大小
    float vx, vy;         // 速度
    int age;              // 跟踪帧数
};

// TTC结果结构
struct TTCResult {
    int trackId;          // 跟踪ID
    int classId;          // 目标类别
    float ttc;            // 碰撞时间（秒）
    float distance;       // 估计距离（米）
    float risk;           // 风险评分 0-1
};
// 如果C++标准小于C++14，提供自己的make_unique实现
#if __cplusplus < 201402L
namespace std {
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

// 帧数据结构体
struct FrameData {
    int frameId;                // 帧ID
    double timestamp;           // 时间戳
    int width;                  // 图像宽度
    int height;                 // 图像高度
    int channels;               // 图像通道数
    uint8_t *imageData;         // 图像数据指针（如果在内存中）
    bool needToFreeImageData;   // 是否需要释放imageData内存
    std::string tempFilePath;   // 图像数据的临时文件路径（如果在SD卡上）
    bool dataOnSD;              // 数据是否存储在SD卡上
    
    // 检测和跟踪结果
    std::vector<Detection> detections;
    std::vector<Track> tracks;
    std::vector<LaneInfo> laneInfo;
    std::vector<TTCResult> ttcResults;
    
    // 从SD卡加载图像数据到内存
    bool loadImageFromSD() {
        if (!dataOnSD || tempFilePath.empty()) {
            return false;
        }
        
        FILE* fp = fopen(tempFilePath.c_str(), "rb");
        if (!fp) {
            return false;
        }
        
        // 读取帧元数据（如果没有在结构中）
        int storedWidth, storedHeight, storedChannels;
        if (fread(&storedWidth, sizeof(int), 1, fp) != 1 ||
            fread(&storedHeight, sizeof(int), 1, fp) != 1 ||
            fread(&storedChannels, sizeof(int), 1, fp) != 1) {
            fclose(fp);
            return false;
        }
        
        // 使用存储的尺寸或者已知尺寸
        int imgWidth = (width > 0) ? width : storedWidth;
        int imgHeight = (height > 0) ? height : storedHeight;
        int imgChannels = (channels > 0) ? channels : storedChannels;
        
        // 分配内存并读取图像数据
        size_t dataSize = imgWidth * imgHeight * imgChannels;
        imageData = (uint8_t*)malloc(dataSize);
        if (!imageData) {
            fclose(fp);
            return false;
        }
        
        if (fread(imageData, 1, dataSize, fp) != dataSize) {
            free(imageData);
            imageData = nullptr;
            fclose(fp);
            return false;
        }
        
        fclose(fp);
        needToFreeImageData = true;
        dataOnSD = false;
        width = imgWidth;
        height = imgHeight;
        channels = imgChannels;
        
        return true;
    }
    
    // 删除SD卡上的临时文件
    void deleteSDFile() {
        if (!tempFilePath.empty()) {
            remove(tempFilePath.c_str());
            tempFilePath.clear();
        }
    }
    
    // 析构函数确保正确释放内存和清理文件
    ~FrameData() {
        if (needToFreeImageData && imageData) {
            free(imageData);
            imageData = nullptr;
        }
        deleteSDFile();
    }
    
    // 默认构造函数
    FrameData() : frameId(0), timestamp(0), width(0), height(0), 
                  channels(0), imageData(nullptr), needToFreeImageData(false),
                  dataOnSD(false) {}
};





// YOLOv5-nano模型配置
typedef struct {
    int input_width;         // 输入宽度
    int input_height;        // 输入高度
    int num_classes;         // 类别数量
    int num_anchors;         // 每个网格的锚点数量
    float conf_threshold;    // 置信度阈值
    float nms_threshold;     // NMS阈值
    std::vector<int> strides;     // 步长列表
    std::vector<std::vector<float>> anchors; // 锚点列表
} yolo_config_t;



enum FrameCacheMode {
    SD_CARD,     // 使用SD卡存储帧
    RAM_ONLY,    // 仅使用RAM存储帧
    HYBRID       // 混合模式，优先使用RAM，不足时使用SD卡
};


// 调试函数
inline void logDetections(const std::vector<Detection>& detections) {
    ESP_LOGI("Model", "检测到 %d 个目标:", detections.size());
    for (size_t i = 0; i < detections.size(); i++) {
        const Detection& det = detections[i];
        ESP_LOGI("Model", "  目标 %d: 类别=%d, 置信度=%.2f, 位置=[%.2f, %.2f, %.2f, %.2f]",
                i, det.classId, det.confidence, det.x, det.y, det.width, det.height);
    }
}

inline void logTracks(const std::vector<Track>& tracks) {
    ESP_LOGI("DeepSORT", "跟踪 %d 个目标:", tracks.size());
    for (size_t i = 0; i < tracks.size(); i++) {
        const Track& track = tracks[i];
        ESP_LOGI("DeepSORT", "  跟踪 %d: ID=%d, 类别=%d, 位置=[%.2f, %.2f, %.2f, %.2f], 速度=[%.2f, %.2f]",
                i, track.trackId, track.classId, track.x, track.y, track.width, track.height, track.vx, track.vy);
    }
}

#endif // __COMMON_TYPES_H__