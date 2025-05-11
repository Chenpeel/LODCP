#ifndef __VIDEO_SAVE_H__
#define __VIDEO_SAVE_H__
#include "esp_all.h"
#include "common_types.h"
#include <string>
#include <vector>
#include <memory>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <dirent.h>

class VideoSave {
private:
    // 常量定义
    static constexpr const char* TAG = "VideoSave";
    static constexpr const char* BASE_SAVE_PATH = "/sdcard/save";
    static constexpr const char* VIDEO_AVI_FOURCC = "MJPG";
    static constexpr int DEFAULT_FPS = 15;
    static constexpr int QUALITY = 90;  // JPEG压缩质量
    
    // AVI文件结构相关常量
    struct AVIConstants {
        static constexpr uint32_t RIFF_FOURCC = 0x46464952;  // "RIFF"
        static constexpr uint32_t AVI_FOURCC = 0x20495641;   // "AVI "
        static constexpr uint32_t LIST_FOURCC = 0x5453494C;  // "LIST"
        static constexpr uint32_t HDRL_FOURCC = 0x6C726468;  // "hdrl"
        static constexpr uint32_t AVIH_FOURCC = 0x68697661;  // "avih"
        static constexpr uint32_t STRL_FOURCC = 0x6C727473;  // "strl"
        static constexpr uint32_t STRH_FOURCC = 0x68727473;  // "strh"
        static constexpr uint32_t STRF_FOURCC = 0x66727473;  // "strf"
        static constexpr uint32_t MOVI_FOURCC = 0x69766F6D;  // "movi"
        static constexpr uint32_t VIDS_FOURCC = 0x73646976;  // "vids"
        static constexpr uint32_t MJPG_FOURCC = 0x47504A4D;  // "MJPG"
        static constexpr uint32_t IDX1_FOURCC = 0x31786469;  // "idx1"
        static constexpr uint32_t DC_FOURCC = 0x63643030;    // "00dc"
    };
    
    // AVI文件头结构
    struct AVIHeader {
        uint32_t microSecPerFrame;    // 帧间隔（微秒）
        uint32_t maxBytesPerSec;      // 最大数据率
        uint32_t paddingGranularity;  // 填充
        uint32_t flags;               // 标志，如是否有索引
        uint32_t totalFrames;         // 总帧数
        uint32_t initialFrames;       // 初始帧
        uint32_t streams;             // 流数量
        uint32_t suggestedBufferSize; // 建议缓冲区大小
        uint32_t width;               // 宽度
        uint32_t height;              // 高度
        uint32_t reserved[4];         // 保留
    };
    
    // 流头结构
    struct AVIStreamHeader {
        uint32_t fccType;            // 流类型（vids表示视频）
        uint32_t fccHandler;         // 处理器（MJPG表示Motion JPEG）
        uint32_t flags;              // 标志
        uint16_t priority;           // 优先级
        uint16_t language;           // 语言
        uint32_t initialFrames;      // 初始帧
        uint32_t scale;              // 时间尺度
        uint32_t rate;               // 帧率（rate/scale）
        uint32_t start;              // 开始时间
        uint32_t length;             // 时长（帧数）
        uint32_t suggestedBufferSize;// 建议缓冲区大小
        uint32_t quality;            // 质量（0-10000）
        uint32_t sampleSize;         // 样本大小
        struct {
            short int left;
            short int top;
            short int right;
            short int bottom;
        } rcFrame;                   // 帧区域
    };
    
    // 位图信息头结构
    struct BITMAPINFOHEADER {
        uint32_t biSize;             // 本结构大小
        int32_t  biWidth;            // 宽度
        int32_t  biHeight;           // 高度
        uint16_t biPlanes;           // 平面数
        uint16_t biBitCount;         // 每像素位数
        uint32_t biCompression;      // 压缩方式
        uint32_t biSizeImage;        // 图像大小
        int32_t  biXPelsPerMeter;    // 水平分辨率
        int32_t  biYPelsPerMeter;    // 垂直分辨率
        uint32_t biClrUsed;          // 使用的颜色数
        uint32_t biClrImportant;     // 重要的颜色数
    };
    
    // AVI索引项结构
    struct AVIINDEXENTRY {
        uint32_t ckid;               // 块ID
        uint32_t dwFlags;            // 标志
        uint32_t dwChunkOffset;      // 块偏移
        uint32_t dwChunkLength;      // 块长度
    };
    
    // 本地变量
    std::string baseDir;              // 基本保存路径
    std::string currentSessionDir;    // 当前会话目录
    std::string currentAviFile;       // 当前AVI文件路径
    FILE* aviFile;                    // AVI文件句柄
    
    int frameWidth;                   // 帧宽度
    int frameHeight;                  // 帧高度
    int frameRate;                    // 帧率
    int totalFrames;                  // 总帧数
    uint64_t moviListPos;             // movi列表位置
    uint64_t aviStartPos;             // AVI文件起始位置（用于计算偏移）
    
    std::vector<AVIINDEXENTRY> indexEntries; // 索引项列表
    
    // 私有方法
    bool initSaveDirectory();         // 初始化保存目录
    std::string generateTimestamp();  // 生成时间戳
    bool writeAVIHeader(int width, int height, int fps); // 写入AVI文件头
    bool finalizeAVI();               // 完成AVI文件写入
    bool convertToJPEG(uint8_t* input, int width, int height, int channels, 
                      uint8_t** output, size_t* outSize); // 转换为JPEG
    void visualizeDetections(uint8_t* imageData, int width, int height, int channels, 
                           const std::vector<Detection>& detections); // 可视化检测结果
    
public:
    VideoSave();                      // 构造函数
    ~VideoSave();                     // 析构函数
    
    // 初始化并开始新的视频保存会话
    bool startSession();
    
    // 结束当前会话
    void endSession();
    
    // 保存单帧图像（自动JPEG编码）
    bool saveFrame(const std::shared_ptr<FrameData>& frameData, bool visualize = true);
    
    // 保存帧序列为AVI视频
    bool startVideoCapture(int width, int height, int fps = DEFAULT_FPS);
    bool addFrameToVideo(const std::shared_ptr<FrameData>& frameData, bool visualize = true);
    bool finishVideoCapture();
    
    // 获取视频文件路径
    std::string getCurrentVideoPath() const { return currentAviFile; }
    
    // 获取当前会话目录
    std::string getCurrentSessionDir() const { return currentSessionDir; }
    
    // 获取已保存的帧数
    int getFrameCount() const { return totalFrames; }
};

#endif // __VIDEO_SAVE_H__