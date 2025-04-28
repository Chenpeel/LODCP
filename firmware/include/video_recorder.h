#ifndef VIDEO_RECORDER_H
#define VIDEO_RECORDER_H
#include "../include/process.h"
#include "esp_camera.h"
#include "esp_err.h"
#include <string>

// 视频格式枚举
enum VideoFormat {
  VIDEO_MJPEG, // 将JPEG帧保存为MJPEG
  VIDEO_H264,  // 使用H264编码（需要额外的编码库）
  VIDEO_FRAMES // 保存为单独的帧序列
};

// 视频录制配置
typedef struct {
  VideoFormat format;     // 视频格式
  std::string filename;   // 基础文件名
  uint32_t max_frames;    // 最大帧数
  uint32_t fps;           // 目标帧率
  bool include_timestamp; // 是否在文件名中包含时间戳
  int quality;            // JPEG质量 (1-63，1最高)
  bool draw_detections;   // 是否在处理后的帧上绘制检测结果
  bool draw_segmentation; // 是否在处理后的帧上绘制分割结果
} video_config_t;

// 初始化视频录制器
esp_err_t init_video_recorder(const video_config_t &config);

// 开始录制
esp_err_t start_recording();

// 停止录制
esp_err_t stop_recording();

// 保存未处理的帧
esp_err_t save_raw_frame(camera_fb_t *fb);

// 保存处理后的帧（包含检测、分割等视觉效果）
esp_err_t save_processed_frame(camera_fb_t *fb,
                               const frame_processing_result_t &result);

// 获取录制状态
bool is_recording();

// 获取剩余SD卡空间（MB）
float get_remaining_storage();

esp_err_t start_video_server();
#endif // VIDEO_RECORDER_H
