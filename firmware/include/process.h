#ifndef PROCESS_H
#define PROCESS_H
typedef struct {
  int class_id;       // 类别ID (0 active, 1 traffic signal)
  char class_name[2]; // 类别名称
  float confidence;   // 置信度 (0~1)
  int x;              // 边界框左上角x坐标
  int y;              // 边界框左上角y坐标
  int width;          // 边界框宽度
  int height;         // 边界框高度
} detection_object_t;

typedef struct {
  detection_object_t *objects; // 对象数组
  int count;                   // 检测到的对象数量
  uint64_t timestamp_ms;       // 时间戳(毫秒)
} object_detection_result_t;

esp_err_t init_processing_pipeline();
void *traditional_processing(camera_fb_t *frame);
void *semantic_segmentation(camera_fb_t *frame);
void *object_detection(camera_fb_t *frame);
void *tracking_matching(void *detection_results, void *prev_tracking_data);
float collision_prediction(void *tracking_data, void *segmentation_results);
frame_processing_result_t process_frame(camera_fb_t *frame);

#endif
