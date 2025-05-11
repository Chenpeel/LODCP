#ifndef __OBJECT_DETECT_H__
#define __OBJECT_DETECT_H__

#include "esp_all.h"
#include "config.h"
#include "common_types.h"
#include "esp_tflite.h"


class ObjectDetect
{
public:
    ObjectDetect();
    ~ObjectDetect();

    // 主要检测接口
    std::vector<Detection> detect(uint8_t *imageData, int width, int height, int channels);
    
    // 配置函数
    bool loadModel();
    void setConfThreshold(float threshold);
    void setNMSThreshold(float threshold);
    bool isModelLoaded() const { return model_loaded; }
    
private:
    // 图像预处理 - 调整大小和归一化
    void preprocess(uint8_t *input, int width, int height, int channels, 
                   float* output, int target_w, int target_h);
    
    // 从输出张量解码检测结果
    std::vector<Detection> decodeOutputs(TfLiteTensor* output_tensor);
    
    // 非极大值抑制
    std::vector<Detection> applyNMS(std::vector<Detection>& detections);
    
    // 计算IoU (交并比)
    float calculateIoU(const Detection& a, const Detection& b);
    
    // 将归一化坐标转换为原始图像坐标
    void mapToOriginalSize(std::vector<Detection>& detections, 
                          int orig_width, int orig_height);

private:
    bool model_loaded;                    // 模型是否已加载
    tflite::MicroInterpreter* interpreter; // TF Lite解释器
    yolo_config_t yolo_config;            // YOLO配置
    std::vector<std::string> class_names; // 类别名称
    int orig_width;                       // 原始图像宽度
    int orig_height;                      // 原始图像高度
    const char* TAG = "ObjectDetect";     // 日志标签
};

#endif // __OBJECT_DETECT_H__