#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "esp_err.h"
#include "tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tflite-micro/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tflite-micro/tensorflow/lite/schema/schema_generated.h"

// 模型文件路径
#define DETECTION_MODEL_PATH "/sdcard/detect.tflite"
#define SEGMENTATION_MODEL_PATH "/sdcard/segment.tflite"

// 模型类型枚举
enum ModelType { MODEL_DETECTION = 0, MODEL_SEGMENTATION = 1 };

// 模型加载器类
class ModelLoader {
public:
  ModelLoader(ModelType type);
  ~ModelLoader();

  // 初始化模型
  esp_err_t init();

  // 运行推理
  esp_err_t runInference(uint8_t *input_data, size_t input_size);

  // 获取输出数据
  float *getOutputData();
  uint8_t *getQuantizedOutputData();
  size_t getOutputSize();
  int getOutputHeight();
  int getOutputWidth();
  int getOutputChannels();

  // 获取输入尺寸
  void getInputDims(int &width, int &height, int &channels);

private:
  const char *getModelPath();

  ModelType model_type_;
  const tflite::Model *model_ = nullptr;
  tflite::MicroErrorReporter error_reporter_;
  tflite::MicroMutableOpResolver<10> resolver_; // 使用MicroMutableOpResolver
  tflite::MicroInterpreter *interpreter_ = nullptr;

  TfLiteTensor *input_tensor_ = nullptr;
  TfLiteTensor *output_tensor_ = nullptr;

  uint8_t *model_data_ = nullptr;
  size_t model_size_ = 0;

  // TFLite需要的内存区域
  static constexpr int kTensorArenaSize = 800 * 1024;
  uint8_t *tensor_arena_ = nullptr;
};

extern ModelLoader g_detection_model;
extern ModelLoader g_segmentation_model;

#endif // MODEL_LOADER_H
