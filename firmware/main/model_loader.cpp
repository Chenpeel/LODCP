#include "../include/model_loader.h"
#include "esp_log.h"
#include "esp_system.h"
#include <stdio.h>

static const char *TAG = "ModelLoader";

// 全局模型实例
ModelLoader g_detection_model(MODEL_DETECTION);
ModelLoader g_segmentation_model(MODEL_SEGMENTATION);

ModelLoader::ModelLoader(ModelType type) : model_type_(type) {
  tensor_arena_ = (uint8_t *)heap_caps_malloc(
      kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!tensor_arena_) {
    ESP_LOGE(TAG, "Failed to allocate memory for tensor arena");
  }
}

ModelLoader::~ModelLoader() {
  if (tensor_arena_) {
    free(tensor_arena_);
  }
  if (model_data_) {
    free(model_data_);
  }
  if (interpreter_) {
    delete interpreter_;
  }
}

const char *ModelLoader::getModelPath() {
  switch (model_type_) {
  case MODEL_DETECTION:
    return DETECTION_MODEL_PATH;
  case MODEL_SEGMENTATION:
    return SEGMENTATION_MODEL_PATH;
  default:
    return "";
  }
}

esp_err_t ModelLoader::init() {
  const char *model_path = getModelPath();
  ESP_LOGI(TAG, "Loading model: %s", model_path);

  // 打开模型文件
  FILE *model_file = fopen(model_path, "rb");
  if (!model_file) {
    ESP_LOGE(TAG, "Failed to open model file: %s", model_path);
    return ESP_FAIL;
  }

  // 获取文件大小
  fseek(model_file, 0, SEEK_END);
  model_size_ = ftell(model_file);
  fseek(model_file, 0, SEEK_SET);

  ESP_LOGI(TAG, "Model file size: %d bytes", model_size_);

  // 分配内存并读取模型数据
  model_data_ = (uint8_t *)malloc(model_size_);
  if (!model_data_) {
    ESP_LOGE(TAG, "Failed to allocate memory for model");
    fclose(model_file);
    return ESP_FAIL;
  }

  size_t bytes_read = fread(model_data_, 1, model_size_, model_file);
  fclose(model_file);

  if (bytes_read != model_size_) {
    ESP_LOGE(TAG, "Failed to read model file completely");
    free(model_data_);
    model_data_ = nullptr;
    return ESP_FAIL;
  }

  // 获取模型
  model_ = tflite::GetModel(model_data_);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch: %d vs %d", model_->version(),
             TFLITE_SCHEMA_VERSION);
    return ESP_FAIL;
  }

  // 创建解释器
  interpreter_ = new tflite::MicroInterpreter(
      model_, resolver_, tensor_arena_, kTensorArenaSize, &error_reporter_);

  // 分配张量
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to allocate tensors");
    return ESP_FAIL;
  }

  // 获取输入和输出张量
  input_tensor_ = interpreter_->input(0);
  output_tensor_ = interpreter_->output(0);

  ESP_LOGI(TAG, "Model %s loaded successfully",
           model_type_ == MODEL_DETECTION ? "Detection" : "Segmentation");
  ESP_LOGI(TAG, "Input shape: %d x %d x %d x %d", input_tensor_->dims->data[0],
           input_tensor_->dims->data[1], input_tensor_->dims->data[2],
           input_tensor_->dims->data[3]);

  if (output_tensor_->dims->size >= 4) {
    ESP_LOGI(TAG, "Output shape: %d x %d x %d x %d",
             output_tensor_->dims->data[0], output_tensor_->dims->data[1],
             output_tensor_->dims->data[2], output_tensor_->dims->data[3]);
  } else {
    ESP_LOGI(TAG, "Output shape: %d", output_tensor_->dims->data[1]);
  }

  return ESP_OK;
}

esp_err_t ModelLoader::runInference(uint8_t *input_data, size_t input_size) {
  // 检查模型和解释器是否已初始化
  if (!model_ || !interpreter_ || !input_tensor_) {
    ESP_LOGE(TAG, "Model not initialized");
    return ESP_FAIL;
  }

  // 检查输入大小
  size_t input_tensor_size = 1;
  for (int i = 0; i < input_tensor_->dims->size; i++) {
    input_tensor_size *= input_tensor_->dims->data[i];
  }

  if (input_tensor_->type == kTfLiteUInt8) {
    input_tensor_size *= sizeof(uint8_t);
  } else if (input_tensor_->type == kTfLiteFloat32) {
    input_tensor_size *= sizeof(float);
  }

  if (input_size != input_tensor_size) {
    ESP_LOGE(TAG, "Input size mismatch: %d vs %d", input_size,
             input_tensor_size);
    return ESP_FAIL;
  }

  // 复制输入数据
  if (input_tensor_->type == kTfLiteUInt8) {
    memcpy(input_tensor_->data.uint8, input_data, input_size);
  } else if (input_tensor_->type == kTfLiteFloat32) {
    memcpy(input_tensor_->data.f, input_data, input_size);
  }

  // 运行推理
  TfLiteStatus invoke_status = interpreter_->Invoke();
  if (invoke_status != kTfLiteOk) {
    ESP_LOGE(TAG, "Failed to invoke interpreter: %d", invoke_status);
    return ESP_FAIL;
  }

  return ESP_OK;
}

float *ModelLoader::getOutputData() {
  if (!output_tensor_) {
    ESP_LOGE(TAG, "No output tensor available");
    return nullptr;
  }

  if (output_tensor_->type == kTfLiteFloat32) {
    return output_tensor_->data.f;
  } else {
    ESP_LOGW(TAG, "Output tensor is not float type");
    return nullptr;
  }
}

uint8_t *ModelLoader::getQuantizedOutputData() {
  if (!output_tensor_) {
    ESP_LOGE(TAG, "No output tensor available");
    return nullptr;
  }

  if (output_tensor_->type == kTfLiteUInt8) {
    return output_tensor_->data.uint8;
  } else {
    ESP_LOGW(TAG, "Output tensor is not uint8 type");
    return nullptr;
  }
}

size_t ModelLoader::getOutputSize() {
  if (!output_tensor_) {
    return 0;
  }

  size_t output_size = 1;
  for (int i = 0; i < output_tensor_->dims->size; i++) {
    output_size *= output_tensor_->dims->data[i];
  }
  return output_size;
}

int ModelLoader::getOutputHeight() {
  if (!output_tensor_ || output_tensor_->dims->size < 2) {
    return 0;
  }
  return output_tensor_->dims->data[1];
}

int ModelLoader::getOutputWidth() {
  if (!output_tensor_ || output_tensor_->dims->size < 3) {
    return 0;
  }
  return output_tensor_->dims->data[2];
}

int ModelLoader::getOutputChannels() {
  if (!output_tensor_ || output_tensor_->dims->size < 4) {
    return 0;
  }
  return output_tensor_->dims->data[3];
}

void ModelLoader::getInputDims(int &width, int &height, int &channels) {
  if (!input_tensor_ || input_tensor_->dims->size < 4) {
    width = height = channels = 0;
    return;
  }

  // 假设输入格式为 [1, height, width, channels]
  height = input_tensor_->dims->data[1];
  width = input_tensor_->dims->data[2];
  channels = input_tensor_->dims->data[3];
}
