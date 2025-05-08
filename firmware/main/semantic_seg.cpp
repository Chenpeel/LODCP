#include "../include/semantic_seg.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_tflite.h"
#include "esp_timer.h"

#include <string.h>

static const char *TAG = "SemanticSeg";

// 全局TFLite解释器
static tflite::MicroInterpreter *tflite_interpreter = nullptr;
static bool model_loaded = false;
static int model_input_width = 0;
static int model_input_height = 0;
static int model_input_channels = 0;
static int model_output_classes = 2; // 仅两个类别：可行区和非可行区

// 模型文件路径
static const char *model_file = "/spiffs/binary_segmentation.tflite";
static float confidence_threshold = 0.5f;
static bool enable_preprocessing = true;

// 类别信息
static const seg_class_info_t class_info[] = {
    {0, "可行区", {0, 255, 0}},  // 绿色表示可行区
    {1, "非可行区", {255, 0, 0}} // 红色表示非可行区
};
static const int num_class_info = sizeof(class_info) / sizeof(class_info[0]);

// 初始化语义分割模型
esp_err_t init_semantic_segmentation() {
  ESP_LOGI(TAG, "初始化二分类语义分割模型(可行区/非可行区)");

  // 检查SPIFFS是否已挂载
  if (!esp_spiffs_mounted(NULL)) {
    ESP_LOGE(TAG, "SPIFFS 未挂载，无法加载模型");
    return ESP_FAIL;
  }

  // 检查模型文件是否存在
  struct stat st;
  if (stat(model_file, &st) != 0) {
    ESP_LOGW(TAG, "模型文件不存在: %s，使用内置分割方法", model_file);
    // 虽然模型文件不存在，但我们仍然返回成功，
    // 因为我们会回退到基于传统方法的分割
    return ESP_OK;
  }

  // 如果已有解释器，先释放
  if (tflite_interpreter) {
    delete tflite_interpreter;
    tflite_interpreter = nullptr;
    model_loaded = false;
  }

  // 创建新的TFLite解释器
  tflite_interpreter = new tflite::MicroInterpreter();
  if (!tflite_interpreter) {
    ESP_LOGE(TAG, "无法创建TFLite解释器，内存不足");
    return ESP_ERR_NO_MEM;
  }

  // 加载模型文件
  ESP_LOGI(TAG, "从文件加载模型: %s", model_file);
  esp_err_t ret = tflite_interpreter->LoadFromFile(model_file);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "无法加载模型文件: %s (%d)", esp_err_to_name(ret), ret);
    delete tflite_interpreter;
    tflite_interpreter = nullptr;
    return ret;
  }

  // 分配张量
  ret = tflite_interpreter->AllocateTensors();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "无法分配张量: %s (%d)", esp_err_to_name(ret), ret);
    delete tflite_interpreter;
    tflite_interpreter = nullptr;
    return ret;
  }

  // 获取模型输入信息
  TfLiteTensor *input_tensor = tflite_interpreter->input(0);
  if (!input_tensor) {
    ESP_LOGE(TAG, "无法获取输入张量");
    delete tflite_interpreter;
    tflite_interpreter = nullptr;
    return ESP_FAIL;
  }

  // 记录模型输入尺寸
  if (input_tensor->dims->size >= 4) {
    // 注意：TFLite通常使用NHWC格式 [batch, height, width, channels]
    model_input_height = input_tensor->dims->data[1];
    model_input_width = input_tensor->dims->data[2];
    model_input_channels = input_tensor->dims->data[3];
  } else {
    ESP_LOGE(TAG, "输入张量维度不正确");
    delete tflite_interpreter;
    tflite_interpreter = nullptr;
    return ESP_FAIL;
  }

  ESP_LOGI(TAG, "模型加载成功. 输入尺寸: %dx%dx%d, 输出类别数: %d",
           model_input_width, model_input_height, model_input_channels,
           model_output_classes);

  model_loaded = true;
  return ESP_OK;
}

// 使用传统方法进行二分类分割
static esp_err_t
traditional_binary_segmentation(camera_fb_t *fb,
                                semantic_segmentation_t *result) {
  ESP_LOGI(TAG, "使用传统方法进行二分类分割");

  // 分配掩码内存
  result->mask = (uint8_t *)heap_caps_malloc(
      fb->width * fb->height, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!result->mask) {
    ESP_LOGE(TAG, "无法分配掩码内存");
    return ESP_ERR_NO_MEM;
  }

  // 设置结果参数
  result->width = fb->width;
  result->height = fb->height;
  result->num_classes = 2; // 可行区和非可行区

  // 分配类别分数内存
  result->class_scores = (float *)malloc(result->num_classes * sizeof(float));
  if (!result->class_scores) {
    free(result->mask);
    result->mask = NULL;
    ESP_LOGE(TAG, "无法分配类别分数内存");
    return ESP_ERR_NO_MEM;
  }

  // 初始化类别分数
  result->class_scores[0] = 0.0f; // 可行区
  result->class_scores[1] = 0.0f; // 非可行区

  // 简单基于颜色的分割算法
  int drivable_count = 0;
  int non_drivable_count = 0;

  // 对于RGB565格式
  if (fb->format == PIXFORMAT_RGB565) {
    for (int y = 0; y < fb->height; y++) {
      for (int x = 0; x < fb->width; x++) {
        uint8_t *pixel = &fb->buf[y * fb->width * 2 + x * 2];
        uint16_t rgb565 = pixel[0] | (pixel[1] << 8);

        // 解析RGB565
        uint8_t r = ((rgb565 >> 11) & 0x1F) << 3;
        uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;
        uint8_t b = (rgb565 & 0x1F) << 3;

        // 简单的道路分割规则 - 亮度高且偏灰/白的区域更可能是道路
        float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
        bool is_bright = luminance > 128;

        // 颜色差异不大的更可能是道路(灰色区域)
        float max_diff = max(max(abs(r - g), abs(r - b)), abs(g - b));
        bool is_gray = max_diff < 50;

        // 位置规则 - 下半部分更可能是道路
        bool is_lower_half = y > fb->height / 2;

        // 综合判断是否为可行区
        bool is_drivable = ((is_bright && is_gray) || is_lower_half);

        // 0表示可行区，1表示非可行区
        result->mask[y * fb->width + x] = is_drivable ? 0 : 1;

        if (is_drivable) {
          drivable_count++;
        } else {
          non_drivable_count++;
        }
      }
    }
  } else {
    // 对于其他格式，默认简单地按照图像下半部分是可行区
    for (int y = 0; y < fb->height; y++) {
      for (int x = 0; x < fb->width; x++) {
        bool is_drivable = y > fb->height * 0.6; // 下60%为可行区
        result->mask[y * fb->width + x] = is_drivable ? 0 : 1;

        if (is_drivable) {
          drivable_count++;
        } else {
          non_drivable_count++;
        }
      }
    }
  }

  // 计算类别分数
  int total_pixels = fb->width * fb->height;
  result->class_scores[0] = (float)drivable_count / total_pixels;
  result->class_scores[1] = (float)non_drivable_count / total_pixels;

  return ESP_OK;
}

// 执行语义分割推理
esp_err_t run_semantic_segmentation(camera_fb_t *fb,
                                    semantic_segmentation_t *result) {
  if (!fb || !result) {
    ESP_LOGE(TAG, "无效输入参数");
    return ESP_ERR_INVALID_ARG;
  }

  // 如果模型未加载，使用传统方法
  if (!model_loaded || !tflite_interpreter) {
    ESP_LOGW(TAG, "TFLite模型未加载，使用传统方法进行分割");
    return traditional_binary_segmentation(fb, result);
  }

  // 记录开始时间
  int64_t start_time = esp_timer_get_time();

  // 获取输入张量
  TfLiteTensor *input_tensor = tflite_interpreter->input(0);
  if (!input_tensor) {
    ESP_LOGE(TAG, "无法获取输入张量");
    return traditional_binary_segmentation(fb, result);
  }

  // 准备输入数据 - 调整图像大小并预处理
  uint8_t *input_data = input_tensor->data.uint8;

  // 将相机帧转换到模型输入尺寸
  for (int y = 0; y < model_input_height; y++) {
    for (int x = 0; x < model_input_width; x++) {
      // 映射到原始图像坐标
      int src_x = x * fb->width / model_input_width;
      int src_y = y * fb->height / model_input_height;

      // 获取像素
      uint8_t r, g, b;

      // 根据图像格式获取RGB值
      if (fb->format == PIXFORMAT_RGB565) {
        // RGB565格式
        uint8_t *pixel = &fb->buf[src_y * fb->width * 2 + src_x * 2];
        uint16_t rgb565 = pixel[0] | (pixel[1] << 8);
        r = ((rgb565 >> 11) & 0x1F) << 3;
        g = ((rgb565 >> 5) & 0x3F) << 2;
        b = (rgb565 & 0x1F) << 3;
      } else if (fb->format == PIXFORMAT_JPEG) {
        // JPEG格式无法直接访问像素，实际应用中应该先解码
        // 这里使用灰度值作为简单替代
        r = g = b = 128;
        ESP_LOGW(TAG, "JPEG格式不支持直接像素访问，使用灰度值代替");
      } else if (fb->format == PIXFORMAT_GRAYSCALE) {
        // 灰度格式
        uint8_t gray = fb->buf[src_y * fb->width + src_x];
        r = g = b = gray;
      } else {
        // 其他格式
        r = g = b = 0;
        ESP_LOGW(TAG, "不支持的图像格式");
      }

      // 填充输入张量 - 假设输入格式为RGB
      int idx = y * model_input_width * model_input_channels +
                x * model_input_channels;
      if (model_input_channels >= 3) {
        input_data[idx] = r;
        input_data[idx + 1] = g;
        input_data[idx + 2] = b;
      } else if (model_input_channels == 1) {
        // 灰度输入
        input_data[idx] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
      }
    }
  }

  // 执行推理
  ESP_LOGI(TAG, "执行语义分割推理");
  esp_err_t ret = tflite_interpreter->Invoke();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "推理失败: %s (%d)", esp_err_to_name(ret), ret);
    return traditional_binary_segmentation(fb, result);
  }

  // 获取输出
  TfLiteTensor *output_tensor = tflite_interpreter->output(0);
  if (!output_tensor) {
    ESP_LOGE(TAG, "无法获取输出张量");
    return traditional_binary_segmentation(fb, result);
  }

  // 分配结果内存
  result->width = fb->width;
  result->height = fb->height;
  result->num_classes = model_output_classes;

  // 分配掩码内存
  result->mask = (uint8_t *)heap_caps_malloc(
      result->width * result->height, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!result->mask) {
    ESP_LOGE(TAG, "无法分配掩码内存");
    return ESP_ERR_NO_MEM;
  }

  // 分配类别分数内存
  result->class_scores =
      (float *)heap_caps_malloc(result->num_classes * sizeof(float),
                                MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);
  if (!result->class_scores) {
    free(result->mask);
    result->mask = NULL;
    ESP_LOGE(TAG, "无法分配类别分数内存");
    return ESP_ERR_NO_MEM;
  }

  // 初始化类别分数
  for (int i = 0; i < result->num_classes; i++) {
    result->class_scores[i] = 0.0f;
  }

  // 获取输出尺寸
  int output_height, output_width, output_channels;
  if (output_tensor->dims->size >= 4) {
    output_height = output_tensor->dims->data[1];
    output_width = output_tensor->dims->data[2];
    output_channels = output_tensor->dims->data[3];
  } else {
    output_height = model_input_height;
    output_width = model_input_width;
    output_channels = model_output_classes;
    ESP_LOGW(TAG, "输出张量维度格式不标准");
  }

  // 解析输出张量 - 为每个像素找到最可能的类别
  int drivable_pixels = 0;
  int non_drivable_pixels = 0;

  for (int y = 0; y < output_height; y++) {
    for (int x = 0; x < output_width; x++) {
  // 计算目标图像中的坐标
int dst_x = x * fb->width / output_width;
            int dst_y = y * fb->height / output_height;
            
            if (dst_x >= 0 && dst_x < fb->width && dst_y >= 0 && dst_y < fb->height) {
                float drivable_score = 0.0f;
                float non_drivable_score = 0.0f;
                
                // 根据输出张量类型获取分数
                if (output_tensor->type == kTfLiteFloat32) {
                    if (output_channels >= 2) {
                        // 多通道输出 - 每个通道是一个类别
                        drivable_score = output_tensor->data.f[y * output_width * output_channels + x * output_channels + 0];
                        non_drivable_score = output_tensor->data.f[y * output_width * output_channels + x * output_channels + 1];
                    } else {
                        // 单通道输出 - 值表示可行区的概率
                        drivable_score = output_tensor->data.f[y * output_width + x];
                        non_drivable_score = 1.0f - drivable_score;
                    }
                } else if (output_tensor->type == kTfLiteUInt8) {
                    if (output_channels >= 2) {
                        drivable_score = output_tensor->data.uint8[y * output_width * output_channels + x * output_channels + 0] / 255.0f;
                        non_drivable_score = output_tensor->data.uint8[y * output_width * output_channels + x * output_channels + 1] / 255.0f;
                    } else {
                        drivable_score = output_tensor->data.uint8[y * output_width + x] / 255.0f;
                        non_drivable_score = 1.0f - drivable_score;
                    }
                }
                
                // 决定像素类别
                bool is_drivable = (drivable_score > non_drivable_score);
                
                // 保存分类结果到掩码 (0=可行区，1=非可行区)
                result->mask[dst_y * fb->width + dst_x] = is_drivable ? 0 : 1;
                
                // 统计像素数量
                if (is_drivable) {
                    drivable_pixels++;
                } else {
                    non_drivable_pixels++;
                }
                
                // 累计类别总分数
                result->class_scores[0] += drivable_score;
                result->class_scores[1] += non_drivable_score;
            }
        }
    }
    
    // 计算平均类别分数
    int total_pixels = fb->width * fb->height;
    result->class_scores[0] /= total_pixels;
    result->class_scores[1] /= total_pixels;
    
    // 计算并打印处理时间
    int64_t end_time = esp_timer_get_time();
    float process_time = (end_time - start_time) / 1000.0; // 转换为毫秒
    ESP_LOGI(TAG, "语义分割完成，处理时间: %.2f ms", process_time);
    ESP_LOGI(TAG, "可行区: %.1f%%, 非可行区: %.1f%%", 
             100.0f * drivable_pixels / total_pixels,
             100.0f * non_drivable_pixels / total_pixels);
    
    return ESP_OK;
}

// 释放语义分割结果
void free_segmentation_result(semantic_segmentation_t* result) {
    if (!result) {
        return;
    }
    
    if (result->mask) {
        free(result->mask);
        result->mask = NULL;
    }
    
    if (result->class_scores) {
        free(result->class_scores);
        result->class_scores = NULL;
    }
    
    result->width = 0;
    result->height = 0;
    result->num_classes = 0;
}

// 加载指定的语义分割模型文件
esp_err_t load_segmentation_model(const char* model_path, seg_model_type_t model_type) {
    if (!model_path) {
        ESP_LOGE(TAG, "无效的模型路径");
        return ESP_ERR_INVALID_ARG;
    }

    // 更新模型路径
    model_file = model_path;
    
    ESP_LOGI(TAG, "设置模型路径: %s, 类型: %d", model_file, model_type);
    
    // 重新初始化模型
    return init_semantic_segmentation();
}

// 获取模型信息
esp_err_t get_segmentation_model_info(int* input_width, int* input_height, int* num_classes) {
    if (input_width) {
        *input_width = model_loaded ? model_input_width : 0;
    }
    
    if (input_height) {
        *input_height = model_loaded ? model_input_height : 0;
    }
    
    if (num_classes) {
        *num_classes = model_output_classes;
    }
    
    return ESP_OK;
}

// 获取类别信息
const seg_class_info_t* get_class_info(int class_id) {
    for (int i = 0; i < num_class_info; i++) {
        if (class_info[i].id == class_id) {
            return &class_info[i];
        }
    }
    
    // 默认返回可行区类别
    return &class_info[0];
}

// 设置推理参数
esp_err_t set_segmentation_params(float conf_threshold, bool enable_preproc) {
    // 验证阈值范围
    if (conf_threshold < 0.0f || conf_threshold > 1.0f) {
        ESP_LOGW(TAG, "无效的阈值（应当在0-1之间），使用默认值0.5");
        confidence_threshold = 0.5f;
    } else {
        confidence_threshold = conf_threshold;
    }
    
    enable_preprocessing = enable_preproc;
    
    ESP_LOGI(TAG, "设置新参数：阈值=%.2f，预处理=%s", 
             confidence_threshold, enable_preprocessing ? "启用" : "禁用");
             
    return ESP_OK;
}