#include "object_detect.h"

// 构造函数
ObjectDetect::ObjectDetect()
{
    ESP_LOGI(TAG, "初始化目标检测模块");

    // 初始化成员变量
    interpreter = nullptr;
    model_loaded = false;
    orig_width = 0;
    orig_height = 0;

    // 配置YOLOv5-nano模型参数
    yolo_config.input_width = 416;
    yolo_config.input_height = 416;
    yolo_config.num_classes = 2;        // BDD100K数据集的2个类别
    yolo_config.num_anchors = 3;        // 每个特征图3个锚点
    yolo_config.conf_threshold = 0.25f; // 置信度阈值
    yolo_config.nms_threshold = 0.45f;  // NMS阈值

    // YOLOv5-nano的步长
    yolo_config.strides = {8, 16, 32};

    // YOLOv5-nano的锚点(按stride分组)
    yolo_config.anchors = {
        {10, 13, 16, 30, 33, 23},     // 小目标(stride=8)
        {30, 61, 62, 45, 59, 119},    // 中目标(stride=16)
        {116, 90, 156, 198, 373, 326} // 大目标(stride=32)
    };

    // BDD100K分类(二分类版本)
    class_names = {"active", "traffic sign"};
}

// 析构函数
ObjectDetect::~ObjectDetect()
{
    // 释放模型资源
    if (interpreter != nullptr)
    {
        delete interpreter;
        interpreter = nullptr;
    }
}

// 设置置信度阈值
void ObjectDetect::setConfThreshold(float threshold)
{
    yolo_config.conf_threshold = threshold;
    ESP_LOGI(TAG, "置信度阈值设置为: %.2f", threshold);
}

// 设置NMS阈值
void ObjectDetect::setNMSThreshold(float threshold)
{
    yolo_config.nms_threshold = threshold;
    ESP_LOGI(TAG, "NMS阈值设置为: %.2f", threshold);
}

// 加载模型
bool ObjectDetect::loadModel()
{
    ESP_LOGI(TAG, "正在加载YOLOv5模型: %s", DETECTION_MODEL);

    // 记录开始时间
    int64_t start_time = esp_timer_get_time();

    // 释放已有资源
    if (interpreter != nullptr)
    {
        delete interpreter;
        interpreter = nullptr;
    }

    // 配置运算符解析器(添加YOLO需要的所有运算符)
    static tflite::MicroMutableOpResolver<14> resolver;

    // 添加YOLO所需的所有操作
    resolver.AddConv2D();
    resolver.AddPad();
    resolver.AddLogistic();
    resolver.AddMul();
    resolver.AddConcatenation();
    resolver.AddAdd();
    resolver.AddMaxPool2D();
    resolver.AddQuantize();
    resolver.AddResizeNearestNeighbor();
    resolver.AddTranspose();
    resolver.AddReshape();
    resolver.AddStridedSlice();
    resolver.AddSoftmax();
    resolver.AddSub();

    // 从SD卡加载模型
    FILE *f = fopen(DETECTION_MODEL, "rb");
    if (!f)
    {
        ESP_LOGE(TAG, "无法打开模型文件: %s", DETECTION_MODEL);
        return false;
    }

    // 获取文件大小
    fseek(f, 0, SEEK_END);
    long model_size = ftell(f);
    model_size = (model_size + 3) & ~3; // 对齐到4字节
    model_size *= 1.1;                  // 预留10%空间
    fseek(f, 0, SEEK_SET);

    ESP_LOGI(TAG, "模型大小: %ld 字节", model_size);

    // 分配内存加载模型
    uint8_t *model_data = (uint8_t *)heap_caps_malloc(model_size, MALLOC_CAP_SPIRAM);
    if (!model_data)
    {
        ESP_LOGE(TAG, "无法分配内存加载模型: %ld 字节", model_size);
        fclose(f);
        return false;
    }

    // 读取模型数据
    size_t bytes_read = fread(model_data, 1, model_size, f);
    fclose(f);

    if (bytes_read != model_size)
    {
        ESP_LOGE(TAG, "模型文件读取不完整: %d/%ld", bytes_read, model_size);
        free(model_data);
        return false;
    }

    // 获取模型
    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "模型版本不匹配: %d vs %d", model->version(), TFLITE_SCHEMA_VERSION);
        free(model_data);
        return false;
    }

    // 分配张量区域(4MB)
    static uint8_t tensor_arena[3 * 1024 * 1024] __attribute__((aligned(16)));

    // 创建解释器
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, sizeof(tensor_arena));

    // 分配张量
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        ESP_LOGE(TAG, "张量分配失败");
        delete interpreter;
        interpreter = nullptr;
        free(model_data);
        return false;
    }

    // 获取输入张量并检查尺寸
    TfLiteTensor *input_tensor = interpreter->input(0);

    // 检查输入维度和类型
    if (input_tensor->type != kTfLiteFloat32)
    {
        ESP_LOGE(TAG, "输入张量类型不支持: %d (需要Float32)", input_tensor->type);
        delete interpreter;
        interpreter = nullptr;
        free(model_data);
        return false;
    }

    // 获取输入尺寸并更新配置
    if (input_tensor->dims->size == 4)
    {
        yolo_config.input_height = input_tensor->dims->data[1];
        yolo_config.input_width = input_tensor->dims->data[2];
        int channels = input_tensor->dims->data[3];

        ESP_LOGI(TAG, "模型输入尺寸: %dx%dx%d",
                 yolo_config.input_height, yolo_config.input_width, channels);
    }
    else
    {
        ESP_LOGE(TAG, "不支持的输入维度: %d (需要4D)", input_tensor->dims->size);
        delete interpreter;
        interpreter = nullptr;
        free(model_data);
        return false;
    }

    // 记录加载时间
    int64_t end_time = esp_timer_get_time();
    ESP_LOGI(TAG, "模型加载完成，耗时: %lld ms", (end_time - start_time) / 1000);

    model_loaded = true;
    return true;
}

// 图像预处理 - 调整大小并归一化到[0,1]
void ObjectDetect::preprocess(uint8_t *input, int width, int height, int channels,
                              float *output, int target_w, int target_h)
{
    // 记录原始尺寸
    orig_width = width;
    orig_height = height;

    // 计算缩放比例
    float scale_w = static_cast<float>(width) / target_w;
    float scale_h = static_cast<float>(height) / target_h;

    // 遍历目标图像的每个像素
    for (int y = 0; y < target_h; y++)
    {
        for (int x = 0; x < target_w; x++)
        {
            // 映射回原图坐标
            int src_x = std::min(static_cast<int>(x * scale_w), width - 1);
            int src_y = std::min(static_cast<int>(y * scale_h), height - 1);

            // 计算原图像素索引
            int src_idx = (src_y * width + src_x) * channels;

            // 计算目标像素索引(CHW格式)
            int dst_idx = y * target_w + x;

            // 对每个通道进行处理
            for (int c = 0; c < 3; c++)
            {
                float pixel_value = 0.0f;

                // 如果原图是RGB，直接获取对应通道
                if (channels >= 3)
                {
                    pixel_value = static_cast<float>(input[src_idx + c]);
                }
                // 如果原图是灰度，则复制到所有通道
                else if (channels == 1)
                {
                    pixel_value = static_cast<float>(input[src_y * width + src_x]);
                }

                // 归一化到[0,1]并写入输出
                output[c * target_h * target_w + dst_idx] = pixel_value / 255.0f;
            }
        }
    }
}

// 检测函数 - 暴露给外部的主要接口
std::vector<Detection> ObjectDetect::detect(uint8_t *imageData, int width, int height, int channels)
{
    // 检测结果
    std::vector<Detection> detections;

    // 检查模型是否已加载
    if (!model_loaded || interpreter == nullptr)
    {
        ESP_LOGW(TAG, "模型未加载，无法进行检测");
        return detections;
    }

    // 记录开始时间
    int64_t start_time = esp_timer_get_time();

    // 获取输入尺寸
    int input_w = yolo_config.input_width;
    int input_h = yolo_config.input_height;

    // 获取输入张量
    TfLiteTensor *input_tensor = interpreter->input(0);

    // 预处理图像
    preprocess(imageData, width, height, channels,
               reinterpret_cast<float *>(input_tensor->data.f),
               input_w, input_h);

    // 执行推理
    ESP_LOGI(TAG, "开始执行模型推理...");
    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk)
    {
        ESP_LOGE(TAG, "模型推理失败");
        return detections;
    }

    // 获取输出张量
    TfLiteTensor *output_tensor = interpreter->output(0);

    // 解码检测结果
    detections = decodeOutputs(output_tensor);

    // 应用NMS去除重叠框
    detections = applyNMS(detections);

    // 将归一化坐标映射回原始图像尺寸
    mapToOriginalSize(detections, width, height);

    // 记录结束时间并计算耗时
    int64_t end_time = esp_timer_get_time();
    ESP_LOGI(TAG, "检测完成，耗时: %lld ms, 检测到 %d 个目标",
             (end_time - start_time) / 1000, detections.size());

    // 输出检测结果日志
    for (size_t i = 0; i < detections.size(); i++)
    {
        const auto &det = detections[i];
        const char *class_name = det.classId < class_names.size() ? class_names[det.classId].c_str() : "未知";

        ESP_LOGI(TAG, "  目标 %d: 类别=%s, 置信度=%.2f, 位置=[%.1f, %.1f, %.1f, %.1f]",
                 i, class_name, det.confidence,
                 det.x, det.y, det.width, det.height);
    }

    return detections;
}

// 解码YOLOv5-nano输出张量
std::vector<Detection> ObjectDetect::decodeOutputs(TfLiteTensor *output_tensor)
{
    std::vector<Detection> detections;

    // 获取输出维度
    int output_size = 1;
    for (int i = 0; i < output_tensor->dims->size; i++)
    {
        output_size *= output_tensor->dims->data[i];
    }

    // YOLOv5输出格式为 [batch, num_predictions, num_classes + 5]
    int num_classes = yolo_config.num_classes;
    int items_per_box = num_classes + 5; // x, y, w, h, conf, class_probs...
    int num_boxes = output_size / items_per_box;

    float *output_data = output_tensor->data.f;

    // 处理每个预测框
    for (int i = 0; i < num_boxes; i++)
    {
        // 获取当前框的起始位置
        float *box_data = &output_data[i * items_per_box];

        // 获取置信度
        float confidence = box_data[4];

        // 仅处理置信度高于阈值的框
        if (confidence > yolo_config.conf_threshold)
        {
            // 找出最高类别概率及其索引
            float max_class_prob = 0;
            int max_class_idx = 0;

            for (int c = 0; c < num_classes; c++)
            {
                float class_prob = box_data[5 + c];
                if (class_prob > max_class_prob)
                {
                    max_class_prob = class_prob;
                    max_class_idx = c;
                }
            }

            // 计算最终置信度
            float final_confidence = confidence * max_class_prob;

            // 仍超过阈值则添加到检测结果
            if (final_confidence > yolo_config.conf_threshold)
            {
                Detection det;

                // 设置类别和置信度
                det.classId = max_class_idx;
                det.confidence = final_confidence;

                // 解析边界框坐标(注意YOLOv5输出的是归一化中心点坐标和宽高)
                float center_x = box_data[0];
                float center_y = box_data[1];
                float width = box_data[2];
                float height = box_data[3];

                // 转换为左上角坐标和宽高表示法(此时都是归一化到0-1的)
                det.x = center_x - width / 2.0f;
                det.y = center_y - height / 2.0f;
                det.width = width;
                det.height = height;

                // 限制在[0,1]范围内
                det.x = std::max(0.0f, std::min(1.0f, det.x));
                det.y = std::max(0.0f, std::min(1.0f, det.y));
                det.width = std::max(0.0f, std::min(1.0f, det.width));
                det.height = std::max(0.0f, std::min(1.0f, det.height));

                detections.push_back(det);
            }
        }
    }

    return detections;
}

// 计算两个边界框的交并比(IoU)
float ObjectDetect::calculateIoU(const Detection &a, const Detection &b)
{
    // 计算交集矩形
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    // 交集面积
    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    // 两个框的面积
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;

    // 并集面积
    float union_area = area_a + area_b - intersection_area;

    // 计算IoU
    if (union_area > 0)
        return intersection_area / union_area;
    return 0;
}

// 应用非极大值抑制(NMS)
std::vector<Detection> ObjectDetect::applyNMS(std::vector<Detection> &detections)
{
    std::vector<Detection> result;

    // 按置信度排序(从高到低)
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b)
              {
                  return a.confidence > b.confidence;
              });

    // 标记是否被抑制
    std::vector<bool> suppressed(detections.size(), false);

    // NMS处理
    for (size_t i = 0; i < detections.size(); i++)
    {
        // 如果当前框已被抑制，跳过
        if (suppressed[i])
            continue;

        // 否则添加到结果
        result.push_back(detections[i]);

        // 抑制所有与当前框IoU大于阈值的后续框
        for (size_t j = i + 1; j < detections.size(); j++)
        {
            // 只对同类别的框进行NMS
            if (detections[i].classId == detections[j].classId)
            {
                if (calculateIoU(detections[i], detections[j]) > yolo_config.nms_threshold)
                {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

// 将归一化坐标映射回原始图像尺寸
void ObjectDetect::mapToOriginalSize(std::vector<Detection> &detections,
                                     int orig_width, int orig_height)
{
    for (auto &det : detections)
    {
        det.x *= orig_width;
        det.y *= orig_height;
        det.width *= orig_width;
        det.height *= orig_height;
    }
}
