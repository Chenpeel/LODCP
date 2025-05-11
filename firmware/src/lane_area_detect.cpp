#include "lane_area_detect.h"
#include "driver/sdmmc_host.h"
#include "driver/sdspi_host.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_tflite.h"
#include "esp_spiffs.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdmmc_cmd.h"
#include "common_types.h"

// 标签定义，用于日志输出
static const char *TAG = "LaneDetector";

// 颜色阈值
const color_threshold_t WHITE_THRESHOLD = {0, 180, 0, 30, 200, 255};
const color_threshold_t YELLOW_THRESHOLD = {20, 30, 100, 255, 100, 255};
// 定义静态成员变量
bool LaneAreaDetect::resolver_initialized = false;

// 释放处理后的图像资源
static void free_processed_image(processed_image_t *img)
{
  if (img && img->data)
  {
    free(img->data);
    img->data = NULL;
    img->width = 0;
    img->height = 0;
    img->channels = 0;
  }
}

// 创建空白处理图像
static esp_err_t create_processed_image(processed_image_t *img, int width,
                                        int height, int channels)
{
  if (!img)
    return ESP_FAIL;

  size_t size = width * height * channels;
  img->data =
      (uint8_t *)heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!img->data)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for processed image");
    return ESP_ERR_NO_MEM;
  }

  img->width = width;
  img->height = height;
  img->channels = channels;
  memset(img->data, 0, size);

  return ESP_OK;
}

// 复制图像
static esp_err_t copy_processed_image(processed_image_t *dst,
                                      const processed_image_t *src)
{
  if (!dst || !src)
    return ESP_FAIL;

  if (dst->data)
  {
    free_processed_image(dst);
  }

  esp_err_t err =
      create_processed_image(dst, src->width, src->height, src->channels);
  if (err != ESP_OK)
    return err;

  memcpy(dst->data, src->data, src->width * src->height * src->channels);
  return ESP_OK;
}

// 构造函数
LaneAreaDetect::LaneAreaDetect()
{
  // 初始化形态学内核
  memset(kernel_large, 1, sizeof(kernel_large));

  // 初始化多项式拟合参数
  left_lane_fit.a = 0;
  left_lane_fit.b = 0;
  left_lane_fit.c = 0;
  left_lane_fit.valid = false;

  right_lane_fit.a = 0;
  right_lane_fit.b = 0;
  right_lane_fit.c = 0;
  right_lane_fit.valid = false;

  // 初始化线段存储
  left_lines = nullptr;
  right_lines = nullptr;
  left_line_count = 0;
  right_line_count = 0;

  // 初始化FastSCNN模型
  model_loaded = false;
  interpreter = nullptr;
  model_input_width = 0;
  model_input_height = 0;
}

// 析构函数
LaneAreaDetect::~LaneAreaDetect()
{
  // 释放模型资源
  if (interpreter != nullptr)
  {
    delete interpreter;
    interpreter = nullptr;
  }
  if (tensor_arena != nullptr)
  {
    free(tensor_arena);
    tensor_arena = nullptr;
  }
  if (model_data != nullptr)
  {
    free(model_data);
    model_data = nullptr;
  }
}

// 颜色分割 - 使用HSV分割黄色和白色
esp_err_t LaneAreaDetect::color_regmentation(camera_fb_t *fb,
                                             processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 创建输出图像 (单通道二值图像)
  esp_err_t err = create_processed_image(output, fb->width, fb->height, 1);
  if (err != ESP_OK)
    return err;

  // 临时HSV缓冲区
  uint8_t hsv[3];

  // 遍历每个像素，对于RGB565格式需要转换
  for (int y = 0; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      // 取得像素值并转换为HSV
      uint8_t *pixel = &fb->buf[y * fb->width * 2 + x * 2];

      // RGB565 -> RGB888
      uint16_t rgb565 = pixel[0] | (pixel[1] << 8);
      uint8_t r = ((rgb565 >> 11) & 0x1F) << 3;
      uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;
      uint8_t b = (rgb565 & 0x1F) << 3;

      // RGB -> HSV (简化版)
      // 这里使用简单转换，也可以用查找表优化
      float r_norm = r / 255.0f;
      float g_norm = g / 255.0f;
      float b_norm = b / 255.0f;

      float max_val = fmax(fmax(r_norm, g_norm), b_norm);
      float min_val = fmin(fmin(r_norm, g_norm), b_norm);
      float delta = max_val - min_val;

      // H calculation
      float h = 0;
      if (delta != 0)
      {
        if (max_val == r_norm)
        {
          h = 60.0f * fmodf((g_norm - b_norm) / delta, 6.0f);
        }
        else if (max_val == g_norm)
        {
          h = 60.0f * ((b_norm - r_norm) / delta + 2.0f);
        }
        else
        {
          h = 60.0f * ((r_norm - g_norm) / delta + 4.0f);
        }

        if (h < 0)
          h += 360.0f;
      }

      // S and V calculation
      float s = (max_val == 0) ? 0 : (delta / max_val);
      float v = max_val;

      // 归一化到0-255
      hsv[0] = (uint8_t)(h / 2.0f);   // H: 0-180 (OpenCV风格)
      hsv[1] = (uint8_t)(s * 255.0f); // S: 0-255
      hsv[2] = (uint8_t)(v * 255.0f); // V: 0-255

      // 判断是否为黄色或白色
      bool is_yellow = (hsv[0] >= YELLOW_THRESHOLD.h_min &&
                        hsv[0] <= YELLOW_THRESHOLD.h_max &&
                        hsv[1] >= YELLOW_THRESHOLD.s_min &&
                        hsv[1] <= YELLOW_THRESHOLD.s_max &&
                        hsv[2] >= YELLOW_THRESHOLD.v_min &&
                        hsv[2] <= YELLOW_THRESHOLD.v_max);

      bool is_white =
          (hsv[0] >= WHITE_THRESHOLD.h_min && hsv[0] <= WHITE_THRESHOLD.h_max &&
           hsv[1] >= WHITE_THRESHOLD.s_min && hsv[1] <= WHITE_THRESHOLD.s_max &&
           hsv[2] >= WHITE_THRESHOLD.v_min && hsv[2] <= WHITE_THRESHOLD.v_max);

      // 设置输出值
      output->data[y * fb->width + x] = (is_yellow || is_white) ? 255 : 0;
    }
  }

  ESP_LOGI(TAG, "Step 1: 颜色分割完成");
  return ESP_OK;
}

// 转换图像为灰度
esp_err_t LaneAreaDetect::color_2_gray(camera_fb_t *fb,
                                       processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 创建输出图像 (单通道灰度图像)
  esp_err_t err = create_processed_image(output, fb->width, fb->height, 1);
  if (err != ESP_OK)
    return err;

  // 对于RGB565格式图像
  for (int y = 0; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      // 获取RGB565像素值
      uint8_t *pixel = &fb->buf[y * fb->width * 2 + x * 2];
      uint16_t rgb565 = pixel[0] | (pixel[1] << 8);

      // 转换为RGB
      uint8_t r = ((rgb565 >> 11) & 0x1F) << 3;
      uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;
      uint8_t b = (rgb565 & 0x1F) << 3;

      // 计算灰度值 (使用标准BT.601权重)
      uint8_t gray = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);

      // 设置输出值
      output->data[y * fb->width + x] = gray;
    }
  }

  ESP_LOGI(TAG, "Step 2: 颜色转灰度完成");
  return ESP_OK;
}

// 多尺度差分高斯边缘检测
esp_err_t LaneAreaDetect::multi_scale_DoG(camera_fb_t *fb,
                                          processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 首先转为灰度图像
  processed_image_t gray_img = {0};
  esp_err_t err = color_2_gray(fb, &gray_img);
  if (err != ESP_OK)
    return err;

  // 创建输出图像 (单通道二值图)
  err = create_processed_image(output, fb->width, fb->height, 1);
  if (err != ESP_OK)
  {
    free_processed_image(&gray_img);
    return err;
  }

  // 临时图像缓冲区
  uint8_t *g1 = (uint8_t *)malloc(fb->width * fb->height);
  uint8_t *g2 = (uint8_t *)malloc(fb->width * fb->height);
  int16_t *dog = (int16_t *)malloc(fb->width * fb->height * sizeof(int16_t));
  uint8_t *dog_norm = (uint8_t *)malloc(fb->width * fb->height);

  if (!g1 || !g2 || !dog || !dog_norm)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for DoG");
    if (g1)
      free(g1);
    if (g2)
      free(g2);
    if (dog)
      free(dog);
    if (dog_norm)
      free(dog_norm);
    free_processed_image(&gray_img);
    return ESP_ERR_NO_MEM;
  }

  // 初始化输出图像为黑色
  memset(output->data, 0, fb->width * fb->height);

  // 多尺度DoG参数 (sigma对)
  float scales[][2] = {{1.0f, 2.0f}, {2.0f, 4.0f}, {3.0f, 6.0f}};
  const int num_scales = 3;

  // 对每对sigma值应用DoG
  for (int s = 0; s < num_scales; s++)
  {
    float sigma1 = scales[s][0];
    float sigma2 = scales[s][1];

    // 计算高斯核大小
    int k_size1 = (int)(6.0f * sigma1 + 1) | 1; // 确保为奇数
    int k_size2 = (int)(6.0f * sigma2 + 1) | 1;

    // 简化版高斯模糊 (可以用更精确的高斯核)
    // 这里使用方框模糊多次迭代来近似高斯模糊
    memcpy(g1, gray_img.data, fb->width * fb->height);
    memcpy(g2, gray_img.data, fb->width * fb->height);

    // 应用方框模糊多次来近似高斯模糊
    int iterations1 = (int)(sigma1 * 2);
    int iterations2 = (int)(sigma2 * 2);

    // g1模糊 (使用简化版方框模糊)
    for (int iter = 0; iter < iterations1; iter++)
    {
      uint8_t *temp = (uint8_t *)malloc(fb->width * fb->height);
      if (!temp)
      {
        ESP_LOGE(TAG, "Failed to allocate temp memory");
        free(g1);
        free(g2);
        free(dog);
        free(dog_norm);
        free_processed_image(&gray_img);
        return ESP_ERR_NO_MEM;
      }

      // 水平方向模糊
      for (int y = 0; y < fb->height; y++)
      {
        for (int x = 0; x < fb->width; x++)
        {
          int sum = 0, count = 0;
          for (int i = -1; i <= 1; i++)
          {
            int nx = x + i;
            if (nx >= 0 && nx < fb->width)
            {
              sum += g1[y * fb->width + nx];
              count++;
            }
          }
          temp[y * fb->width + x] = sum / count;
        }
      }

      // 垂直方向模糊
      for (int y = 0; y < fb->height; y++)
      {
        for (int x = 0; x < fb->width; x++)
        {
          int sum = 0, count = 0;
          for (int j = -1; j <= 1; j++)
          {
            int ny = y + j;
            if (ny >= 0 && ny < fb->height)
            {
              sum += temp[ny * fb->width + x];
              count++;
            }
          }
          g1[y * fb->width + x] = sum / count;
        }
      }

      free(temp);
    }

    // g2模糊 (类似g1但迭代次数不同)
    for (int iter = 0; iter < iterations2; iter++)
    {
      uint8_t *temp = (uint8_t *)malloc(fb->width * fb->height);
      if (!temp)
      {
        ESP_LOGE(TAG, "Failed to allocate temp memory");
        free(g1);
        free(g2);
        free(dog);
        free(dog_norm);
        free_processed_image(&gray_img);
        return ESP_ERR_NO_MEM;
      }

      // 水平方向模糊
      for (int y = 0; y < fb->height; y++)
      {
        for (int x = 0; x < fb->width; x++)
        {
          int sum = 0, count = 0;
          for (int i = -1; i <= 1; i++)
          {
            int nx = x + i;
            if (nx >= 0 && nx < fb->width)
            {
              sum += g2[y * fb->width + nx];
              count++;
            }
          }
          temp[y * fb->width + x] = sum / count;
        }
      }

      // 垂直方向模糊
      for (int y = 0; y < fb->height; y++)
      {
        for (int x = 0; x < fb->width; x++)
        {
          int sum = 0, count = 0;
          for (int j = -1; j <= 1; j++)
          {
            int ny = y + j;
            if (ny >= 0 && ny < fb->height)
            {
              sum += temp[ny * fb->width + x];
              count++;
            }
          }
          g2[y * fb->width + x] = sum / count;
        }
      }

      free(temp);
    }

    // 计算两个高斯模糊图像的差分
    int16_t min_val = INT16_MAX;
    int16_t max_val = INT16_MIN;
    for (int i = 0; i < fb->width * fb->height; i++)
    {
      dog[i] = (int16_t)g1[i] - (int16_t)g2[i];
      if (dog[i] < min_val)
        min_val = dog[i];
      if (dog[i] > max_val)
        max_val = dog[i];
    }

    // 归一化差分图像到0-255
    for (int i = 0; i < fb->width * fb->height; i++)
    {
      dog_norm[i] =
          (uint8_t)(255.0f * (dog[i] - min_val) / (max_val - min_val));
    }

    // 基于OTSU阈值进行二值化
    // 简化版OTSU (可以改进为真正的OTSU)
    int histogram[256] = {0};
    for (int i = 0; i < fb->width * fb->height; i++)
    {
      histogram[dog_norm[i]]++;
    }

    float sum = 0;
    for (int i = 0; i < 256; i++)
    {
      sum += i * histogram[i];
    }

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float varMax = 0;
    int threshold = 0;

    for (int i = 0; i < 256; i++)
    {
      wB += histogram[i];
      if (wB == 0)
        continue;

      wF = fb->width * fb->height - wB;
      if (wF == 0)
        break;

      sumB += i * histogram[i];
      float mB = sumB / wB;
      float mF = (sum - sumB) / wF;

      float varBetween = wB * wF * (mB - mF) * (mB - mF);
      if (varBetween > varMax)
      {
        varMax = varBetween;
        threshold = i;
      }
    }

    // 应用阈值
    for (int i = 0; i < fb->width * fb->height; i++)
    {
      if (dog_norm[i] > threshold)
      {
        output->data[i] = 255; // 直接在输出图像上进行OR操作
      }
    }
  }

  // 应用3x3中值滤波去除噪声
  uint8_t *filtered = (uint8_t *)malloc(fb->width * fb->height);
  if (filtered)
  {
    for (int y = 0; y < fb->height; y++)
    {
      for (int x = 0; x < fb->width; x++)
      {
        uint8_t values[9];
        int idx = 0;

        // 收集3x3邻域内的值
        for (int j = -1; j <= 1; j++)
        {
          for (int i = -1; i <= 1; i++)
          {
            int nx = x + i;
            int ny = y + j;

            if (nx >= 0 && nx < fb->width && ny >= 0 && ny < fb->height)
            {
              values[idx++] = output->data[ny * fb->width + nx];
            }
            else
            {
              values[idx++] = 0;
            }
          }
        }

        // 简单排序找中值
        for (int i = 0; i < 9 - 1; i++)
        {
          for (int j = 0; j < 9 - i - 1; j++)
          {
            if (values[j] > values[j + 1])
            {
              uint8_t temp = values[j];
              values[j] = values[j + 1];
              values[j + 1] = temp;
            }
          }
        }

        filtered[y * fb->width + x] = values[4]; // 中值
      }
    }

    // 复制结果到输出
    memcpy(output->data, filtered, fb->width * fb->height);
    free(filtered);
  }

  // 清理临时缓冲区
  free(g1);
  free(g2);
  free(dog);
  free(dog_norm);
  free_processed_image(&gray_img);

  ESP_LOGI(TAG, "Multi-scale DoG completed");
  return ESP_OK;
}

// 边缘增强 - 结合顶帽变换和形态学处理
esp_err_t LaneAreaDetect::enhance_edges(camera_fb_t *fb,
                                        processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 首先获取边缘或灰度图像
  processed_image_t edges = {0};
  processed_image_t gray = {0};

  esp_err_t err = color_2_gray(fb, &gray);
  if (err != ESP_OK)
    return err;

  err = multi_scale_DoG(fb, &edges);
  if (err != ESP_OK)
  {
    free_processed_image(&gray);
    return err;
  }

  // 创建输出图像
  err = create_processed_image(output, fb->width, fb->height, 1);
  if (err != ESP_OK)
  {
    free_processed_image(&edges);
    free_processed_image(&gray);
    return err;
  }

  // 临时缓冲区
  uint8_t *temp = (uint8_t *)malloc(fb->width * fb->height);
  uint8_t *temp2 = (uint8_t *)malloc(fb->width * fb->height);
  uint8_t *tophat = (uint8_t *)malloc(fb->width * fb->height);

  if (!temp || !temp2 || !tophat)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for edge enhancement");
    if (temp)
      free(temp);
    if (temp2)
      free(temp2);
    if (tophat)
      free(tophat);
    free_processed_image(&edges);
    free_processed_image(&gray);
    return ESP_ERR_NO_MEM;
  }

  // 应用顶帽变换
  // 1. 首先对灰度图进行膨胀
  memcpy(temp, gray.data, fb->width * fb->height);

  // 大内核膨胀 (15x15)
  for (int y = 0; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      uint8_t max_val = 0;
      for (int j = -7; j <= 7; j++)
      {
        for (int i = -7; i <= 7; i++)
        {
          int nx = x + i;
          int ny = y + j;
          if (nx >= 0 && nx < fb->width && ny >= 0 && ny < fb->height)
          {
            if (gray.data[ny * fb->width + nx] > max_val)
            {
              max_val = gray.data[ny * fb->width + nx];
            }
          }
        }
      }
      temp2[y * fb->width + x] = max_val;
    }
  }

  // 2. 原图减去膨胀结果得到顶帽
  for (int i = 0; i < fb->width * fb->height; i++)
  {
    tophat[i] = (gray.data[i] > temp2[i]) ? (gray.data[i] - temp2[i]) : 0;
  }

  // 二值化边缘图像
  for (int i = 0; i < fb->width * fb->height; i++)
  {
    temp[i] = (edges.data[i] > 100) ? 255 : 0;
  }

  // 应用闭运算连接断开的线 (5x5内核)
  // 1. 膨胀
  memcpy(temp2, temp, fb->width * fb->height);
  for (int y = 0; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      bool dilate = false;
      for (int j = -2; j <= 2 && !dilate; j++)
      {
        for (int i = -2; i <= 2 && !dilate; i++)
        {
          int nx = x + i;
          int ny = y + j;
          if (nx >= 0 && nx < fb->width && ny >= 0 && ny < fb->height)
          {
            if (temp[ny * fb->width + nx] > 0)
            {
              dilate = true;
            }
          }
        }
      }
      temp2[y * fb->width + x] = dilate ? 255 : 0;
    }
  }

  // 2. 腐蚀 (两次迭代)
  memcpy(temp, temp2, fb->width * fb->height);
  for (int iter = 0; iter < 2; iter++)
  {
    memcpy(temp2, temp, fb->width * fb->height);
    for (int y = 0; y < fb->height; y++)
    {
      for (int x = 0; x < fb->width; x++)
      {
        bool erode = true;
        for (int j = -2; j <= 2 && erode; j++)
        {
          for (int i = -2; i <= 2 && erode; i++)
          {
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < fb->width && ny >= 0 && ny < fb->height)
            {
              if (temp[ny * fb->width + nx] == 0)
              {
                erode = false;
              }
            }
          }
        }
        temp2[y * fb->width + x] = erode ? 255 : 0;
      }
    }
    memcpy(temp, temp2, fb->width * fb->height);
  }

  // 应用开运算去除噪点 (3x3内核)
  // 1. 腐蚀
  memcpy(temp2, temp, fb->width * fb->height);
  for (int y = 0; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      bool erode = true;
      for (int j = -1; j <= 1 && erode; j++)
      {
        for (int i = -1; i <= 1 && erode; i++)
        {
          int nx = x + i;
          int ny = y + j;
          if (nx >= 0 && nx < fb->width && ny >= 0 && ny < fb->height)
          {
            if (temp[ny * fb->width + nx] == 0)
            {
              erode = false;
            }
          }
        }
      }
      temp2[y * fb->width + x] = erode ? 255 : 0;
    }
  }

  // 2. 膨胀
  memcpy(temp, temp2, fb->width * fb->height);
  for (int y = 0; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      bool dilate = false;
      for (int j = -1; j <= 1 && !dilate; j++)
      {
        for (int i = -1; i <= 1 && !dilate; i++)
        {
          int nx = x + i;
          int ny = y + j;
          if (nx >= 0 && nx < fb->width && ny >= 0 && ny < fb->height)
          {
            if (temp[ny * fb->width + nx] > 0)
            {
              dilate = true;
            }
          }
        }
      }
      temp2[y * fb->width + x] = dilate ? 255 : 0;
    }
  }

  // 最后再膨胀一次完善车道线
  memcpy(temp, temp2, fb->width * fb->height);
  for (int y = 0; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      bool dilate = false;
      for (int j = -1; j <= 1 && !dilate; j++)
      {
        for (int i = -1; i <= 1 && !dilate; i++)
        {
          int nx = x + i;
          int ny = y + j;
          if (nx >= 0 && nx < fb->width && ny >= 0 && ny < fb->height)
          {
            if (temp[ny * fb->width + nx] > 0)
            {
              dilate = true;
            }
          }
        }
      }
      temp2[y * fb->width + x] = dilate ? 255 : 0;
    }
  }

  // 二值化顶帽结果
  for (int i = 0; i < fb->width * fb->height; i++)
  {
    tophat[i] = (tophat[i] > 30) ? 255 : 0;
  }

  // 结合边缘和顶帽结果
  for (int i = 0; i < fb->width * fb->height; i++)
  {
    output->data[i] = (temp2[i] > 0 || tophat[i] > 0) ? 255 : 0;
  }

  // 清理
  free(temp);
  free(temp2);
  free(tophat);
  free_processed_image(&edges);
  free_processed_image(&gray);

  ESP_LOGI(TAG, "Edge enhancement completed");
  return ESP_OK;
}

// 应用ROI掩码
esp_err_t LaneAreaDetect::ROI_mask(camera_fb_t *fb, processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 先获取边缘图像
  processed_image_t edges = {0};
  esp_err_t err = enhance_edges(fb, &edges);
  if (err != ESP_OK)
    return err;

  // 创建输出图像
  err = create_processed_image(output, fb->width, fb->height, 1);
  if (err != ESP_OK)
  {
    free_processed_image(&edges);
    return err;
  }

  // 创建ROI掩码
  uint8_t *roi_mask =
      (uint8_t *)calloc(fb->width * fb->height, sizeof(uint8_t));
  if (!roi_mask)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for ROI mask");
    free_processed_image(&edges);
    return ESP_ERR_NO_MEM;
  }

  // 定义梯形ROI区域的顶点
  int roi_vertices[6][2] = {
      {0, static_cast<int>(fb->height - 1)},                                 // 左下
      {0, (int)(fb->height * 0.8)},                                          // 左中
      {(int)(fb->width * 0.4), (int)(fb->height * 0.4)},                     // 左上
      {(int)(fb->width * 0.6), (int)(fb->height * 0.4)},                     // 右上
      {static_cast<int>(fb->width - 1), static_cast<int>(fb->height * 0.8)}, // 右中
      {static_cast<int>(fb->width - 1), static_cast<int>(fb->height - 1)}    // 右下
  };

  // 使用扫描线算法填充多边形
  // 简化版多边形填充
  int min_y = fb->height;
  int max_y = 0;

  // 找到多边形的最小和最大y
  for (int i = 0; i < 6; i++)
  {
    if (roi_vertices[i][1] < min_y)
      min_y = roi_vertices[i][1];
    if (roi_vertices[i][1] > max_y)
      max_y = roi_vertices[i][1];
  }

  // 对每一行计算多边形的交点
  for (int y = min_y; y <= max_y; y++)
  {
    int intersections[10]; // 假设不会有超过10个交点
    int num_intersections = 0;

    // 检查每条边
    for (int i = 0; i < 6; i++)
    {
      int next = (i + 1) % 6;
      int y1 = roi_vertices[i][1];
      int y2 = roi_vertices[next][1];

      // 如果这条边与当前扫描线相交
      if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y))
      {
        int x1 = roi_vertices[i][0];
        int x2 = roi_vertices[next][0];

        // 计算交点的x坐标
        int x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);

        intersections[num_intersections++] = x;
      }
    }

    // 排序交点
    for (int i = 0; i < num_intersections - 1; i++)
    {
      for (int j = 0; j < num_intersections - i - 1; j++)
      {
        if (intersections[j] > intersections[j + 1])
        {
          int temp = intersections[j];
          intersections[j] = intersections[j + 1];
          intersections[j + 1] = temp;
        }
      }
    }

    // 填充多边形内部
    for (int i = 0; i < num_intersections; i += 2)
    {
      int x1 = intersections[i];
      int x2 = intersections[i + 1];

      if (x1 < 0)
        x1 = 0;
      if (x2 >= fb->width)
        x2 = fb->width - 1;

      for (int x = x1; x <= x2; x++)
      {
        roi_mask[y * fb->width + x] = 255;
      }
    }
  }

  // 应用掩码到边缘图像
  for (int i = 0; i < fb->width * fb->height; i++)
  {
    output->data[i] = (edges.data[i] > 0 && roi_mask[i] > 0) ? 255 : 0;
  }

  // 清理
  free(roi_mask);
  free_processed_image(&edges);

  ESP_LOGI(TAG, "ROI mask applied");
  return ESP_OK;
}

// 检测直线
esp_err_t LaneAreaDetect::detect_lines(camera_fb_t *fb,
                                       processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 先应用ROI掩码
  processed_image_t masked_edges = {0};
  esp_err_t err = ROI_mask(fb, &masked_edges);
  if (err != ESP_OK)
    return err;

  // 创建输出图像 (3通道RGB图像)
  err = create_processed_image(output, fb->width, fb->height, 3);
  if (err != ESP_OK)
  {
    free_processed_image(&masked_edges);
    return err;
  }

  // 将输出图像初始化为黑色
  memset(output->data, 0, fb->width * fb->height * 3);

  // 霍夫变换参数
  const int rho = 1;
  const float theta = PI / 180.0f;
  const int threshold = 20;
  const int minLineLength = 20;
  const int maxLineGap = 300;

  // 使用概率霍夫变换检测线段
  // 这里省略具体实现，改为简化版
  // 正常情况下会使用霍夫变换检测线段，但这需要大量计算

  // 简单线段检测 (作为示例)
  // 这里使用简单的垂直积分方法检测线段

  // 左右车道线列表
  line_segment_t *left_lines =
      (line_segment_t *)malloc(100 * sizeof(line_segment_t));
  line_segment_t *right_lines =
      (line_segment_t *)malloc(100 * sizeof(line_segment_t));
  int left_count = 0;
  int right_count = 0;

  if (!left_lines || !right_lines)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for line segments");
    if (left_lines)
      free(left_lines);
    if (right_lines)
      free(right_lines);
    free_processed_image(&masked_edges);
    return ESP_ERR_NO_MEM;
  }

  // 垂直投影检测
  int *histogram = (int *)calloc(fb->width, sizeof(int));
  if (!histogram)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for histogram");
    free(left_lines);
    free(right_lines);
    free_processed_image(&masked_edges);
    return ESP_ERR_NO_MEM;
  }

  // 计算下半部分的垂直投影
  for (int y = fb->height / 2; y < fb->height; y++)
  {
    for (int x = 0; x < fb->width; x++)
    {
      if (masked_edges.data[y * fb->width + x] > 0)
      {
        histogram[x]++;
      }
    }
  }

  // 寻找左右车道线的起始点
  int left_base = 0;
  int right_base = fb->width - 1;
  int mid_point = fb->width / 2;
  int max_left = 0;
  int max_right = 0;

  for (int x = 0; x < mid_point; x++)
  {
    if (histogram[x] > max_left)
    {
      max_left = histogram[x];
      left_base = x;
    }
  }

  for (int x = mid_point; x < fb->width; x++)
  {
    if (histogram[x] > max_right)
    {
      max_right = histogram[x];
      right_base = x;
    }
  }

  // 使用滑动窗口法检测线段
  int window_height = fb->height / 10;
  int window_width = fb->width / 10;
  int n_windows = fb->height / window_height;

  int left_x = left_base;
  int right_x = right_base;

  // 滑动窗口检测左车道线
  for (int window = 0; window < n_windows; window++)
  {
    int win_y_high = fb->height - window * window_height;
    int win_y_low = fb->height - (window + 1) * window_height;
    int win_x_left = left_x - window_width / 2;
    int win_x_right = left_x + window_width / 2;

    if (win_x_left < 0)
      win_x_left = 0;
    if (win_x_right >= fb->width)
      win_x_right = fb->width - 1;

    bool found = false;
    int x_sum = 0;
    int count = 0;

    // 找出窗口内的白色像素
    for (int y = win_y_low; y < win_y_high; y++)
    {
      for (int x = win_x_left; x < win_x_right; x++)
      {
        if (masked_edges.data[y * fb->width + x] > 0)
        {
          x_sum += x;
          count++;
          found = true;
        }
      }
    }

    // 如果找到了像素，更新窗口中心
    if (found && count > 0)
    {
      left_x = x_sum / count;

      if (left_count > 0 && left_count < 100)
      {
        // 添加线段
        line_segment_t line;
        line.x1 = left_lines[left_count - 1].x2;
        line.y1 = left_lines[left_count - 1].y2;
        line.x2 = left_x;
        line.y2 = (win_y_high + win_y_low) / 2;
        line.slope = (float)(line.y2 - line.y1) / (line.x2 - line.x1);

        // 只添加斜率合理的线段
        if (fabsf(line.slope) > MIN_SLOPE)
        {
          left_lines[left_count] = line;
          left_count++;

          // 在输出图像上绘制线段
          int line_start_index = (line.y1 * fb->width + line.x1) * 3;
          int line_end_index = (line.y2 * fb->width + line.x2) * 3;

          if (line_start_index >= 0 &&
              line_start_index < fb->width * fb->height * 3 - 2 &&
              line_end_index >= 0 &&
              line_end_index < fb->width * fb->height * 3 - 2)
          {
            // 红色
            output->data[line_start_index] = 255;
            output->data[line_start_index + 1] = 0;
            output->data[line_start_index + 2] = 0;

            output->data[line_end_index] = 255;
            output->data[line_end_index + 1] = 0;
            output->data[line_end_index + 2] = 0;
          }
        }
      }
      else if (left_count == 0)
      {
        // 添加第一个点
        line_segment_t line;
        line.x1 = left_x;
        line.y1 = win_y_high;
        line.x2 = left_x;
        line.y2 = (win_y_high + win_y_low) / 2;
        line.slope = 0;

        left_lines[left_count] = line;
        left_count++;
      }
    }
  }

  // 滑动窗口检测右车道线（与左侧类似）
  for (int window = 0; window < n_windows; window++)
  {
    int win_y_high = fb->height - window * window_height;
    int win_y_low = fb->height - (window + 1) * window_height;
    int win_x_left = right_x - window_width / 2;
    int win_x_right = right_x + window_width / 2;

    if (win_x_left < 0)
      win_x_left = 0;
    if (win_x_right >= fb->width)
      win_x_right = fb->width - 1;

    bool found = false;
    int x_sum = 0;
    int count = 0;

    // 找出窗口内的白色像素
    for (int y = win_y_low; y < win_y_high; y++)
    {
      for (int x = win_x_left; x < win_x_right; x++)
      {
        if (masked_edges.data[y * fb->width + x] > 0)
        {
          x_sum += x;
          count++;
          found = true;
        }
      }
    }

    // 如果找到了像素，更新窗口中心
    if (found && count > 0)
    {
      right_x = x_sum / count;

      if (right_count > 0 && right_count < 100)
      {
        // 添加线段
        line_segment_t line;
        line.x1 = right_lines[right_count - 1].x2;
        line.y1 = right_lines[right_count - 1].y2;
        line.x2 = right_x;
        line.y2 = (win_y_high + win_y_low) / 2;
        line.slope = (float)(line.y2 - line.y1) / (line.x2 - line.x1);

        // 只添加斜率合理的线段
        if (fabsf(line.slope) > MIN_SLOPE)
        {
          right_lines[right_count] = line;
          right_count++;

          // 在输出图像上绘制线段
          int line_start_index = (line.y1 * fb->width + line.x1) * 3;
          int line_end_index = (line.y2 * fb->width + line.x2) * 3;

          if (line_start_index >= 0 &&
              line_start_index < fb->width * fb->height * 3 - 2 &&
              line_end_index >= 0 &&
              line_end_index < fb->width * fb->height * 3 - 2)
          {
            // 蓝色
            output->data[line_start_index] = 0;
            output->data[line_start_index + 1] = 0;
            output->data[line_start_index + 2] = 255;

            output->data[line_end_index] = 0;
            output->data[line_end_index + 1] = 0;
            output->data[line_end_index + 2] = 255;
          }
        }
      }
      else if (right_count == 0)
      {
        // 添加第一个点
        line_segment_t line;
        line.x1 = right_x;
        line.y1 = win_y_high;
        line.x2 = right_x;
        line.y2 = (win_y_high + win_y_low) / 2;
        line.slope = 0;

        right_lines[right_count] = line;
        right_count++;
      }
    }
  }

  // 将检测到的线段信息保存起来供拟合使用
  this->left_line_count = left_count;
  this->right_line_count = right_count;

  if (this->left_lines)
    free(this->left_lines);
  if (this->right_lines)
    free(this->right_lines);

  this->left_lines =
      (line_segment_t *)malloc(left_count * sizeof(line_segment_t));
  this->right_lines =
      (line_segment_t *)malloc(right_count * sizeof(line_segment_t));

  if (!this->left_lines || !this->right_lines)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for class line storage");
  }
  else
  {
    memcpy(this->left_lines, left_lines, left_count * sizeof(line_segment_t));
    memcpy(this->right_lines, right_lines,
           right_count * sizeof(line_segment_t));
  }

  // 清理
  free(histogram);
  free(left_lines);
  free(right_lines);
  free_processed_image(&masked_edges);

  ESP_LOGI(TAG, "Line detection completed: left=%d, right=%d", left_count,
           right_count);
  return ESP_OK;
}

// 拟合车道线
esp_err_t LaneAreaDetect::fit_lane_lines(camera_fb_t *fb,
                                         processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 先检测直线
  processed_image_t line_img = {0};
  esp_err_t err = detect_lines(fb, &line_img);
  if (err != ESP_OK)
    return err;

  // 创建输出图像 (3通道RGB图像)
  err = create_processed_image(output, fb->width, fb->height, 3);
  if (err != ESP_OK)
  {
    free_processed_image(&line_img);
    return err;
  }

  // 将输出图像初始化为黑色
  memset(output->data, 0, fb->width * fb->height * 3);

  // 拟合左右车道线
  poly_fit_t left_fit;
  poly_fit_t right_fit;
  left_fit.a = 0;
  left_fit.b = 0;
  left_fit.c = 0;
  left_fit.valid = false;

  right_fit.a = 0;
  right_fit.b = 0;
  right_fit.c = 0;
  right_fit.valid = false;
  // 拟合左车道线 (二次多项式)
  if (left_line_count >= 3)
  {
    // 提取左侧车道线所有点
    int max_points = left_line_count * 2;
    float *x = (float *)malloc(max_points * sizeof(float));
    float *y = (float *)malloc(max_points * sizeof(float));
    int point_count = 0;

    if (!x || !y)
    {
      ESP_LOGE(TAG, "Failed to allocate memory for fitting points");
      if (x)
        free(x);
      if (y)
        free(y);
      free_processed_image(&line_img);
      return ESP_ERR_NO_MEM;
    }

    for (int i = 0; i < left_line_count; i++)
    {
      if (point_count < max_points - 1)
      {
        x[point_count] = left_lines[i].x1;
        y[point_count] = left_lines[i].y1;
        point_count++;

        x[point_count] = left_lines[i].x2;
        y[point_count] = left_lines[i].y2;
        point_count++;
      }
    }

    // 简单多项式拟合方法 (最小二乘法)
    if (point_count >= 3)
    {
      // 构造系数矩阵
      float sum_y4 = 0, sum_y3 = 0, sum_y2 = 0, sum_y = 0;
      float sum_xy2 = 0, sum_xy = 0, sum_x = 0;
      int n = point_count;

      for (int i = 0; i < n; i++)
      {
        float y_val = y[i];
        float x_val = x[i];

        sum_y4 += y_val * y_val * y_val * y_val;
        sum_y3 += y_val * y_val * y_val;
        sum_y2 += y_val * y_val;
        sum_y += y_val;
        sum_xy2 += x_val * y_val * y_val;
        sum_xy += x_val * y_val;
        sum_x += x_val;
      }

      // 解线性方程组 (Ax = b) 求解二次多项式系数
      float A[3][3] = {{sum_y4, sum_y3, sum_y2},
                       {sum_y3, sum_y2, sum_y},
                       {sum_y2, sum_y, (float)n}};

      float b[3] = {sum_xy2, sum_xy, sum_x};
      float coef[3] = {0};

      // 高斯消元求解 (简化版)
      // 前向消元
      for (int i = 0; i < 3; i++)
      {
        // 查找主元
        int max_row = i;
        for (int j = i + 1; j < 3; j++)
        {
          if (fabsf(A[j][i]) > fabsf(A[max_row][i]))
          {
            max_row = j;
          }
        }

        // 交换行
        if (max_row != i)
        {
          for (int j = i; j < 3; j++)
          {
            float temp = A[i][j];
            A[i][j] = A[max_row][j];
            A[max_row][j] = temp;
          }
          float temp = b[i];
          b[i] = b[max_row];
          b[max_row] = temp;
        }

        // 消元
        for (int j = i + 1; j < 3; j++)
        {
          float factor = A[j][i] / A[i][i];
          b[j] -= factor * b[i];
          for (int k = i; k < 3; k++)
          {
            A[j][k] -= factor * A[i][k];
          }
        }
      }

      // 回代
      for (int i = 2; i >= 0; i--)
      {
        float sum = 0;
        for (int j = i + 1; j < 3; j++)
        {
          sum += A[i][j] * coef[j];
        }
        coef[i] = (b[i] - sum) / A[i][i];
      }

      // 保存结果
      left_fit.a = coef[0];
      left_fit.b = coef[1];
      left_fit.c = coef[2];
      left_fit.valid = true;
    }

    free(x);
    free(y);
  }

  // 拟合右车道线 (与左侧类似)
  if (right_line_count >= 3)
  {
    int max_points = right_line_count * 2;
    float *x = (float *)malloc(max_points * sizeof(float));
    float *y = (float *)malloc(max_points * sizeof(float));
    int point_count = 0;

    if (!x || !y)
    {
      ESP_LOGE(TAG, "Failed to allocate memory for fitting points");
      if (x)
        free(x);
      if (y)
        free(y);
      free_processed_image(&line_img);
      return ESP_ERR_NO_MEM;
    }

    for (int i = 0; i < right_line_count; i++)
    {
      if (point_count < max_points - 1)
      {
        x[point_count] = right_lines[i].x1;
        y[point_count] = right_lines[i].y1;
        point_count++;

        x[point_count] = right_lines[i].x2;
        y[point_count] = right_lines[i].y2;
        point_count++;
      }
    }

    // 简单多项式拟合方法 (最小二乘法)
    if (point_count >= 3)
    {
      // 构造系数矩阵
      float sum_y4 = 0, sum_y3 = 0, sum_y2 = 0, sum_y = 0;
      float sum_xy2 = 0, sum_xy = 0, sum_x = 0;
      int n = point_count;

      for (int i = 0; i < n; i++)
      {
        float y_val = y[i];
        float x_val = x[i];

        sum_y4 += y_val * y_val * y_val * y_val;
        sum_y3 += y_val * y_val * y_val;
        sum_y2 += y_val * y_val;
        sum_y += y_val;
        sum_xy2 += x_val * y_val * y_val;
        sum_xy += x_val * y_val;
        sum_x += x_val;
      }

      // 解线性方程组 (Ax = b) 求解二次多项式系数
      float A[3][3] = {{sum_y4, sum_y3, sum_y2},
                       {sum_y3, sum_y2, sum_y},
                       {sum_y2, sum_y, (float)n}};

      float b[3] = {sum_xy2, sum_xy, sum_x};
      float coef[3] = {0};

      // 高斯消元求解 (简化版)
      // 前向消元
      for (int i = 0; i < 3; i++)
      {
        // 查找主元
        int max_row = i;
        for (int j = i + 1; j < 3; j++)
        {
          if (fabsf(A[j][i]) > fabsf(A[max_row][i]))
          {
            max_row = j;
          }
        }

        // 交换行
        if (max_row != i)
        {
          for (int j = i; j < 3; j++)
          {
            float temp = A[i][j];
            A[i][j] = A[max_row][j];
            A[max_row][j] = temp;
          }
          float temp = b[i];
          b[i] = b[max_row];
          b[max_row] = temp;
        }

        // 消元
        for (int j = i + 1; j < 3; j++)
        {
          float factor = A[j][i] / A[i][i];
          b[j] -= factor * b[i];
          for (int k = i; k < 3; k++)
          {
            A[j][k] -= factor * A[i][k];
          }
        }
      }

      // 回代
      for (int i = 2; i >= 0; i--)
      {
        float sum = 0;
        for (int j = i + 1; j < 3; j++)
        {
          sum += A[i][j] * coef[j];
        }
        coef[i] = (b[i] - sum) / A[i][i];
      }

      // 保存结果
      right_fit.a = coef[0];
      right_fit.b = coef[1];
      right_fit.c = coef[2];
      right_fit.valid = true;
    }

    free(x);
    free(y);
  }

  // 保存拟合结果供其他函数使用
  this->left_lane_fit = left_fit;
  this->right_lane_fit = right_fit;

  // 在图像上绘制拟合曲线
  if (left_fit.valid)
  {
    // 生成y点集合
    int num_points = 20;
    float y_step = (float)(fb->height - fb->height * ROI_RATIO) / num_points;
    float y_start = fb->height * ROI_RATIO;

    // 创建点集
    int16_t *x_points = (int16_t *)malloc(num_points * sizeof(int16_t));
    int16_t *y_points = (int16_t *)malloc(num_points * sizeof(int16_t));

    if (!x_points || !y_points)
    {
      ESP_LOGE(TAG, "Failed to allocate memory for curve points");
      if (x_points)
        free(x_points);
      if (y_points)
        free(y_points);
      free_processed_image(&line_img);
      return ESP_ERR_NO_MEM;
    }

    // 计算每个点的坐标
    int valid_points = 0;
    for (int i = 0; i < num_points; i++)
    {
      float y = y_start + i * y_step;
      float x = left_fit.a * y * y + left_fit.b * y + left_fit.c;

      // 只保留有效范围内的点
      if (x >= 0 && x < fb->width)
      {
        y_points[valid_points] = (int16_t)y;
        x_points[valid_points] = (int16_t)x;
        valid_points++;
      }
    }

    // 在输出图像上绘制曲线
    for (int i = 0; i < valid_points - 1; i++)
    {
      // 绘制线段
      bresenham_line(output->data, output->width, output->height, output->channels,
                     x_points[i], y_points[i], x_points[i + 1], y_points[i + 1], 255, 0, 0);
    }

    free(x_points);
    free(y_points);
  }

  if (right_fit.valid)
  {
    // 生成y点集合
    int num_points = 20;
    float y_step = (float)(fb->height - fb->height * ROI_RATIO) / num_points;
    float y_start = fb->height * ROI_RATIO;

    // 创建点集
    int16_t *x_points = (int16_t *)malloc(num_points * sizeof(int16_t));
    int16_t *y_points = (int16_t *)malloc(num_points * sizeof(int16_t));

    if (!x_points || !y_points)
    {
      ESP_LOGE(TAG, "Failed to allocate memory for curve points");
      if (x_points)
        free(x_points);
      if (y_points)
        free(y_points);
      free_processed_image(&line_img);
      return ESP_ERR_NO_MEM;
    }

    // 计算每个点的坐标
    int valid_points = 0;
    for (int i = 0; i < num_points; i++)
    {
      float y = y_start + i * y_step;
      float x = right_fit.a * y * y + right_fit.b * y + right_fit.c;

      // 只保留有效范围内的点
      if (x >= 0 && x < fb->width)
      {
        y_points[valid_points] = (int16_t)y;
        x_points[valid_points] = (int16_t)x;
        valid_points++;
      }
    }

    // 在输出图像上绘制曲线
    for (int i = 0; i < valid_points - 1; i++)
    {
      // 绘制线段
      bresenham_line(output->data, output->width, output->height, output->channels,
                     x_points[i], y_points[i], x_points[i + 1], y_points[i + 1], 0, 0, 255);
    }

    free(x_points);
    free(y_points);
  }

  // 清理
  free_processed_image(&line_img);

  ESP_LOGI(TAG, "Lane line fitting completed");
  return ESP_OK;
}

// Bresenham直线绘制算法
void LaneAreaDetect::bresenham_line(processed_image_t *img, int x1, int y1,
                                    int x2, int y2, uint8_t r, uint8_t g,
                                    uint8_t b)
{
  bresenham_line(img->data, img->width, img->height, img->channels,
                 x1, y1, x2, y2, r, g, b);
}

// 在lane_area_detect.cpp中添加辅助版本
void LaneAreaDetect::bresenham_line(uint8_t *data, int width, int height, int channels,
                                    int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b)
{
  int dx = abs(x2 - x1);
  int dy = -abs(y2 - y1);
  int sx = (x1 < x2) ? 1 : -1;
  int sy = (y1 < y2) ? 1 : -1;
  int err = dx + dy;
  int e2;

  while (true)
  {
    // 绘制点
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
    {
      int idx = (y1 * width + x1) * channels;

      if (channels >= 3)
      {
        data[idx] = b;     // B
        data[idx + 1] = g; // G
        data[idx + 2] = r; // R
      }
      else if (channels == 1)
      {
        // 对于灰度图，使用加权平均
        data[idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
      }
    }

    if (x1 == x2 && y1 == y2)
      break;
    e2 = 2 * err;
    if (e2 >= dy)
    {
      if (x1 == x2)
        break;
      err += dy;
      x1 += sx;
    }
    if (e2 <= dx)
    {
      if (y1 == y2)
        break;
      err += dx;
      y1 += sy;
    }
  }
}

// 车道线扩展 - 向左右两侧各扩展100像素的红色掩膜区域
esp_err_t LaneAreaDetect::lane_line_expand(camera_fb_t *fb,
                                           processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 先拟合车道线
  processed_image_t lane_img = {0};
  esp_err_t err = fit_lane_lines(fb, &lane_img);
  if (err != ESP_OK)
    return err;

  // 创建输出图像 (单通道掩膜图像)
  err = create_processed_image(output, fb->width, fb->height, 1);
  if (err != ESP_OK)
  {
    free_processed_image(&lane_img);
    return err;
  }

  // 将输出图像初始化为0
  memset(output->data, 0, fb->width * fb->height);

  // 判断左右拟合是否有效
  if (!left_lane_fit.valid || !right_lane_fit.valid)
  {
    ESP_LOGW(TAG, "Lane line expand: Invalid lane fit");
    free_processed_image(&lane_img);
    return ESP_OK; // 返回空掩膜
  }

  // 扩展参数
  const int expand_width = 100; // 扩展100像素

  // 创建采样点集合
  int num_points = 20;
  float y_step = (float)(fb->height - fb->height * ROI_RATIO) / num_points;
  float y_start = fb->height * ROI_RATIO;

  // 左车道线采样点
  int16_t *left_x = (int16_t *)malloc(num_points * sizeof(int16_t));
  int16_t *left_y = (int16_t *)malloc(num_points * sizeof(int16_t));

  // 右车道线采样点
  int16_t *right_x = (int16_t *)malloc(num_points * sizeof(int16_t));
  int16_t *right_y = (int16_t *)malloc(num_points * sizeof(int16_t));

  if (!left_x || !left_y || !right_x || !right_y)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for lane points");
    if (left_x)
      free(left_x);
    if (left_y)
      free(left_y);
    if (right_x)
      free(right_x);
    if (right_y)
      free(right_y);
    free_processed_image(&lane_img);
    return ESP_ERR_NO_MEM;
  }

  // 生成每条车道线的点集
  int left_valid_points = 0;
  int right_valid_points = 0;

  for (int i = 0; i < num_points; i++)
  {
    float y = y_start + i * y_step;

    // 左车道线点
    float left_point_x =
        left_lane_fit.a * y * y + left_lane_fit.b * y + left_lane_fit.c;
    if (left_point_x >= 0 && left_point_x < fb->width)
    {
      left_y[left_valid_points] = (int16_t)y;
      left_x[left_valid_points] = (int16_t)left_point_x;
      left_valid_points++;
    }

    // 右车道线点
    float right_point_x =
        right_lane_fit.a * y * y + right_lane_fit.b * y + right_lane_fit.c;
    if (right_point_x >= 0 && right_point_x < fb->width)
    {
      right_y[right_valid_points] = (int16_t)y;
      right_x[right_valid_points] = (int16_t)right_point_x;
      right_valid_points++;
    }
  }

  // 根据左车道线生成扩展区域
  for (int i = 0; i < left_valid_points; i++)
  {
    int y = left_y[i];
    int x = left_x[i];

    // 计算法线方向(简化为车道线切线的垂直方向)
    float dx = 0, dy = 0;
    if (i > 0 && i < left_valid_points - 1)
    {
      // 使用相邻点计算切线
      dx = (float)(left_x[i + 1] - left_x[i - 1]);
      dy = (float)(left_y[i + 1] - left_y[i - 1]);
    }
    else if (i == 0 && left_valid_points > 1)
    {
      // 第一个点
      dx = (float)(left_x[1] - left_x[0]);
      dy = (float)(left_y[1] - left_y[0]);
    }
    else if (i == left_valid_points - 1 && left_valid_points > 1)
    {
      // 最后一个点
      dx = (float)(left_x[i] - left_x[i - 1]);
      dy = (float)(left_y[i] - left_y[i - 1]);
    }

    // 计算单位法向量（向左）
    float norm = sqrtf(dx * dx + dy * dy);
    if (norm > 0)
    {
      float nx = -dy / norm; // 向左的法向量
      float ny = dx / norm;

      // 绘制扩展区域
      for (int d = 0; d <= expand_width; d++)
      {
        int nx_point = x + (int)(nx * d);
        int ny_point = y + (int)(ny * d);

        // 检查边界
        if (nx_point >= 0 && nx_point < fb->width && ny_point >= 0 &&
            ny_point < fb->height)
        {
          output->data[ny_point * fb->width + nx_point] =
              AREA_LEVEL_1; // 1级区域
        }
      }
    }
  }

  // 根据右车道线生成扩展区域
  for (int i = 0; i < right_valid_points; i++)
  {
    int y = right_y[i];
    int x = right_x[i];

    // 计算法线方向
    float dx = 0, dy = 0;
    if (i > 0 && i < right_valid_points - 1)
    {
      // 使用相邻点计算切线
      dx = (float)(right_x[i + 1] - right_x[i - 1]);
      dy = (float)(right_y[i + 1] - right_y[i - 1]);
    }
    else if (i == 0 && right_valid_points > 1)
    {
      // 第一个点
      dx = (float)(right_x[1] - right_x[0]);
      dy = (float)(right_y[1] - right_y[0]);
    }
    else if (i == right_valid_points - 1 && right_valid_points > 1)
    {
      // 最后一个点
      dx = (float)(right_x[i] - right_x[i - 1]);
      dy = (float)(right_y[i] - right_y[i - 1]);
    }

    // 计算单位法向量（向右）
    float norm = sqrtf(dx * dx + dy * dy);
    if (norm > 0)
    {
      float nx = dy / norm; // 向右的法向量
      float ny = -dx / norm;

      // 绘制扩展区域
      for (int d = 0; d <= expand_width; d++)
      {
        int nx_point = x + (int)(nx * d);
        int ny_point = y + (int)(ny * d);

        // 检查边界
        if (nx_point >= 0 && nx_point < fb->width && ny_point >= 0 &&
            ny_point < fb->height)
        {
          output->data[ny_point * fb->width + nx_point] =
              AREA_LEVEL_1; // 1级区域
        }
      }
    }
  }

  // 清理
  free(left_x);
  free(left_y);
  free(right_x);
  free(right_y);
  free_processed_image(&lane_img);

  ESP_LOGI(TAG, "Lane expansion completed");
  return ESP_OK;
}

// 从SD卡加载FastSCNN模型
esp_err_t LaneAreaDetect::fastSCNN_load()
{
  ESP_LOGI(TAG, "Loading FastSCNN model from: %s", SEGMENTATION_MODEL);

  // 获取内存状态
  size_t free_psram = ESP.getFreePsram();
  size_t largest_free_block = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);

  ESP_LOGI(TAG, "Memory status - Free PSRAM: %d bytes, Largest block: %d bytes",
           free_psram, largest_free_block);

  // 检查是否已经加载
  if (model_loaded && interpreter != nullptr)
  {
    ESP_LOGI(TAG, "Model already loaded");
    return ESP_OK;
  }

  // 清理之前的资源
  if (tensor_arena != nullptr)
  {
    free(tensor_arena);
    tensor_arena = nullptr;
  }

  if (tflite_model != nullptr)
  {
    // 注意：不要直接删除tflite_model，它通常指向model_data
    tflite_model = nullptr;
  }

  if (model_data != nullptr)
  {
    free(model_data);
    model_data = nullptr;
  }

  if (interpreter != nullptr)
  {
    delete interpreter;
    interpreter = nullptr;
  }

  // 打开模型文件
  FILE *f = fopen(SEGMENTATION_MODEL, "rb");
  if (f == nullptr)
  {
    ESP_LOGE(TAG, "Failed to open model file: %s", SEGMENTATION_MODEL);
    return ESP_FAIL;
  }

  // 获取文件大小
  fseek(f, 0, SEEK_END);
  long model_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  ESP_LOGI(TAG, "Model file size: %ld bytes", model_size);

  // 分配内存加载模型
  model_data = (uint8_t *)heap_caps_malloc(model_size, MALLOC_CAP_SPIRAM);
  if (model_data == nullptr)
  {
    ESP_LOGE(TAG, "Failed to allocate memory for model: %ld bytes", model_size);
    fclose(f);
    return ESP_FAIL;
  }

  // 读取模型数据
  size_t bytes_read = fread(model_data, 1, model_size, f);
  fclose(f);

  if (bytes_read != model_size)
  {
    ESP_LOGE(TAG, "Failed to read model data: %d/%ld", bytes_read, model_size);
    free(model_data);
    model_data = nullptr;
    return ESP_FAIL;
  }

  // 获取模型
  tflite_model = tflite::GetModel(model_data);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION)
  {
    ESP_LOGE(TAG, "Model version mismatch: %d vs %d",
             tflite_model->version(), TFLITE_SCHEMA_VERSION);
    free(model_data);
    model_data = nullptr;
    tflite_model = nullptr;
    return ESP_FAIL;
  }

  // 动态尝试不同的张量区域大小
  static const size_t tensor_arena_sizes[] = {
      64 * 1024,  // 64KB
      96 * 1024,  // 96KB
      128 * 1024, // 128KB
      160 * 1024, // 160KB
      192 * 1024, // 192KB
      256 * 1024, // 256KB
      384 * 1024  // 384KB
  };

  static int attempt = 0;
  int max_attempts = sizeof(tensor_arena_sizes) / sizeof(tensor_arena_sizes[0]);

  // 获取当前尝试的大小
  size_t current_tensor_size = tensor_arena_sizes[attempt % max_attempts];

  // 检查最大连续块是否足够
  if (largest_free_block < current_tensor_size)
  {
    ESP_LOGW(TAG, "Largest free block (%d bytes) is smaller than requested tensor size (%d bytes)",
             largest_free_block, current_tensor_size);

    // 调整为最大可用大小的95%
    current_tensor_size = largest_free_block * 0.95;
    ESP_LOGI(TAG, "Adjusted tensor size to 95%% of largest free block: %d bytes",
             current_tensor_size);
  }

  ESP_LOGI(TAG, "Attempting to allocate %d KB for tensor arena (attempt %d/%d)",
           current_tensor_size / 1024, attempt + 1, max_attempts);

  // 分配张量区域
  tensor_arena = (uint8_t *)heap_caps_malloc(current_tensor_size, MALLOC_CAP_SPIRAM);
  if (tensor_arena == nullptr)
  {
    ESP_LOGE(TAG, "Failed to allocate tensor arena memory");
    free(model_data);
    model_data = nullptr;
    tflite_model = nullptr;
    attempt++; // 下次尝试不同大小
    return ESP_FAIL;
  }

  // 初始化解析器
  static tflite::MicroMutableOpResolver<7> resolver;
  if (!resolver_initialized)
  {
    // 添加所需操作
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddMean();
    resolver.AddLogistic();
    resolver.AddMul();
    resolver.AddResizeBilinear();
    resolver.AddPad();

    resolver_initialized = true;
  }

  // 创建解释器
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena,
                                             current_tensor_size);

  // 分配张量
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    ESP_LOGE(TAG, "Failed to allocate tensors");
    delete interpreter;
    interpreter = nullptr;
    free(tensor_arena);
    tensor_arena = nullptr;
    // 不要释放model_data，因为tflite_model指向它
    attempt++; // 下次尝试不同大小
    return ESP_FAIL;
  }

  // 获取并检查输入张量维度
  TfLiteTensor *input = interpreter->input(0);
  if (input->dims->size != 4)
  {
    ESP_LOGE(TAG, "Unexpected input dimensions: %d (expected 4)", input->dims->size);
    delete interpreter;
    interpreter = nullptr;
    free(tensor_arena);
    tensor_arena = nullptr;
    free(model_data);
    model_data = nullptr;
    tflite_model = nullptr;
    return ESP_FAIL;
  }

  // 记录输入张量维度
  int batch = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels = input->dims->data[3];

  ESP_LOGI(TAG, "Model input dimensions: %dx%dx%dx%d", batch, height, width, channels);

  // 获取输出张量
  TfLiteTensor *output = interpreter->output(0);

  // 记录输出张量维度
  int out_batch = output->dims->data[0];
  int out_height = output->dims->data[1];
  int out_width = output->dims->data[2];
  int out_channels = output->dims->data[3];

  ESP_LOGI(TAG, "Model output dimensions: %dx%dx%dx%d",
           out_batch, out_height, out_width, out_channels);

  // 模型加载成功
  model_loaded = true;
  ESP_LOGI(TAG, "Model loaded successfully with %d KB tensor arena", current_tensor_size / 1024);

  // 重置尝试计数
  attempt = 0;

  return ESP_OK;
}

// 生成区域拟合图
esp_err_t LaneAreaDetect::model_fit_area(camera_fb_t *fb,
                                         processed_image_t *output)
{
  if (!fb || !output)
    return ESP_FAIL;

  // 创建输出图像 (单通道分级区域图)
  esp_err_t err = create_processed_image(output, fb->width, fb->height, 1);
  if (err != ESP_OK)
    return err;

  // 初始化输出为2级区域(最外侧)
  memset(output->data, AREA_LEVEL_2, fb->width * fb->height);

  // 先获取车道线扩展的红色掩膜区域
  processed_image_t lane_mask = {0};
  err = lane_line_expand(fb, &lane_mask);
  if (err != ESP_OK)
    return err;

  // 获取FastSCNN模型预测结果
  processed_image_t model_output = {0};
  err = fastSCNN_load();
  if (err != ESP_OK)
  {
    free_processed_image(&lane_mask);
    return err;
  }

  // 生成内侧区域
  if (left_lane_fit.valid && right_lane_fit.valid)
  {
    // 生成车道线内侧区域
    for (int y = fb->height * ROI_RATIO; y < fb->height; y++)
    {
      // 计算左右车道线在当前y值的x坐标
      float left_x =
          left_lane_fit.a * y * y + left_lane_fit.b * y + left_lane_fit.c;
      float right_x =
          right_lane_fit.a * y * y + right_lane_fit.b * y + right_lane_fit.c;

      // 确保左右顺序正确
      if (left_x > right_x)
      {
        float temp = left_x;
        left_x = right_x;
        right_x = temp;
      }

      // 限制在有效范围内
      int x_start = (int)left_x;
      int x_end = (int)right_x;

      if (x_start < 0)
        x_start = 0;
      if (x_end >= fb->width)
        x_end = fb->width - 1;

      // 填充内侧区域为级别2
      for (int x = x_start; x <= x_end; x++)
      {
        int idx = y * fb->width + x;

        // 检查是否与模型预测结果重叠
        if (model_output.data[idx] > 0)
        {
          // 两车道线内侧和模型预测的重叠区域为0级区域
          output->data[idx] = AREA_LEVEL_0;
        }
        else
        {
          // 如果没有重叠，保留原状态
          // 如果是扩展区域，则设置为1级区域
          if (lane_mask.data[idx] == AREA_LEVEL_1)
          {
            output->data[idx] = AREA_LEVEL_1;
          }
        }
      }
    }
  }

  // 复制扩展区域的1级标记
  for (int i = 0; i < fb->width * fb->height; i++)
  {
    if (lane_mask.data[i] == AREA_LEVEL_1)
    {
      output->data[i] = AREA_LEVEL_1;
    }
  }

  // 清理
  free_processed_image(&lane_mask);
  free_processed_image(&model_output);

  ESP_LOGI(TAG, "Model fit area generated");
  return ESP_OK;
}

std::vector<LaneInfo> LaneAreaDetect::detect(uint8_t *imageData, int width, int height, int channels)
{
  ESP_LOGI(TAG, "Lane detection started with image size: %dx%d", width, height);

  // 结果向量
  std::vector<LaneInfo> result;
  // 检查输入参数
  if (!imageData || width <= 0 || height <= 0 || (channels != 1 && channels != 3))
  {
    ESP_LOGE(TAG, "Invalid input parameters: imageData=%p, size=%dx%d, channels=%d",
             imageData, width, height, channels);
    return result;
  }

  // 检查大小，限制最大大小以减少栈空间使用
  if (width > 640 || height > 480)
  {
    ESP_LOGW(TAG, "Image too large, may cause stack overflow: %dx%d", width, height);
    // 这里可以选择降低分辨率或返回空结果
    if (width > 800 || height > 600)
    {
      ESP_LOGE(TAG, "Image too large, skipping detection");
      return result;
    }
  }

  // 步骤1: 创建临时相机帧结构
  camera_fb_t fb_temp;
  fb_temp.width = width;
  fb_temp.height = height;

  // 根据通道数决定格式并处理数据
  if (channels == 1)
  {
    fb_temp.format = PIXFORMAT_GRAYSCALE;
    fb_temp.len = width * height;
    fb_temp.buf = imageData; // 直接使用原图像数据
  }
  else if (channels == 3)
  {
    // RGB888需要转换为RGB565
    fb_temp.format = PIXFORMAT_RGB565;
    fb_temp.len = width * height * 2; // RGB565每像素2字节
    fb_temp.buf = (uint8_t *)malloc(fb_temp.len);

    if (!fb_temp.buf)
    {
      ESP_LOGE(TAG, "Failed to allocate memory for RGB565 conversion");
      return result;
    }

    // 转换RGB888到RGB565
    for (int i = 0; i < width * height; i++)
    {
      uint8_t r = imageData[i * 3];
      uint8_t g = imageData[i * 3 + 1];
      uint8_t b = imageData[i * 3 + 2];

      // RGB888到RGB565的转换
      uint16_t rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
      fb_temp.buf[i * 2] = rgb565 & 0xFF;
      fb_temp.buf[i * 2 + 1] = (rgb565 >> 8) & 0xFF;
    }
  }
  else
  {
    ESP_LOGE(TAG, "Unsupported channel count: %d", channels);
    return result;
  }

  processed_image_t traditionalResult = {0};
  esp_err_t err = ESP_FAIL;
  bool traditional_success = false;

  // 根据设置决定是否使用传统车道线检测方法
  if (m_useTraditionalMethod)
  {
    ESP_LOGI(TAG, "使用传统车道线检测方法 (step1-7)");

    // 步骤2.1: 先执行颜色转灰度
    err = color_2_gray(&fb_temp, &traditionalResult);
    if (err != ESP_OK)
    {
      ESP_LOGW(TAG, "Color to gray conversion failed");
      // 不使用goto，避免交叉初始化问题
      traditional_success = false;
    }
    else
    {
      // 步骤2.2: 多尺度DoG滤波增强边缘
      err = multi_scale_DoG(&fb_temp, &traditionalResult);
      if (err != ESP_OK)
      {
        ESP_LOGW(TAG, "Multi-scale DoG failed");
        traditional_success = false;
      }
      else
      {
        // 步骤2.3: 边缘增强
        err = enhance_edges(&fb_temp, &traditionalResult);
        if (err != ESP_OK)
        {
          ESP_LOGW(TAG, "Edge enhancement failed");
          traditional_success = false;
        }
        else
        {
          // 步骤2.4: 应用ROI掩码
          err = ROI_mask(&fb_temp, &traditionalResult);
          if (err != ESP_OK)
          {
            ESP_LOGW(TAG, "ROI mask application failed");
            traditional_success = false;
          }
          else
          {
            // 步骤2.5: 检测直线
            err = detect_lines(&fb_temp, &traditionalResult);
            if (err != ESP_OK)
            {
              ESP_LOGW(TAG, "Line detection failed");
              traditional_success = false;
            }
            else
            {
              // 步骤2.6: 拟合车道线
              err = fit_lane_lines(&fb_temp, &traditionalResult);
              if (err != ESP_OK)
              {
                ESP_LOGW(TAG, "Lane line fitting failed");
                traditional_success = false;
                if (traditionalResult.data)
                {
                  free_processed_image(&traditionalResult);
                  traditionalResult.data = NULL;
                }
              }
              else
              {
                traditional_success = true;
              }
            }
          }
        }
      }
    }
  }

  // 如果传统方法失败或者已配置为不使用传统方法，则使用直接方法
  if (!traditional_success)
  {
    // 确保释放之前的资源
    if (traditionalResult.data)
    {
      free_processed_image(&traditionalResult);
      traditionalResult.data = NULL;
    }

    ESP_LOGI(TAG, "使用直接segmentation车道线检测方法");

    // 直接使用FastSCNN模型
    err = fastSCNN_load();

    if (err != ESP_OK)
    {
      ESP_LOGE(TAG, "FastSCNN model loading failed");
      // 清理资源
      if (traditionalResult.data)
      {
        free_processed_image(&traditionalResult);
      }

      // 释放临时帧缓冲
      if (channels == 3 && fb_temp.buf != imageData)
      {
        free(fb_temp.buf);
      }

      return result;
    }
  }

  // 步骤3: 生成最终的三级区域划分（结合传统方法和模型方法）
  processed_image_t finalMask = {0};
  err = model_fit_area(&fb_temp, &traditionalResult);

  if (err != ESP_OK)
  {
    ESP_LOGE(TAG, "Failed to generate area mask");

    // 清理资源
    if (traditionalResult.data)
    {
      free_processed_image(&traditionalResult);
    }
    if (channels == 3 && fb_temp.buf != imageData)
    {
      free(fb_temp.buf);
    }

    return result;
  }

  // 步骤4: 将结果转换为LaneInfo结构
  if (traditionalResult.data)
  {
    // 创建LaneInfo对象并填充数据
    LaneInfo info;
    info.left_lane = left_lane_fit;
    info.right_lane = right_lane_fit;

    // 设置区域掩码
    info.width = width;
    info.height = height;
    info.area_mask = (uint8_t *)malloc(width * height);

    if (info.area_mask)
    {
      // 复制区域掩码数据
      memcpy(info.area_mask, traditionalResult.data, width * height);

      // 有效性标记
      info.valid = (left_lane_fit.valid && right_lane_fit.valid);

      // 添加到结果向量
      result.push_back(info);
    }
  }

  // 步骤5: 清理资源
  if (traditionalResult.data)
    free_processed_image(&traditionalResult);
  if (finalMask.data)
    free_processed_image(&finalMask);
  if (channels == 3 && fb_temp.buf != imageData)
    free(fb_temp.buf);

  ESP_LOGI(TAG, "Lane detection completed, result count: %d", result.size());
  return result;
}
