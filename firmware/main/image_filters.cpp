#include "../include/image_filters.h"
#include "esp_log.h"
#include "img_converters.h"
#include <algorithm>
#include <cmath>
#include <vector>

static const char *TAG = "ImageFilters";

// 将相机帧转换为灰度图像
esp_err_t convert_to_grayscale(camera_fb_t *fb, processed_image_t *output) {
  if (!fb || !output) {
    ESP_LOGE(TAG, "Invalid input parameters");
    return ESP_ERR_INVALID_ARG;
  }

  // 分配内存
  output->width = fb->width;
  output->height = fb->height;
  output->channels = 1;
  output->data = (uint8_t *)malloc(output->width * output->height);

  if (!output->data) {
    ESP_LOGE(TAG, "Failed to allocate memory for grayscale image");
    return ESP_ERR_NO_MEM;
  }

  // RGB缓冲区
  uint8_t *rgb_buffer = NULL;
  bool free_rgb = false;

  if (fb->format == PIXFORMAT_JPEG) {
    // 转换JPEG到RGB888
    rgb_buffer = (uint8_t *)malloc(fb->width * fb->height * 3);
    if (!rgb_buffer) {
      ESP_LOGE(TAG, "Failed to allocate memory for RGB buffer");
      free(output->data);
      output->data = NULL;
      return ESP_ERR_NO_MEM;
    }
    free_rgb = true;

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb_buffer);
    if (!converted) {
      ESP_LOGE(TAG, "Failed to convert JPEG to RGB888");
      free(rgb_buffer);
      free(output->data);
      output->data = NULL;
      return ESP_FAIL;
    }
  } else if (fb->format == PIXFORMAT_RGB888) {
    rgb_buffer = fb->buf;
  } else {
    ESP_LOGE(TAG, "Unsupported image format");
    free(output->data);
    output->data = NULL;
    return ESP_ERR_NOT_SUPPORTED;
  }

  // 转换RGB到灰度
  for (int i = 0; i < fb->height; i++) {
    for (int j = 0; j < fb->width; j++) {
      int rgb_idx = (i * fb->width + j) * 3;
      int gray_idx = i * fb->width + j;

      // 标准RGB到灰度转换: Y = 0.299*R + 0.587*G + 0.114*B
      output->data[gray_idx] = (uint8_t)(0.299f * rgb_buffer[rgb_idx] +
                                         0.587f * rgb_buffer[rgb_idx + 1] +
                                         0.114f * rgb_buffer[rgb_idx + 2]);
    }
  }

  if (free_rgb) {
    free(rgb_buffer);
  }

  return ESP_OK;
}

// 释放处理后的图像
void free_processed_image(processed_image_t *img) {
  if (img && img->data) {
    free(img->data);
    img->data = NULL;
  }
}

// 创建高斯核
static std::vector<float> create_gaussian_kernel(int size, float sigma) {
  std::vector<float> kernel(size * size);
  const float PI = 3.14159265358979323846f;

  int center = size / 2;
  float sum = 0.0f;

  // 计算高斯核
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      int dx = x - center;
      int dy = y - center;
      float value = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
      value /= 2.0f * PI * sigma * sigma;
      kernel[y * size + x] = value;
      sum += value;
    }
  }

  // 归一化
  for (int i = 0; i < size * size; i++) {
    kernel[i] /= sum;
  }

  return kernel;
}

// 应用核卷积(带填充)
static esp_err_t apply_kernel(const processed_image_t *input,
                              processed_image_t *output,
                              const std::vector<float> &kernel,
                              int kernel_size) {
  if (!input || !input->data || !output) {
    return ESP_ERR_INVALID_ARG;
  }

  // 分配输出内存
  output->width = input->width;
  output->height = input->height;
  output->channels = input->channels;
  output->data =
      (uint8_t *)malloc(output->width * output->height * output->channels);

  if (!output->data) {
    return ESP_ERR_NO_MEM;
  }

  int padding = kernel_size / 2;

  // 检查内核大小是否合理
  if (kernel.size() != kernel_size * kernel_size) {
    ESP_LOGE(TAG, "Kernel size mismatch");
    free(output->data);
    output->data = NULL;
    return ESP_ERR_INVALID_ARG;
  }

  // 对每个像素应用卷积
  for (int c = 0; c < input->channels; c++) {
    for (int y = 0; y < input->height; y++) {
      for (int x = 0; x < input->width; x++) {
        float sum = 0.0f;

        // 应用卷积核
        for (int ky = 0; ky < kernel_size; ky++) {
          for (int kx = 0; kx < kernel_size; kx++) {
            int px = x + kx - padding;
            int py = y + ky - padding;

            // 使用重复填充处理边界
            px = std::max(0, std::min(px, input->width - 1));
            py = std::max(0, std::min(py, input->height - 1));

            // 安全地访问输入数据
            size_t input_idx =
                ((size_t)py * input->width + px) * input->channels + c;
            if (input_idx >= input->width * input->height * input->channels) {
              ESP_LOGW(TAG, "Input buffer access out of bounds");
              continue;
            }

            float pixel_value = input->data[input_idx];
            sum += pixel_value * kernel[ky * kernel_size + kx];
          }
        }

        // 存储结果
        size_t output_idx =
            ((size_t)y * output->width + x) * output->channels + c;
        if (output_idx >= output->width * output->height * output->channels) {
          ESP_LOGW(TAG, "Output buffer access out of bounds");
          continue;
        }

        output->data[output_idx] =
            (uint8_t)std::max(0.0f, std::min(255.0f, sum));
      }
    }
  }

  return ESP_OK;
}

// 高斯滤波器
esp_err_t apply_gaussian_filter(camera_fb_t *fb, processed_image_t *output,
                                float sigma) {
  processed_image_t gray;
  esp_err_t ret = convert_to_grayscale(fb, &gray);
  if (ret != ESP_OK) {
    return ret;
  }

  // 创建高斯核 (核大小 = 6*sigma)
  int kernel_size = (int)(6.0f * sigma);
  kernel_size =
      kernel_size % 2 == 0 ? kernel_size + 1 : kernel_size; // 确保大小为奇数
  kernel_size = std::max(3, std::min(kernel_size, 15));     // 限制大小

  std::vector<float> kernel = create_gaussian_kernel(kernel_size, sigma);

  // 应用高斯核
  ret = apply_kernel(&gray, output, kernel, kernel_size);

  free_processed_image(&gray);
  return ret;
}

// LoG (拉普拉斯高斯) 滤波器
esp_err_t apply_log_filter(camera_fb_t *fb, processed_image_t *output,
                           float sigma) {
  processed_image_t gray;
  esp_err_t ret = convert_to_grayscale(fb, &gray);
  if (ret != ESP_OK) {
    return ret;
  }

  // 创建LoG核 (核大小 = 6*sigma)
  int kernel_size = (int)(6.0f * sigma);
  kernel_size =
      kernel_size % 2 == 0 ? kernel_size + 1 : kernel_size; // 确保大小为奇数
  kernel_size = std::max(3, std::min(kernel_size, 15));     // 限制大小

  std::vector<float> kernel(kernel_size * kernel_size);
  int center = kernel_size / 2;
  float sigma2 = sigma * sigma;
  float sum = 0.0f;

  // 计算LoG核
  for (int y = 0; y < kernel_size; y++) {
    for (int x = 0; x < kernel_size; x++) {
      int dx = x - center;
      int dy = y - center;
      float r2 = dx * dx + dy * dy;

      // LoG公式: ∇²G = [r² - 2*sigma²]/(sigma⁴) * exp(-r²/(2*sigma²))
      float value = (r2 - 2.0f * sigma2) * expf(-r2 / (2.0f * sigma2));
      value /= (2.0f * 3.14159265358979323846f * sigma2 * sigma2);

      kernel[y * kernel_size + x] = value;
      sum += value;
    }
  }

  // 应用LoG核
  ret = apply_kernel(&gray, output, kernel, kernel_size);

  free_processed_image(&gray);
  return ret;
}

// DoG (高斯差分) 滤波器
esp_err_t apply_dog_filter(camera_fb_t *fb, processed_image_t *output,
                           float sigma1, float sigma2) {
  processed_image_t gaussian1, gaussian2;

  // 应用两个不同sigma的高斯滤波
  esp_err_t ret = apply_gaussian_filter(fb, &gaussian1, sigma1);
  if (ret != ESP_OK) {
    return ret;
  }

  ret = apply_gaussian_filter(fb, &gaussian2, sigma2);
  if (ret != ESP_OK) {
    free_processed_image(&gaussian1);
    return ret;
  }

  // 分配输出内存
  output->width = gaussian1.width;
  output->height = gaussian1.height;
  output->channels = gaussian1.channels;
  output->data =
      (uint8_t *)malloc(output->width * output->height * output->channels);

  if (!output->data) {
    free_processed_image(&gaussian1);
    free_processed_image(&gaussian2);
    return ESP_ERR_NO_MEM;
  }

  // 计算差分
  int total_pixels = output->width * output->height * output->channels;
  for (int i = 0; i < total_pixels; i++) {
    int diff = gaussian1.data[i] - gaussian2.data[i];
    output->data[i] =
        (uint8_t)std::max(0, std::min(255, 128 + diff)); // 加偏移使其可见
  }

  free_processed_image(&gaussian1);
  free_processed_image(&gaussian2);

  return ESP_OK;
}

// Gabor滤波器
esp_err_t apply_gabor_filter(camera_fb_t *fb, processed_image_t *output,
                             float lambda, float theta, float sigma,
                             float gamma) {
  processed_image_t gray;
  esp_err_t ret = convert_to_grayscale(fb, &gray);
  if (ret != ESP_OK) {
    return ret;
  }

  // 创建Gabor核 (核大小 = 6*sigma)
  int kernel_size = (int)(6.0f * sigma);
  kernel_size =
      kernel_size % 2 == 0 ? kernel_size + 1 : kernel_size; // 确保大小为奇数
  kernel_size = std::max(3, std::min(kernel_size, 15));     // 限制大小

  std::vector<float> kernel(kernel_size * kernel_size);
  int center = kernel_size / 2;

  // 预计算三角函数值
  float cos_theta = cosf(theta);
  float sin_theta = sinf(theta);

  // 计算Gabor核
  float sum = 0.0f;
  for (int y = 0; y < kernel_size; y++) {
    for (int x = 0; x < kernel_size; x++) {
      float dx = x - center;
      float dy = y - center;

      // 旋转坐标
      float x_theta = dx * cos_theta + dy * sin_theta;
      float y_theta = -dx * sin_theta + dy * cos_theta;

      // Gabor公式
      float exp_term =
          expf(-(x_theta * x_theta + gamma * gamma * y_theta * y_theta) /
               (2.0f * sigma * sigma));
      float cos_term = cosf(2.0f * 3.14159265358979323846f * x_theta / lambda);
      float value = exp_term * cos_term;

      kernel[y * kernel_size + x] = value;
      sum += fabsf(value);
    }
  }

  // 归一化
  for (int i = 0; i < kernel_size * kernel_size; i++) {
    kernel[i] /= sum;
  }

  // 应用Gabor核
  ret = apply_kernel(&gray, output, kernel, kernel_size);

  free_processed_image(&gray);
  return ret;
}

// Canny边缘检测
esp_err_t detect_edges_canny(camera_fb_t *fb, processed_image_t *output,
                             float low_threshold, float high_threshold) {
  processed_image_t gray, blurred, gradient_x, gradient_y, gradient_mag,
      gradient_dir;

  // 1. 转换为灰度图像
  esp_err_t ret = convert_to_grayscale(fb, &gray);
  if (ret != ESP_OK) {
    return ret;
  }

  // 2. 应用高斯滤波减少噪声
  ret = apply_gaussian_filter(fb, &blurred, 1.4f);
  if (ret != ESP_OK) {
    free_processed_image(&gray);
    return ret;
  }

  // 3. 计算梯度 (使用Sobel算子)
  // Sobel X核: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
  std::vector<float> sobel_x = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

  // Sobel Y核: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
  std::vector<float> sobel_y = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  ret = apply_kernel(&blurred, &gradient_x, sobel_x, 3);
  if (ret != ESP_OK) {
    free_processed_image(&gray);
    free_processed_image(&blurred);
    return ret;
  }

  ret = apply_kernel(&blurred, &gradient_y, sobel_y, 3);
  if (ret != ESP_OK) {
    free_processed_image(&gray);
    free_processed_image(&blurred);
    free_processed_image(&gradient_x);
    return ret;
  }

  // 计算梯度幅值和方向
  gradient_mag.width = blurred.width;
  gradient_mag.height = blurred.height;
  gradient_mag.channels = 1;
  gradient_mag.data =
      (uint8_t *)malloc(gradient_mag.width * gradient_mag.height);

  gradient_dir.width = blurred.width;
  gradient_dir.height = blurred.height;
  gradient_dir.channels = 1;
  gradient_dir.data =
      (uint8_t *)malloc(gradient_dir.width * gradient_dir.height);

  if (!gradient_mag.data || !gradient_dir.data) {
    free_processed_image(&gray);
    free_processed_image(&blurred);
    free_processed_image(&gradient_x);
    free_processed_image(&gradient_y);
    free_processed_image(&gradient_mag);
    free_processed_image(&gradient_dir);
    return ESP_ERR_NO_MEM;
  }

  for (int i = 0; i < blurred.height; i++) {
    for (int j = 0; j < blurred.width; j++) {
      int idx = i * blurred.width + j;

      float gx = (float)gradient_x.data[idx];
      float gy = (float)gradient_y.data[idx];

      // 计算梯度幅值
      float magnitude = sqrtf(gx * gx + gy * gy);
      gradient_mag.data[idx] = (uint8_t)std::min(255.0f, magnitude);

      // 计算梯度方向 (0, 45, 90, 135度)
      float angle = atan2f(gy, gx) * 180.0f / 3.14159265358979323846f;
      if (angle < 0)
        angle += 180.0f;

      if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        gradient_dir.data[idx] = 0; // 0度
      } else if (angle >= 22.5 && angle < 67.5) {
        gradient_dir.data[idx] = 45; // 45度
      } else if (angle >= 67.5 && angle < 112.5) {
        gradient_dir.data[idx] = 90; // 90度
      } else {
        gradient_dir.data[idx] = 135; // 135度
      }
    }
  }

  // 4. 非极大值抑制
  output->width = blurred.width;
  output->height = blurred.height;
  output->channels = 1;
  output->data = (uint8_t *)malloc(output->width * output->height);

  if (!output->data) {
    free_processed_image(&gray);
    free_processed_image(&blurred);
    free_processed_image(&gradient_x);
    free_processed_image(&gradient_y);
    free_processed_image(&gradient_mag);
    free_processed_image(&gradient_dir);
    return ESP_ERR_NO_MEM;
  }

  // 初始化为0
  memset(output->data, 0, output->width * output->height);

  for (int i = 1; i < blurred.height - 1; i++) {
    for (int j = 1; j < blurred.width - 1; j++) {
      int idx = i * blurred.width + j;
      int dir = gradient_dir.data[idx];
      int mag = gradient_mag.data[idx];

      // 沿梯度方向比较
      bool is_max = false;
      if (dir == 0) {
        // 水平方向
        is_max = mag > gradient_mag.data[idx - 1] &&
                 mag > gradient_mag.data[idx + 1];
      } else if (dir == 45) {
        // 45度方向
        is_max = mag > gradient_mag.data[(i - 1) * blurred.width + j + 1] &&
                 mag > gradient_mag.data[(i + 1) * blurred.width + j - 1];
      } else if (dir == 90) {
        // 垂直方向
        is_max = mag > gradient_mag.data[(i - 1) * blurred.width + j] &&
                 mag > gradient_mag.data[(i + 1) * blurred.width + j];
      } else if (dir == 135) {
        // 135度方向
        is_max = mag > gradient_mag.data[(i - 1) * blurred.width + j - 1] &&
                 mag > gradient_mag.data[(i + 1) * blurred.width + j + 1];
      }

      // 如果是极大值，保留梯度值，否则为0
      if (is_max) {
        output->data[idx] = mag;
      }
    }
  }

  // 5. 双阈值处理和边缘连接
  uint8_t low = (uint8_t)(low_threshold * 255);
  uint8_t high = (uint8_t)(high_threshold * 255);

  // 临时缓冲区
  std::vector<uint8_t> temp(output->width * output->height, 0);

  // 强边缘 = 255, 弱边缘 = 100, 非边缘 = 0
  for (int i = 0; i < output->width * output->height; i++) {
    if (output->data[i] >= high) {
      temp[i] = 255; // 强边缘
    } else if (output->data[i] >= low) {
      temp[i] = 100; // 弱边缘
    }
  }

  // 连接边缘 (将与强边缘相连的弱边缘转为强边缘)
  for (int i = 1; i < blurred.height - 1; i++) {
    for (int j = 1; j < blurred.width - 1; j++) {
      int idx = i * blurred.width + j;

      if (temp[idx] == 100) {
        // 检查8邻域是否有强边缘
        bool connected = false;
        for (int di = -1; di <= 1 && !connected; di++) {
          for (int dj = -1; dj <= 1 && !connected; dj++) {
            if (di == 0 && dj == 0)
              continue;
            int neighbor_idx = (i + di) * blurred.width + (j + dj);
            if (temp[neighbor_idx] == 255) {
              connected = true;
            }
          }
        }

        if (connected) {
          output->data[idx] = 255; // 强边缘
        } else {
          output->data[idx] = 0; // 非边缘
        }
      } else {
        output->data[idx] = temp[idx] == 255 ? 255 : 0;
      }
    }
  }

  free_processed_image(&gray);
  free_processed_image(&blurred);
  free_processed_image(&gradient_x);
  free_processed_image(&gradient_y);
  free_processed_image(&gradient_mag);
  free_processed_image(&gradient_dir);

  return ESP_OK;
}

// 多项式曲线拟合（提取轮廓并拟合）
esp_err_t polynomial_curve_fitting(const processed_image_t *edge_image,
                                   int degree, float *coefficients) {
  if (!edge_image || !edge_image->data || !coefficients || degree < 1) {
    return ESP_ERR_INVALID_ARG;
  }

  // 收集边缘点
  std::vector<std::pair<float, float>> points;

  for (int y = 0; y < edge_image->height; y++) {
    for (int x = 0; x < edge_image->width; x++) {
      int idx = y * edge_image->width + x;
      if (edge_image->data[idx] > 0) {
        // 归一化坐标以获得更好的数值稳定性
        float norm_x = (float)x / edge_image->width;
        float norm_y = (float)y / edge_image->height;
        points.push_back(std::make_pair(norm_x, norm_y));
      }
    }
  }

  if (points.empty()) {
    ESP_LOGE(TAG, "No edge points found for fitting");
    return ESP_ERR_NOT_FOUND;
  }

  // 最小平方拟合 (ax^n + bx^(n-1) + ... + z)
  // 构建矩阵 A 和向量 b, 解线性方程 Ax = b
  const int n = points.size();
  const int m = degree + 1; // 系数的数量

  // 构建矩阵
  std::vector<std::vector<float>> A(m, std::vector<float>(m, 0.0f));
  std::vector<float> b(m, 0.0f);

  // 计算矩阵 A 和向量 b
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      float sum = 0.0f;
      for (const auto &point : points) {
        sum += powf(point.first, i + j);
      }
      A[i][j] = sum;
    }

    float sum = 0.0f;
    for (const auto &point : points) {
      sum += powf(point.first, i) * point.second;
    }
    b[i] = sum;
  }

  // 高斯消元法求解线性方程组
  for (int i = 0; i < m - 1; i++) {
    // 查找主元
    int max_row = i;
    float max_val = fabsf(A[i][i]);

    for (int j = i + 1; j < m; j++) {
      if (fabsf(A[j][i]) > max_val) {
        max_row = j;
        max_val = fabsf(A[j][i]);
      }
    }

    // 如果主元太小，矩阵几乎是奇异的
    if (max_val < 1e-10f) {
      ESP_LOGE(TAG, "Matrix is singular, cannot solve");
      return ESP_FAIL;
    }

    // 交换行
    if (max_row != i) {
      std::swap(A[i], A[max_row]);
      std::swap(b[i], b[max_row]);
    }

    // 消元
    for (int j = i + 1; j < m; j++) {
      float factor = A[j][i] / A[i][i];

      for (int k = i; k < m; k++) {
        A[j][k] -= factor * A[i][k];
      }

      b[j] -= factor * b[i];
    }
  }

  // 回代
  for (int i = m - 1; i >= 0; i--) {
    coefficients[i] = b[i];

    for (int j = i + 1; j < m; j++) {
      coefficients[i] -= A[i][j] * coefficients[j];
    }

    coefficients[i] /= A[i][i];
  }

  ESP_LOGI(TAG, "Polynomial fitting completed with %d points for degree %d",
           (int)points.size(), degree);

  return ESP_OK;
}
