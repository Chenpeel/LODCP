#include "../include/video_recorder.h"
#include "../include/deep_sort.h"
#include "../include/semantic_seg.h"

#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_vfs.h"
#include "img_converters.h"

#include <dirent.h>
#include <mutex>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#define SCRATCH_BUFSIZE 8192

static const char *TAG = "VideoRecorder";

struct file_server_data {
  char base_path[ESP_VFS_PATH_MAX + 1];
  char scratch[SCRATCH_BUFSIZE];
};

static struct file_server_data *server_data = NULL;
static httpd_handle_t server = NULL;

// 发送目录列表
static esp_err_t http_resp_dir_html(httpd_req_t *req, const char *dirpath) {
  char entrypath[FILE_PATH_MAX];
  char entrysize[16];
  const char *entrytype;

  struct dirent *entry;
  struct stat entry_stat;
  DIR *dir = opendir(dirpath);

  if (!dir) {
    ESP_LOGE(TAG, "Failed to open directory : %s", dirpath);
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }

  // 构建HTML页面头部
  const char *header = "<!DOCTYPE html>"
                       "<html>"
                       "<head>"
                       "<title>ESP32-CAM Video Recorder</title>"
                       "<style>"
                       "body { font-family: Arial; margin: 20px; }"
                       "h1 { color: #0055AA; }"
                       "table { border-collapse: collapse; width: 100%; }"
                       "th, td { text-align: left; padding: 8px; }"
                       "tr:nth-child(even) { background-color: #f2f2f2; }"
                       "th { background-color: #0055AA; color: white; }"
                       ".folder { color: #0055AA; }"
                       ".file { color: #000000; }"
                       ".size { text-align: right; }"
                       "</style>"
                       "</head>"
                       "<body>"
                       "<h1>ESP32-CAM Video Recorder</h1>"
                       "<table>"
                       "<tr><th>Name</th><th>Type</th><th>Size</th></tr>";

  httpd_resp_send_chunk(req, header, strlen(header));

  // 遍历目录条目
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] == '.') {
      continue;
    }

    snprintf(entrypath, sizeof(entrypath), "%s/%s", dirpath, entry->d_name);
    if (stat(entrypath, &entry_stat) == -1) {
      ESP_LOGE(TAG, "Failed to stat %s : %s", entrypath, strerror(errno));
      continue;
    }

    if (S_ISDIR(entry_stat.st_mode)) {
      entrytype = "directory";
      strcpy(entrysize, "-");
    } else {
      entrytype = "file";
      sprintf(entrysize, "%ld KB", entry_stat.st_size / 1024);
    }

    // 构建表格行
    char row[512];
    if (S_ISDIR(entry_stat.st_mode)) {
      sprintf(
          row,
          "<tr><td><a class='folder' href=\"%s/\">%s/</a></td><td>%s</td><td "
          "class='size'>%s</td></tr>",
          entry->d_name, entry->d_name, entrytype, entrysize);
    } else {
      sprintf(row,
              "<tr><td><a class='file' href=\"%s\">%s</a></td><td>%s</td><td "
              "class='size'>%s</td></tr>",
              entry->d_name, entry->d_name, entrytype, entrysize);
    }
    httpd_resp_send_chunk(req, row, strlen(row));
  }
  closedir(dir);

  // HTML页面底部
  const char *footer =
      "</table>"
      "<div style='margin-top: 20px;'>"
      "<form method='post' action='/record'>"
      "<button type='submit' name='action' value='start'>Start "
      "Recording</button>"
      "<button type='submit' name='action' value='stop'>Stop Recording</button>"
      "</form>"
      "<p>Storage: %.1f MB free</p>"
      "</div>"
      "</body>"
      "</html>";

  char footer_formatted[512];
  float free_space = get_remaining_storage();
  sprintf(footer_formatted, footer, free_space);
  httpd_resp_send_chunk(req, footer_formatted, strlen(footer_formatted));

  // 发送空块表示响应结束
  httpd_resp_send_chunk(req, NULL, 0);
  return ESP_OK;
}

// 获取文件类型
static const char *get_content_type(const char *filename) {
  const char *dot = strrchr(filename, '.');
  if (dot) {
    if (strcasecmp(dot, ".html") == 0)
      return "text/html";
    if (strcasecmp(dot, ".css") == 0)
      return "text/css";
    if (strcasecmp(dot, ".js") == 0)
      return "application/javascript";
    if (strcasecmp(dot, ".jpg") == 0)
      return "image/jpeg";
    if (strcasecmp(dot, ".jpeg") == 0)
      return "image/jpeg";
    if (strcasecmp(dot, ".png") == 0)
      return "image/png";
    if (strcasecmp(dot, ".ico") == 0)
      return "image/x-icon";
    if (strcasecmp(dot, ".avi") == 0)
      return "video/x-msvideo";
    if (strcasecmp(dot, ".mp4") == 0)
      return "video/mp4";
    if (strcasecmp(dot, ".h264") == 0)
      return "video/h264";
  }
  return "application/octet-stream";
}

// 处理GET请求
static esp_err_t file_get_handler(httpd_req_t *req) {
  char filepath[FILE_PATH_MAX];
  struct stat file_stat;

  // 提取请求的URI路径
  const char *filename = req->uri;
  // 跳过URI前缀
  if (strcmp(filename, "/") == 0) {
    // 根目录, 列出SD卡内容
    sprintf(filepath, "/sdcard");
    if (stat(filepath, &file_stat) == -1 || !S_ISDIR(file_stat.st_mode)) {
      ESP_LOGE(TAG, "SD card not mounted");
      httpd_resp_send_404(req);
      return ESP_FAIL;
    }
    return http_resp_dir_html(req, filepath);
  }

  // 拼接完整文件路径
  sprintf(filepath, "/sdcard%s", filename);
  if (stat(filepath, &file_stat) == -1) {
    ESP_LOGE(TAG, "Failed to stat file : %s", filepath);
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }

  if (S_ISDIR(file_stat.st_mode)) {
    // 是目录，列出内容
    return http_resp_dir_html(req, filepath);
  }

  // 处理文件
  FILE *fd = fopen(filepath, "r");
  if (!fd) {
    ESP_LOGE(TAG, "Failed to read file : %s", filepath);
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  ESP_LOGI(TAG, "Sending file : %s (%ld bytes)", filename, file_stat.st_size);
  httpd_resp_set_type(req, get_content_type(filename));

  // 逐块发送文件
  char *chunk = server_data->scratch;
  size_t chunksize;
  do {
    chunksize = fread(chunk, 1, SCRATCH_BUFSIZE, fd);
    if (chunksize > 0) {
      if (httpd_resp_send_chunk(req, chunk, chunksize) != ESP_OK) {
        fclose(fd);
        ESP_LOGE(TAG, "File sending failed!");
        httpd_resp_sendstr_chunk(req, NULL);
        httpd_resp_send_500(req);
        return ESP_FAIL;
      }
    }
  } while (chunksize != 0);

  // 关闭文件
  fclose(fd);
  httpd_resp_send_chunk(req, NULL, 0);
  return ESP_OK;
}

// 处理录制控制请求
static esp_err_t record_post_handler(httpd_req_t *req) {
  char buf[100];
  int ret = httpd_req_recv(req, buf, sizeof(buf));
  if (ret <= 0) {
    if (ret == HTTPD_SOCK_ERR_TIMEOUT) {
      httpd_resp_send_408(req);
    }
    return ESP_FAIL;
  }
  buf[ret] = '\0';

  if (strstr(buf, "action=start")) {
    if (!is_recording()) {
      start_recording();
      httpd_resp_sendstr(req, "Recording started");
    } else {
      httpd_resp_sendstr(req, "Already recording");
    }
  } else if (strstr(buf, "action=stop")) {
    if (is_recording()) {
      stop_recording();
      httpd_resp_sendstr(req, "Recording stopped");
    } else {
      httpd_resp_sendstr(req, "Not recording");
    }
  } else {
    httpd_resp_send_404(req);
  }

  return ESP_OK;
}

// 启动HTTP服务器
esp_err_t start_video_server() {
  ESP_LOGI(TAG, "Starting HTTP server");

  server_data = (file_server_data *)calloc(1, sizeof(file_server_data));
  if (!server_data) {
    ESP_LOGE(TAG, "Failed to allocate memory for server data");
    return ESP_ERR_NO_MEM;
  }
  strlcpy(server_data->base_path, "/sdcard", sizeof(server_data->base_path));

  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_uri_handlers = 8;
  config.uri_match_fn = httpd_uri_match_wildcard;

  ESP_LOGI(TAG, "Starting HTTP server on port: '%d'", config.server_port);
  if (httpd_start(&server, &config) != ESP_OK) {
    ESP_LOGE(TAG, "Failed to start HTTP server!");
    free(server_data);
    return ESP_FAIL;
  }

  // URI处理程序
  httpd_uri_t file_download = {.uri = "/*",
                               .method = HTTP_GET,
                               .handler = file_get_handler,
                               .user_ctx = server_data};
  httpd_register_uri_handler(server, &file_download);

  httpd_uri_t record_control = {.uri = "/record",
                                .method = HTTP_POST,
                                .handler = record_post_handler,
                                .user_ctx = NULL};
  httpd_register_uri_handler(server, &record_control);

  return ESP_OK;
}
// 录制状态
static bool recording = false;
static video_config_t current_config;
static FILE *raw_video_file = NULL;
static FILE *processed_video_file = NULL;
static uint32_t frame_count = 0;
static int64_t recording_start_time = 0;
static std::mutex record_mutex;

// MJPEG文件头和帧分隔符
static const uint8_t mjpeg_header[] = {0x00, 0x00, 0x00, 0x0C, 'R',  'I', 'F',
                                       'F',  0x00, 0x00, 0x00, 0x00, 'A', 'V',
                                       'I',  ' ',  'L',  'I',  'S',  'T'};

// 为MJPEG文件创建头
static esp_err_t write_mjpeg_header(FILE *file, int width, int height,
                                    int fps) {
  // 这是一个简化的MJPEG头，实际实现需要更完整的AVI文件结构
  if (fwrite(mjpeg_header, 1, sizeof(mjpeg_header), file) !=
      sizeof(mjpeg_header)) {
    ESP_LOGE(TAG, "Failed to write MJPEG header");
    return ESP_FAIL;
  }
  return ESP_OK;
}

// 生成带时间戳的文件名
static std::string generate_filename(const std::string &base_name,
                                     bool processed) {
  char timestamp[32];
  time_t now;
  time(&now);
  strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));

  std::string prefix = processed ? "proc_" : "raw_";

  if (current_config.format == VIDEO_MJPEG) {
    return "/sdcard/" + prefix + base_name + "_" + timestamp + ".avi";
  } else if (current_config.format == VIDEO_FRAMES) {
    // 创建目录来存储帧
    std::string dir_name = "/sdcard/" + prefix + base_name + "_" + timestamp;
    mkdir(dir_name.c_str(), 0777);
    return dir_name + "/frame_";
  } else {
    return "/sdcard/" + prefix + base_name + "_" + timestamp + ".h264";
  }
}

// 创建目录（如果不存在）
static esp_err_t create_directory(const std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    // 目录不存在，创建它
    if (mkdir(path.c_str(), 0777) != 0) {
      ESP_LOGE(TAG, "Failed to create directory: %s", path.c_str());
      return ESP_FAIL;
    }
  }
  return ESP_OK;
}

// 在处理后的图像上绘制检测框和分割结果
static void draw_on_frame(uint8_t *rgb_buffer, int width, int height,
                          const frame_processing_result_t &result) {
  if (!rgb_buffer)
    return;

  // 绘制检测框
  if (current_config.draw_detections) {
    for (const auto &box : result.detection.boxes) {
      // 将归一化坐标转换为像素坐标
      int x1 = (int)((box.x - box.width / 2) * width);
      int y1 = (int)((box.y - box.height / 2) * height);
      int x2 = (int)((box.x + box.width / 2) * width);
      int y2 = (int)((box.y + box.height / 2) * height);

      // 确保坐标在图像范围内
      x1 = std::max(0, std::min(x1, width - 1));
      y1 = std::max(0, std::min(y1, height - 1));
      x2 = std::max(0, std::min(x2, width - 1));
      y2 = std::max(0, std::min(y2, height - 1));

      // 绘制边框 (红色)
      for (int i = std::max(0, x1 - 2); i <= std::min(width - 1, x1 + 2); i++) {
        for (int j = y1; j <= y2; j++) {
          if (j >= 0 && j < height) {
            rgb_buffer[(j * width + i) * 3] = 255;   // R
            rgb_buffer[(j * width + i) * 3 + 1] = 0; // G
            rgb_buffer[(j * width + i) * 3 + 2] = 0; // B
          }
        }
      }

      for (int i = std::max(0, x2 - 2); i <= std::min(width - 1, x2 + 2); i++) {
        for (int j = y1; j <= y2; j++) {
          if (j >= 0 && j < height) {
            rgb_buffer[(j * width + i) * 3] = 255;   // R
            rgb_buffer[(j * width + i) * 3 + 1] = 0; // G
            rgb_buffer[(j * width + i) * 3 + 2] = 0; // B
          }
        }
      }

      for (int i = x1; i <= x2; i++) {
        for (int j = std::max(0, y1 - 2); j <= std::min(height - 1, y1 + 2);
             j++) {
          if (i >= 0 && i < width) {
            rgb_buffer[(j * width + i) * 3] = 255;   // R
            rgb_buffer[(j * width + i) * 3 + 1] = 0; // G
            rgb_buffer[(j * width + i) * 3 + 2] = 0; // B
          }
        }
      }

      for (int i = x1; i <= x2; i++) {
        for (int j = std::max(0, y2 - 2); j <= std::min(height - 1, y2 + 2);
             j++) {
          if (i >= 0 && i < width) {
            rgb_buffer[(j * width + i) * 3] = 255;   // R
            rgb_buffer[(j * width + i) * 3 + 1] = 0; // G
            rgb_buffer[(j * width + i) * 3 + 2] = 0; // B
          }
        }
      }

      // 标注跟踪ID（如果有）
      for (const auto &track : result.tracking.tracks) {
        int track_x = (int)(track.x * width);
        int track_y = (int)(track.y * height);

        // 在跟踪目标上方绘制跟踪ID
        char id_text[8];
        sprintf(id_text, "ID:%d", track.id);

        // 简单文本渲染（只是示例，不是真正的文本渲染）
        int text_y = std::max(0, track_y - 10);
        int text_x = track_x;

        // 绘制一个小方块作为文本背景
        for (int i = text_x; i < text_x + 30; i++) {
          for (int j = text_y; j < text_y + 10; j++) {
            if (i >= 0 && i < width && j >= 0 && j < height) {
              rgb_buffer[(j * width + i) * 3] = 0;     // R
              rgb_buffer[(j * width + i) * 3 + 1] = 0; // G
              rgb_buffer[(j * width + i) * 3 + 2] = 0; // B
            }
          }
        }
      }
    }
  }

  // 绘制分割结果（半透明覆盖）
  if (current_config.draw_segmentation && result.segmentation.mask) {
    // 获取分割掩码
    const uint8_t *mask = result.segmentation.mask;
    int mask_width = result.segmentation.width;
    int mask_height = result.segmentation.height;

    // 调整分割掩码到图像尺寸
    float scale_x = (float)width / mask_width;
    float scale_y = (float)height / mask_height;

    // 为每个类别设置不同的颜色
    const uint8_t class_colors[][3] = {
        {0, 255, 0}, // 类别0: 绿色 (可行区)
        {255, 0, 0}  // 类别1: 红色 (非可行区)
    };

    // 绘制分割掩码（半透明）
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // 映射到掩码坐标
        int mask_x = (int)(x / scale_x);
        int mask_y = (int)(y / scale_y);

        if (mask_x >= 0 && mask_x < mask_width && mask_y >= 0 &&
            mask_y < mask_height) {
          // 获取像素类别
          uint8_t pixel_class = mask[mask_y * mask_width + mask_x];
          if (pixel_class < 2) { // 确保类别在范围内
            // 半透明绘制
            rgb_buffer[(y * width + x) * 3] =
                (rgb_buffer[(y * width + x) * 3] * 0.7) +
                (class_colors[pixel_class][0] * 0.3); // R
            rgb_buffer[(y * width + x) * 3 + 1] =
                (rgb_buffer[(y * width + x) * 3 + 1] * 0.7) +
                (class_colors[pixel_class][1] * 0.3); // G
            rgb_buffer[(y * width + x) * 3 + 2] =
                (rgb_buffer[(y * width + x) * 3 + 2] * 0.7) +
                (class_colors[pixel_class][2] * 0.3); // B
          }
        }
      }
    }
  }
}

// 初始化视频录制器
esp_err_t init_video_recorder(const video_config_t &config) {
  std::lock_guard<std::mutex> lock(record_mutex);

  // 检查SD卡是否已挂载
  struct stat st;
  if (stat("/sdcard", &st) != 0) {
    ESP_LOGE(TAG, "SD card not mounted at /sdcard");
    return ESP_FAIL;
  }

  // 保存配置
  current_config = config;

  ESP_LOGI(TAG, "Video recorder initialized with format: %d, filename: %s",
           config.format, config.filename.c_str());

  return ESP_OK;
}

// 开始录制
esp_err_t start_recording() {
  std::lock_guard<std::mutex> lock(record_mutex);

  if (recording) {
    ESP_LOGW(TAG, "Recording already in progress");
    return ESP_OK;
  }

  // 生成文件名
  std::string raw_filename = generate_filename(current_config.filename, false);
  std::string processed_filename =
      generate_filename(current_config.filename, true);

  ESP_LOGI(TAG, "Starting recording to: %s and %s", raw_filename.c_str(),
           processed_filename.c_str());

  if (current_config.format == VIDEO_FRAMES) {
    // 对于帧序列，创建目录
    create_directory(raw_filename.substr(0, raw_filename.find_last_of("/")));
    create_directory(
        processed_filename.substr(0, processed_filename.find_last_of("/")));
  } else {
    // 对于视频文件，打开文件
    raw_video_file = fopen(raw_filename.c_str(), "wb");
    if (!raw_video_file) {
      ESP_LOGE(TAG, "Failed to open raw video file: %s", raw_filename.c_str());
      return ESP_FAIL;
    }

    processed_video_file = fopen(processed_filename.c_str(), "wb");
    if (!processed_video_file) {
      ESP_LOGE(TAG, "Failed to open processed video file: %s",
               processed_filename.c_str());
      fclose(raw_video_file);
      raw_video_file = NULL;
      return ESP_FAIL;
    }

    // 写入视频头
    if (current_config.format == VIDEO_MJPEG) {
      // 暂时使用默认值，后面会根据第一帧的尺寸进行更新
      write_mjpeg_header(raw_video_file, 640, 480, current_config.fps);
      write_mjpeg_header(processed_video_file, 640, 480, current_config.fps);
    }
  }

  // 重置帧计数和开始时间
  frame_count = 0;
  recording_start_time = esp_timer_get_time();
  recording = true;

  ESP_LOGI(TAG, "Recording started");
  return ESP_OK;
}

// 停止录制
esp_err_t stop_recording() {
  std::lock_guard<std::mutex> lock(record_mutex);

  if (!recording) {
    ESP_LOGW(TAG, "No recording in progress");
    return ESP_OK;
  }

  // 关闭文件
  if (current_config.format != VIDEO_FRAMES) {
    if (raw_video_file) {
      fclose(raw_video_file);
      raw_video_file = NULL;
    }

    if (processed_video_file) {
      fclose(processed_video_file);
      processed_video_file = NULL;
    }
  }

  // 计算录制统计信息
  int64_t recording_duration =
      (esp_timer_get_time() - recording_start_time) / 1000; // ms
  float actual_fps = (recording_duration > 0)
                         ? (frame_count * 1000.0f / recording_duration)
                         : 0;

  ESP_LOGI(TAG,
           "Recording stopped. Frames: %d, Duration: %lld ms, Avg FPS: %.2f",
           frame_count, recording_duration, actual_fps);

  recording = false;
  return ESP_OK;
}

// 保存未处理的帧
esp_err_t save_raw_frame(camera_fb_t *fb) {
  if (!recording || !fb) {
    return ESP_OK; // 不录制或无效帧，直接返回
  }

  std::lock_guard<std::mutex> lock(record_mutex);

  if (current_config.format == VIDEO_FRAMES) {
    // 保存为单独的帧
    std::string raw_filename =
        generate_filename(current_config.filename, false);
    char frame_filename[128];
    sprintf(frame_filename, "%s%06d.jpg", raw_filename.c_str(), frame_count);

    FILE *f = fopen(frame_filename, "wb");
    if (!f) {
      ESP_LOGE(TAG, "Failed to open frame file: %s", frame_filename);
      return ESP_FAIL;
    }

    esp_err_t ret = ESP_OK;
    // 如果是JPEG格式直接写入，否则转换为JPEG
    if (fb->format == PIXFORMAT_JPEG) {
      if (fwrite(fb->buf, 1, fb->len, f) != fb->len) {
        ESP_LOGE(TAG, "Failed to write JPEG data");
        ret = ESP_FAIL;
      }
    } else {
      uint8_t *jpeg_buf = NULL;
      size_t jpeg_len = 0;
      if (!fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_RGB888,
                   current_config.quality, &jpeg_buf, &jpeg_len)) {
        ESP_LOGE(TAG, "Failed to convert to JPEG");
        ret = ESP_FAIL;
      } else if (jpeg_buf) {
        if (fwrite(jpeg_buf, 1, jpeg_len, f) != jpeg_len) {
          ESP_LOGE(TAG, "Failed to write converted JPEG data");
          ret = ESP_FAIL;
        }
        free(jpeg_buf);
      }
    }

    fclose(f);
    if (ret != ESP_OK) {
      return ret;
    }
  } else if (current_config.format == VIDEO_MJPEG && raw_video_file) {
    // 写入MJPEG文件
    if (fb->format == PIXFORMAT_JPEG) {
      // 写入帧大小和帧数据
      uint32_t frame_size = fb->len;
      fwrite(&frame_size, sizeof(frame_size), 1, raw_video_file);
      fwrite(fb->buf, 1, fb->len, raw_video_file);
    } else {
      // 转换为JPEG
      uint8_t *jpeg_buf = NULL;
      size_t jpeg_len = 0;
      fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_RGB888,
              current_config.quality, &jpeg_buf, &jpeg_len);

      if (jpeg_buf) {
        // 写入帧大小和帧数据
        uint32_t frame_size = jpeg_len;
        fwrite(&frame_size, sizeof(frame_size), 1, raw_video_file);
        fwrite(jpeg_buf, 1, jpeg_len, raw_video_file);
        free(jpeg_buf);
      }
    }
  }

  frame_count++;
  return ESP_OK;
}

// 保存处理后的帧
esp_err_t save_processed_frame(camera_fb_t *fb,
                               const frame_processing_result_t &result) {
  if (!recording || !fb) {
    return ESP_OK; // 不录制或无效帧，直接返回
  }

  std::lock_guard<std::mutex> lock(record_mutex);

  // 创建RGB缓冲区
  uint8_t *rgb_buffer = NULL;
  size_t rgb_len = fb->width * fb->height * 3;

  if (fb->format == PIXFORMAT_JPEG) {
    // 将JPEG转换为RGB
    rgb_buffer = (uint8_t *)malloc(rgb_len);
    if (!rgb_buffer) {
      ESP_LOGE(TAG, "Failed to allocate memory for RGB buffer");
      return ESP_FAIL;
    }

    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb_buffer);
    if (!converted) {
      ESP_LOGE(TAG, "Failed to convert JPEG to RGB");
      free(rgb_buffer);
      return ESP_FAIL;
    }
  } else if (fb->format == PIXFORMAT_RGB888) {
    // 复制RGB数据
    rgb_buffer = (uint8_t *)malloc(rgb_len);
    if (!rgb_buffer) {
      ESP_LOGE(TAG, "Failed to allocate memory for RGB buffer");
      return ESP_FAIL;
    }
    memcpy(rgb_buffer, fb->buf, rgb_len);
  } else {
    ESP_LOGE(TAG, "Unsupported pixel format for processing");
    return ESP_FAIL;
  }

  // 在帧上绘制检测结果和分割结果
  draw_on_frame(rgb_buffer, fb->width, fb->height, result);

  // 转换回JPEG
  uint8_t *jpeg_buf = NULL;
  size_t jpeg_len = 0;
  bool success =
      fmt2jpg(rgb_buffer, rgb_len, fb->width, fb->height, PIXFORMAT_RGB888,
              current_config.quality, &jpeg_buf, &jpeg_len);

  free(rgb_buffer);

  if (!success || !jpeg_buf) {
    ESP_LOGE(TAG, "Failed to convert processed frame to JPEG");
    return ESP_FAIL;
  }

  if (current_config.format == VIDEO_FRAMES) {
    // 保存为单独的帧
    std::string processed_filename =
        generate_filename(current_config.filename, true);
    char frame_filename[128];
    sprintf(frame_filename, "%s%06d.jpg", processed_filename.c_str(),
            frame_count - 1);

    FILE *f = fopen(frame_filename, "wb");
    if (!f) {
      ESP_LOGE(TAG, "Failed to open processed frame file: %s", frame_filename);
      free(jpeg_buf);
      return ESP_FAIL;
    }

    fwrite(jpeg_buf, 1, jpeg_len, f);
    fclose(f);
  } else if (current_config.format == VIDEO_MJPEG && processed_video_file) {
    // 写入MJPEG文件
    uint32_t frame_size = jpeg_len;
    fwrite(&frame_size, sizeof(frame_size), 1, processed_video_file);
    fwrite(jpeg_buf, 1, jpeg_len, processed_video_file);
  }

  free(jpeg_buf);
  return ESP_OK;
}

// 获取录制状态
bool is_recording() { return recording; }

// 获取剩余SD卡空间（MB）
float get_remaining_storage() {
  FILE *f = popen("df -m /sdcard | tail -1 | awk '{print $4}'", "r");
  if (!f) {
    return -1;
  }

  char output[32];
  if (fgets(output, sizeof(output), f) != NULL) {
    pclose(f);
    return atof(output);
  }

  pclose(f);
  return -1;
}
