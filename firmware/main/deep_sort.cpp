#include "../include/deepsort.h"
#include "esp_log.h"
#include <algorithm>
#include <cmath>
#include <mutex>

static const char *TAG = "DeepSORT";

// 跟踪器参数
#define MAX_AGE 30        // 最大消失帧数
#define MIN_HITS 3        // 确认跟踪所需的最小命中数
#define IOU_THRESHOLD 0.7 // IoU匹配阈值

// 轨迹类 - 在DeepSORT中每个跟踪对象都有一个轨迹
class KalmanTracker {
public:
  KalmanTracker(detection_box_t init_box, int init_id) {
    id = init_id;

    // 初始化状态 [cx, cy, w, h, vx, vy, 0, 0]
    state = {init_box.x, init_box.y, init_box.width, init_box.height, 0, 0,
             0,          0};

    // 记录类别ID
    class_id = init_box.class_id;

    // 设置为新轨迹
    age = 1;
    hits = 1;
    hit_streak = 1;
    time_since_update = 0;

    ESP_LOGI(TAG, "Created tracker ID %d at (%.2f, %.2f, %.2f, %.2f)", id,
             state[0], state[1], state[2], state[3]);
  }

  // 预测下一个状态
  void predict() {
    // 简化的预测 - 添加速度给位置
    state[0] += state[4]; // cx += vx
    state[1] += state[5]; // cy += vy

    // 更新计数器
    age += 1;
    time_since_update += 1;

    // 如果连续命中中断，重置计数
    if (time_since_update > 0)
      hit_streak = 0;
  }

  // 用新的检测更新状态
  void update(detection_box_t detection) {
    // 更新位置 (简单的EMA平滑)
    const float alpha = 0.7f;
    state[0] = alpha * detection.x + (1 - alpha) * state[0];
    state[1] = alpha * detection.y + (1 - alpha) * state[1];
    state[2] = alpha * detection.width + (1 - alpha) * state[2];
    state[3] = alpha * detection.height + (1 - alpha) * state[3];

    // 计算速度
    state[4] = detection.x - last_x;
    state[5] = detection.y - last_y;

    // 保存当前位置用于下次计算速度
    last_x = detection.x;
    last_y = detection.y;

    // 更新计数器
    hits += 1;
    hit_streak += 1;
    time_since_update = 0;

    ESP_LOGD(TAG, "Updated tracker ID %d at (%.2f, %.2f, %.2f, %.2f)", id,
             state[0], state[1], state[2], state[3]);
  }

  // 获取当前边界框
  detection_box_t get_state() const {
    detection_box_t box;
    box.x = state[0];
    box.y = state[1];
    box.width = state[2];
    box.height = state[3];
    box.class_id = class_id;
    box.confidence = 1.0f; // 跟踪器不输出置信度
    return box;
  }

  // 获取跟踪对象
  track_t get_track() const {
    track_t track;
    track.id = id;
    track.x = state[0];
    track.y = state[1];
    track.width = state[2];
    track.height = state[3];
    track.vx = state[4];
    track.vy = state[5];
    track.class_id = class_id;
    track.age = age;
    track.time_since_update = time_since_update;
    return track;
  }

  // 检查是否是确认的跟踪
  bool is_confirmed() const { return hit_streak >= MIN_HITS; }

  // 检查是否应该删除
  bool is_deleted() const { return time_since_update > MAX_AGE; }

public:
  std::vector<float> state; // [cx, cy, width, height, vx, vy, ...]
  int id;                   // 唯一跟踪ID
  int hits;                 // 总命中数
  int hit_streak;           // 连续命中数
  int age;                  // 总帧数
  int time_since_update;    // 自上次更新以来的帧数
  int class_id;             // 对象类别

  float last_x = 0; // 上次更新的x位置
  float last_y = 0; // 上次更新的y位置
};

// 全局跟踪器
static std::vector<std::shared_ptr<KalmanTracker>> trackers;
static int next_id = 1;
static int frame_count = 0;
static std::mutex tracker_mutex;
static tracking_result_t latest_tracking = {std::vector<track_t>(), 0};

// 计算两个边界框的IoU
float calculate_iou(const detection_box_t &box1, const detection_box_t &box2) {
  // 计算每个边界框的坐标
  float box1_x1 = box1.x - box1.width / 2;
  float box1_y1 = box1.y - box1.height / 2;
  float box1_x2 = box1.x + box1.width / 2;
  float box1_y2 = box1.y + box1.height / 2;

  float box2_x1 = box2.x - box2.width / 2;
  float box2_y1 = box2.y - box2.height / 2;
  float box2_x2 = box2.x + box2.width / 2;
  float box2_y2 = box2.y + box2.height / 2;

  // 计算交集区域
  float x_left = std::max(box1_x1, box2_x1);
  float y_top = std::max(box1_y1, box2_y1);
  float x_right = std::min(box1_x2, box2_x2);
  float y_bottom = std::min(box1_y2, box2_y2);

  if (x_right < x_left || y_bottom < y_top)
    return 0.0f;

  float intersection_area = (x_right - x_left) * (y_bottom - y_top);

  // 计算并集区域
  float box1_area = box1.width * box1.height;
  float box2_area = box2.width * box2.height;
  float union_area = box1_area + box2_area - intersection_area;

  // 返回IoU
  return intersection_area / union_area;
}

// 初始化DeepSORT跟踪器
esp_err_t init_deep_sort() {
  ESP_LOGI(TAG, "Initializing DeepSORT tracker");

  {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    trackers.clear();
    next_id = 1;
    frame_count = 0;
  }

  return ESP_OK;
}

// 更新跟踪器并获取跟踪结果
tracking_result_t update_tracker(const detection_result_t &detections) {
  std::lock_guard<std::mutex> lock(tracker_mutex);

  frame_count++;
  ESP_LOGI(TAG, "Updating tracker for frame %d with %d detections", frame_count,
           detections.boxes.size());

  // 如果没有跟踪器，初始化跟踪器
  if (trackers.empty() && !detections.boxes.empty()) {
    for (const auto &det : detections.boxes) {
      trackers.push_back(std::make_shared<KalmanTracker>(det, next_id++));
    }

    // 更新最新跟踪结果
    tracking_result_t result;
    result.frame_count = frame_count;

    for (const auto &tracker : trackers) {
      if (tracker->is_confirmed() && !tracker->is_deleted()) {
        result.tracks.push_back(tracker->get_track());
      }
    }

    latest_tracking = result;
    return result;
  }

  // 预测所有跟踪器的新状态
  for (auto &tracker : trackers) {
    tracker->predict();
  }

  // 关联检测与现有跟踪
  std::vector<std::pair<int, int>>
      matched_pairs; // (tracker_idx, detection_idx)
  std::vector<int> unmatched_detections;
  std::vector<int> unmatched_trackers;

  // 初始化未匹配的检测和跟踪器
  for (int i = 0; i < detections.boxes.size(); i++) {
    unmatched_detections.push_back(i);
  }

  for (int i = 0; i < trackers.size(); i++) {
    if (!trackers[i]->is_deleted()) {
      unmatched_trackers.push_back(i);
    }
  }

  // 计算所有检测和跟踪之间的IoU
  std::vector<std::vector<float>> iou_matrix(
      trackers.size(), std::vector<float>(detections.boxes.size(), 0));

  for (int t = 0; t < trackers.size(); t++) {
    detection_box_t trk_box = trackers[t]->get_state();

    for (int d = 0; d < detections.boxes.size(); d++) {
      iou_matrix[t][d] = calculate_iou(trk_box, detections.boxes[d]);
    }
  }

  // 使用贪婪算法进行匹配
  while (!unmatched_detections.empty() && !unmatched_trackers.empty()) {
    // 找到最大IoU
    float max_iou = -1;
    int max_t = -1, max_d = -1;

    for (int t_idx = 0; t_idx < unmatched_trackers.size(); t_idx++) {
      int t = unmatched_trackers[t_idx];

      for (int d_idx = 0; d_idx < unmatched_detections.size(); d_idx++) {
        int d = unmatched_detections[d_idx];

        if (iou_matrix[t][d] > max_iou) {
          max_iou = iou_matrix[t][d];
          max_t = t_idx;
          max_d = d_idx;
        }
      }
    }

    // 如果找到了匹配且IoU足够高
    if (max_iou >= IOU_THRESHOLD) {
      int t = unmatched_trackers[max_t];
      int d = unmatched_detections[max_d];

      matched_pairs.push_back(std::make_pair(t, d));

      // 从未匹配列表中移除
      unmatched_trackers.erase(unmatched_trackers.begin() + max_t);
      unmatched_detections.erase(unmatched_detections.begin() + max_d);
    } else {
      // 没有更多有效的匹配
      break;
    }
  }

  // 更新匹配的跟踪器
  for (const auto &match : matched_pairs) {
    trackers[match.first]->update(detections.boxes[match.second]);
  }

  // 为未匹配的检测创建新跟踪器
  for (int d_idx : unmatched_detections) {
    trackers.push_back(
        std::make_shared<KalmanTracker>(detections.boxes[d_idx], next_id++));
  }

  // 删除过期的跟踪器
  trackers.erase(
      std::remove_if(trackers.begin(), trackers.end(),
                     [](const std::shared_ptr<KalmanTracker> &tracker) {
                       return tracker->is_deleted();
                     }),
      trackers.end());

  // 准备返回跟踪结果
  tracking_result_t result;
  result.frame_count = frame_count;

  for (const auto &tracker : trackers) {
    if (tracker->is_confirmed() && !tracker->is_deleted()) {
      result.tracks.push_back(tracker->get_track());
    }
  }

  ESP_LOGI(TAG, "Tracking completed with %d active tracks",
           result.tracks.size());
  latest_tracking = result;
  return result;
}

// 获取最新的跟踪结果
tracking_result_t get_latest_tracking() {
  std::lock_guard<std::mutex> lock(tracker_mutex);
  return latest_tracking;
}
