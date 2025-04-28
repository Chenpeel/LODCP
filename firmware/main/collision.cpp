#include "../include/collision.h"
#include "esp_log.h"
#include <algorithm>
#include <cmath>
#include <mutex>

static const char *TAG = "Collision";

// 碰撞预测参数
#define MAX_PREDICTION_FRAMES 10 // 最大预测帧数
#define RISK_THRESHOLD 0.6       // 风险阈值
#define COLLISION_DISTANCE 0.2   // 碰撞距离

// 道路相关类别ID
#define ROAD_CLASS_ID 0     // 可行区类别ID
#define NON_ROAD_CLASS_ID 1 // 不可行区类别ID

static std::mutex collision_mutex;
static collision_prediction_t latest_prediction = {
    std::vector<collision_risk_t>(), 0.0f};

// 初始化碰撞预测模块
void init_collision_prediction() {
  ESP_LOGI(TAG, "Initializing collision prediction module");
}

// 检查点是否在分割掩码的特定类中
bool is_point_in_class(const segmentation_result_t &seg, int x, int y,
                       int class_id) {
  if (!seg.mask || x < 0 || y < 0 || x >= seg.width || y >= seg.height) {
    return false;
  }

  return seg.mask[y * seg.width + x] == class_id;
}

// 预测物体的未来位置
std::vector<std::pair<float, float>>
predict_future_positions(const track_t &track, int num_frames) {
  std::vector<std::pair<float, float>> positions;

  for (int i = 1; i <= num_frames; i++) {
    float future_x = track.x + track.vx * i;
    float future_y = track.y + track.vy * i;
    positions.push_back(std::make_pair(future_x, future_y));
  }

  return positions;
}

// 计算点到最近非道路区域的距离
float distance_to_non_road(const segmentation_result_t &seg, float x, float y) {
  if (!seg.mask) {
    return -1.0f;
  }

  // 归一化坐标到分割尺寸
  int seg_x = (int)(x * seg.width);
  int seg_y = (int)(y * seg.height);

  // 检查点是否已经在非道路区域
  if (seg_x >= 0 && seg_y >= 0 && seg_x < seg.width && seg_y < seg.height) {
    if (seg.mask[seg_y * seg.width + seg_x] != ROAD_CLASS_ID) {
      return 0.0f;
    }
  }

  // 找到最近的非道路像素
  float min_distance = INFINITY;

  // 简化实现：检查附近的像素
  const int search_radius = std::min(20, std::min(seg.width, seg.height) / 4);

  for (int dy = -search_radius; dy <= search_radius; dy++) {
    for (int dx = -search_radius; dx <= search_radius; dx++) {
      int check_x = seg_x + dx;
      int check_y = seg_y + dy;

      if (check_x < 0 || check_y < 0 || check_x >= seg.width ||
          check_y >= seg.height) {
        continue;
      }

      if (seg.mask[check_y * seg.width + check_x] != ROAD_CLASS_ID) {
        float distance = std::sqrt(dx * dx + dy * dy) / (float)seg.height;
        min_distance = std::min(min_distance, distance);
      }
    }
  }

  return min_distance;
}

// 预测碰撞风险
collision_prediction_t
predict_collisions(const tracking_result_t &tracking,
                   const segmentation_result_t &segmentation) {
  ESP_LOGI(TAG, "Predicting collisions for %d tracks", tracking.tracks.size());

  collision_prediction_t prediction;
  prediction.overall_risk = 0.0f;

  if (tracking.tracks.empty() || !segmentation.mask) {
    return prediction;
  }

  // 对每个跟踪对象进行预测
  for (const auto &track : tracking.tracks) {
    // 只对移动物体进行预测
    float speed = std::sqrt(track.vx * track.vx + track.vy * track.vy);
    if (speed < 0.01f) {
      continue;
    }

    // 预测未来位置
    auto future_positions =
        predict_future_positions(track, MAX_PREDICTION_FRAMES);

    // 检查是否会离开道路区域
    float min_distance = INFINITY;
    int collision_frame = -1;

    for (int i = 0; i < future_positions.size(); i++) {
      float future_x = future_positions[i].first;
      float future_y = future_positions[i].second;

      // 计算到非道路区域的距离
      float distance = distance_to_non_road(segmentation, future_x, future_y);

      if (distance < min_distance) {
        min_distance = distance;
        collision_frame = i + 1;
      }

      // 发现碰撞
      if (distance < COLLISION_DISTANCE) {
        break;
      }
    }

    // 计算风险分数
    float risk_score = 0.0f;
    float time_to_collision = INFINITY;

    if (collision_frame > 0) {
      // 风险分数基于预测的碰撞时间
      risk_score = 1.0f - (float)collision_frame / MAX_PREDICTION_FRAMES;
      risk_score = std::max(0.0f, std::min(1.0f, risk_score));

      // 估计碰撞时间 (假设30FPS)
      time_to_collision = collision_frame / 30.0f;
    }

    // 如果风险超过阈值，添加到预测结果
    if (risk_score > RISK_THRESHOLD) {
      collision_risk_t risk;
      risk.track_id = track.id;
      risk.risk_score = risk_score;
      risk.time_to_collision = time_to_collision;
      risk.class_id = track.class_id;

      prediction.risks.push_back(risk);

      ESP_LOGI(TAG, "Track %d has collision risk %.2f in %.2f seconds",
               track.id, risk_score, time_to_collision);
    }
  }

  // 计算总体风险
  if (!prediction.risks.empty()) {
    float max_risk = 0.0f;
    for (const auto &risk : prediction.risks) {
      max_risk = std::max(max_risk, risk.risk_score);
    }
    prediction.overall_risk = max_risk;
  }

  ESP_LOGI(TAG, "Overall collision risk: %.2f", prediction.overall_risk);

  // 更新最新预测
  {
    std::lock_guard<std::mutex> lock(collision_mutex);
    latest_prediction = prediction;
    latest_prediction.risks.clear();
    latest_prediction.risks = prediction.risks;
    latest_prediction.overall_risk = prediction.overall_risk;
  }

  return prediction;
}

// 获取最新的碰撞预测结果
collision_prediction_t get_latest_prediction() {
  std::lock_guard<std::mutex> lock(collision_mutex);
  return latest_prediction;
}
