#ifndef COLLISION_H
#define COLLISION_H

#include "../include/deep_sort.h"
#include "../include/semantic_seg.h"
#include <vector>

// 碰撞风险结构体
typedef struct {
  int track_id;            // 跟踪ID
  float risk_score;        // 风险分数 (0-1)
  float time_to_collision; // 预计碰撞时间 (秒)
  int class_id;            // 对象类别
} collision_risk_t;

// 碰撞预测结果
typedef struct {
  std::vector<collision_risk_t> risks;
  float overall_risk; // 总体风险评分 (0-1)
} collision_prediction_t;

// 初始化碰撞预测模块
void init_collision_prediction();

// 预测碰撞风险
collision_prediction_t
predict_collisions(const tracking_result_t &tracking,
                   const segmentation_result_t &segmentation);

// 获取最新的碰撞预测结果
collision_prediction_t get_latest_prediction();

#endif // COLLISION_H
