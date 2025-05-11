#include "deep_sort.h"
#include "object_detect.h"
#include "esp_log.h"
#include <algorithm>
#include <limits>

// ==== KalmanTracker实现 ====

KalmanTracker::KalmanTracker(const Detection& det, int id, int max_age, int min_hits) 
    : trackId(id), hits(1), age(1), time_since_update(0), state(TENTATIVE),
      classId(det.classId), confidence(det.confidence),
      max_age(max_age), min_hits(min_hits),
      kf(8, 4)  // 状态向量大小为8，测量向量大小为4
{
    // 转换为中心点+宽高比表示
    float centerX = det.x + det.width / 2.0f;
    float centerY = det.y + det.height / 2.0f;
    float aspect_ratio = det.width / det.height;
    float height = det.height;
    
    // 初始化状态向量 [cx,cy,a,h,vx,vy,va,vh]
    std::vector<float> initialState = {
        centerX, centerY, aspect_ratio, height,
        0.0f, 0.0f, 0.0f, 0.0f  // 初始速度为0
    };
    kf.setStatePost(initialState);
    
    // 设置状态转移矩阵 (匀速模型)
    std::vector<std::vector<float>> transMatrix(8, std::vector<float>(8, 0.0f));
    for (int i = 0; i < 8; i++) {
        transMatrix[i][i] = 1.0f;
    }
    // 位置 += 速度
    transMatrix[0][4] = 1.0f;
    transMatrix[1][5] = 1.0f;
    transMatrix[2][6] = 1.0f;
    transMatrix[3][7] = 1.0f;
    kf.setTransitionMatrix(transMatrix);
    
    // 设置测量矩阵 (只测量位置和大小，不测量速度)
    std::vector<std::vector<float>> measMatrix(4, std::vector<float>(8, 0.0f));
    measMatrix[0][0] = 1.0f;  // cx
    measMatrix[1][1] = 1.0f;  // cy
    measMatrix[2][2] = 1.0f;  // a
    measMatrix[3][3] = 1.0f;  // h
    kf.setMeasurementMatrix(measMatrix);
    
    // 设置过程噪声协方差
    std::vector<std::vector<float>> procNoise(8, std::vector<float>(8, 0.0f));
    for (int i = 0; i < 8; i++) {
        procNoise[i][i] = 0.01f;  // 较小的过程噪声
    }
    // 速度部分的噪声稍大
    for (int i = 4; i < 8; i++) {
        procNoise[i][i] = 0.05f;
    }
    kf.setProcessNoiseCov(procNoise);
    
    // 设置测量噪声协方差
    std::vector<std::vector<float>> measNoise(4, std::vector<float>(4, 0.0f));
    for (int i = 0; i < 4; i++) {
        measNoise[i][i] = 0.1f;  // 较大的测量噪声
    }
    kf.setMeasurementNoiseCov(measNoise);
    
    // 设置后验误差协方差
    std::vector<std::vector<float>> errorCov(8, std::vector<float>(8, 0.0f));
    for (int i = 0; i < 8; i++) {
        errorCov[i][i] = 1.0f;
    }
    kf.setErrorCovPost(errorCov);
}

void KalmanTracker::predict() {
    // 使用卡尔曼滤波器预测下一个状态
    kf.predict();
    age++;
}

void KalmanTracker::update(const Detection& det) {
    // 更新目标类别和置信度
    classId = det.classId;
    confidence = std::max(confidence, det.confidence); // 保留最高置信度
    
    // 转换为中心点+宽高比表示
    float centerX = det.x + det.width / 2.0f;
    float centerY = det.y + det.height / 2.0f;
    float aspect_ratio = det.width / det.height;
    float height = det.height;
    
    // 创建测量向量
    std::vector<float> measurement = {centerX, centerY, aspect_ratio, height};
    
    // 使用卡尔曼滤波器更新状态
    kf.correct(measurement);
    
    // 更新跟踪器状态
    hits++;
    time_since_update = 0;
    
    // 检查是否需要从临时状态提升为确认状态
    if (state == TENTATIVE && hits >= min_hits) {
        tentativeToConfirmed();
    }
}

Detection KalmanTracker::getPredictedState() const {
    // 获取当前状态
    std::vector<float> state = kf.getState();
    
    // 如果返回了空状态（出错），创建一个默认检测
    if (state.empty()) {
        ESP_LOGE("KalmanTracker", "Failed to get state from Kalman filter!");
        Detection det;
        det.classId = classId;
        det.confidence = 0.0f;
        det.x = 0.0f;
        det.y = 0.0f;
        det.width = 1.0f;
        det.height = 1.0f;
        return det;
    }
    
    float centerX = state[0];
    float centerY = state[1];
    float aspect_ratio = state[2];
    float height = state[3];
    float width = aspect_ratio * height;
    
    // 计算左上角坐标
    float x = centerX - width / 2.0f;
    float y = centerY - height / 2.0f;
    
    // 创建检测结果
    Detection det;
    det.classId = classId;
    det.confidence = confidence;
    det.x = x;
    det.y = y;
    det.width = width;
    det.height = height;
    
    return det;
}

void KalmanTracker::tentativeToConfirmed() {
    state = CONFIRMED;
}

void KalmanTracker::checkForDeletion() {
    if (time_since_update > max_age) {
        state = DELETED;
    }
}

// ==== DeepSORT实现 ====

DeepSORT::DeepSORT(float max_iou_distance, int max_age, int n_init)
    : next_track_id(0), max_iou_distance(max_iou_distance),
      max_age(max_age), n_init(n_init) {
    ESP_LOGI(TAG, "初始化DeepSORT跟踪器");
}

DeepSORT::~DeepSORT() {
    // 清理资源
}

void DeepSORT::reset() {
    trackers.clear();
    next_track_id = 0;
}

float DeepSORT::calculateIoU(const Detection& det1, const Detection& det2) {
    // 计算两个边界框的IoU
    
    // 计算交集矩形
    float xmin = std::max(det1.x, det2.x);
    float ymin = std::max(det1.y, det2.y);
    float xmax = std::min(det1.x + det1.width, det2.x + det2.width);
    float ymax = std::min(det1.y + det1.height, det2.y + det2.height);
    
    // 如果没有重叠，返回0
    if (xmax < xmin || ymax < ymin) {
        return 0.0f;
    }
    
    // 计算交集面积
    float intersection = (xmax - xmin) * (ymax - ymin);
    
    // 计算并集面积
    float area1 = det1.width * det1.height;
    float area2 = det2.width * det2.height;
    float union_area = area1 + area2 - intersection;
    
    // 返回IoU
    return intersection / union_area;
}

std::vector<std::vector<float>> DeepSORT::computeIouMatrix(
    const std::vector<Detection>& detections,
    const std::vector<KalmanTracker>& trackers) {
    
    int num_detections = detections.size();
    int num_trackers = trackers.size();
    
    // 创建成本矩阵
    std::vector<std::vector<float>> cost_matrix(num_detections, std::vector<float>(num_trackers, 0.0f));
    
    // 计算每个检测和跟踪器之间的IoU
    for (int i = 0; i < num_detections; i++) {
        for (int j = 0; j < num_trackers; j++) {
            // 获取跟踪器的当前预测状态
            Detection predicted_det = trackers[j].getPredictedState();
            
            // 计算IoU
            float iou = calculateIoU(detections[i], predicted_det);
            
            // IoU越大，成本越小
            cost_matrix[i][j] = 1.0f - iou;
        }
    }
    
    return cost_matrix;
}

std::vector<std::pair<int, int>> DeepSORT::hungrarianMatching(const std::vector<std::vector<float>>& cost_matrix) {
    // 贪婪匹配算法（简化版替代匈牙利算法）
    int rows = cost_matrix.size();
    int cols = rows > 0 ? cost_matrix[0].size() : 0;
    
    std::vector<std::pair<int, int>> matches;
    
    if (rows == 0 || cols == 0) {
        return matches;
    }
    
    std::vector<bool> used_rows(rows, false);
    std::vector<bool> used_cols(cols, false);
    
    // 创建排序后的成本矩阵元素
    struct CostElement {
        int row;
        int col;
        float cost;
        
        bool operator<(const CostElement& other) const {
            return cost < other.cost;
        }
    };
    
    std::vector<CostElement> all_costs;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            all_costs.push_back({i, j, cost_matrix[i][j]});
        }
    }
    
    // 按成本升序排序
    std::sort(all_costs.begin(), all_costs.end());
    
    // 贪婪选择最小成本的配对
    for (const auto& element : all_costs) {
        if (!used_rows[element.row] && !used_cols[element.col] && element.cost <= max_iou_distance) {
            matches.push_back({element.row, element.col});
            used_rows[element.row] = true;
            used_cols[element.col] = true;
        }
    }
    
    return matches;
}

void DeepSORT::initTracker(const Detection& detection) {
    // 创建新的跟踪器
    KalmanTracker tracker(detection, next_track_id++, max_age, n_init);
    trackers.push_back(tracker);
}

void DeepSORT::updateTracker(int track_idx, const Detection& detection) {
    // 更新现有跟踪器
    trackers[track_idx].update(detection);
}

void DeepSORT::predictTrackers() {
    // 对所有跟踪器进行预测
    for (auto& tracker : trackers) {
        tracker.predict();
    }
}

void DeepSORT::deleteOldTrackers() {
    // 删除过期的跟踪器
    auto it = trackers.begin();
    while (it != trackers.end()) {
        // 检查是否该删除该跟踪器
        it->checkForDeletion();
        if (it->isDeleted()) {
            it = trackers.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<Track> DeepSORT::generateTrackResults() {
    std::vector<Track> results;
    
    // 仅返回确认状态的跟踪结果
    for (const auto& tracker : trackers) {
        if (tracker.isConfirmed()) {
            // 获取当前预测状态
            Detection state = tracker.getPredictedState();
            std::vector<float> kf_state = tracker.kf.getState();
            
            // 创建跟踪结果
            Track track;
            track.trackId = tracker.trackId;
            track.classId = tracker.classId;
            track.confidence = tracker.confidence;
            track.x = state.x;
            track.y = state.y;
            track.width = state.width;
            track.height = state.height;
            
            // 从卡尔曼状态获取速度
            track.vx = kf_state[4]; // 速度x
            track.vy = kf_state[5]; // 速度y
            
            // 设置年龄
            track.age = tracker.age;
            
            results.push_back(track);
        }
    }
    
    return results;
}

std::vector<Track> DeepSORT::update(const std::vector<Detection>& detections, uint8_t* imageData, int width, int height) {
    ESP_LOGI(TAG, "更新跟踪，检测数: %d，跟踪器数: %d", detections.size(), trackers.size());
    
    // 步骤1: 预测所有现有跟踪器的新位置
    predictTrackers();
    
    // 步骤2: 使用匈牙利算法关联检测和跟踪器
    std::vector<std::vector<float>> cost_matrix = computeIouMatrix(detections, trackers);
    std::vector<std::pair<int, int>> matches = hungrarianMatching(cost_matrix);
    
    // 步骤3: 处理匹配结果
    
    // 创建已匹配检测和跟踪器的集合
    std::set<int> matched_detections;
    std::set<int> matched_trackers;
    
    // 更新匹配的跟踪器
    for (const auto& match : matches) {
        int det_idx = match.first;
        int track_idx = match.second;
        
        updateTracker(track_idx, detections[det_idx]);
        
        matched_detections.insert(det_idx);
        matched_trackers.insert(track_idx);
    }
    
    // 步骤4: 为未匹配的检测创建新的跟踪器
    for (size_t i = 0; i < detections.size(); i++) {
        if (matched_detections.find(i) == matched_detections.end()) {
            initTracker(detections[i]);
        }
    }
    
    // 步骤5: 更新未匹配跟踪器的状态
    for (size_t i = 0; i < trackers.size(); i++) {
        if (matched_trackers.find(i) == matched_trackers.end()) {
            trackers[i].markMissed();
        }
    }
    
    // 步骤6: 删除过期的跟踪器
    deleteOldTrackers();
    
    // 步骤7: 生成跟踪结果
    std::vector<Track> results = generateTrackResults();
    
    ESP_LOGI(TAG, "跟踪结果: %d 个目标", results.size());
    return results;
}

std::vector<Track> DeepSORT::track(uint8_t *imageData, int width, int height, int channels) {
    // 步骤1: 检测物体
    std::vector<Detection> detections = detector.detect(imageData, width, height, channels);
    
    // 步骤2: 使用检测结果更新跟踪器
    return update(detections, imageData, width, height);
}