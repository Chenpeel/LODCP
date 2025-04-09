import numpy as np
from scipy.spatial.distance import euclidean
from collections import deque

class CollisionPredictor:
    def __init__(self, time_horizon=2.0, safety_threshold=1.5):
        self.time_horizon = time_horizon  # 预测时间范围(秒)
        self.safety_threshold = safety_threshold  # 安全距离阈值(米)
        self.ego_vehicle_speed = 0  # 自车速度(m/s)
        self.ego_vehicle_position = (0, 0)  # 自车位置(像素坐标)

    def update_ego_vehicle(self, speed, position):
        self.ego_vehicle_speed = speed
        self.ego_vehicle_position = position

    def predict_collision_risk(self, track_history, frame_width, frame_height):
        risks = []

        for track_id, history in track_history.items():
            if len(history['positions']) < 2:
                continue

            # 计算目标速度和方向
            positions = np.array(history['positions'])
            timestamps = np.array(history['timestamps'])

            # 计算速度(像素/秒)
            dx = positions[-1, 0] - positions[-2, 0]
            dy = positions[-1, 1] - positions[-2, 1]
            dt = timestamps[-1] - timestamps[-2]

            if dt <= 0:
                continue

            vx = dx / dt
            vy = dy / dt

            # 预测未来位置
            current_pos = positions[-1]
            predicted_pos = (
                current_pos[0] + vx * self.time_horizon,
                current_pos[1] + vy * self.time_horizon
            )

            # 预测自车位置
            ego_predicted_pos = (
                self.ego_vehicle_position[0],
                self.ego_vehicle_position[1] - self.ego_vehicle_speed * self.time_horizon * (frame_height / 30)  # 假设30米对应图像高度
            )

            # 计算预测距离
            distance = euclidean(predicted_pos[:2], ego_predicted_pos)

            # 评估碰撞风险
            risk_level = 0
            if distance < self.safety_threshold * (frame_width / 10):  # 假设10米对应图像宽度
                risk_level = 1 - (distance / (self.safety_threshold * (frame_width / 10)))
                risk_level = min(max(risk_level, 0), 1)

            risks.append({
                'track_id': track_id,
                'current_position': current_pos,
                'predicted_position': predicted_pos,
                'speed': (vx, vy),
                'distance': distance,
                'risk_level': risk_level
            })

        return sorted(risks, key=lambda x: x['risk_level'], reverse=True)
