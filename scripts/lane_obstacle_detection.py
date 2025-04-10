import cv2
import numpy as np
from yolov5 import YOLOv5

class LaneObstacleDetector:
    def __init__(self, model_path):
        self.model = YOLOv5(model_path)
        self.lane_color = (0, 255, 0)  # 车道线颜色（绿色）
        self.obstacle_color = (0, 0, 255)  # 障碍物颜色（红色）

    def _preprocess_lane(self, image):
        """车道线预处理：灰度化+边缘检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detect_lanes(self, image):
        """基于霍夫变换的车道线检测"""
        edges = self._preprocess_lane(image)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=20
        )
        return lines

    def detect_obstacles(self, image):
        """YOLOv5障碍物检测"""
        results = self.model.predict(image)
        return results.pandas().xyxy[0]  # 返回DataFrame格式结果

    def visualize(self, image, lanes, obstacles):
        """可视化叠加结果"""
        # 绘制车道线
        if lanes is not None:
            for line in lanes:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), self.lane_color, 3)

        # 绘制障碍物
        for _, det in obstacles.iterrows():
            if det['confidence'] > 0.5:
                x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                cv2.rectangle(image, (x1, y1), (x2, y2), self.obstacle_color, 2)
                cv2.putText(
                    image,
                    f"{det['name']}: {det['confidence']:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.obstacle_color,
                    2
                )
        return image

    def process_frame(self, frame):
        """完整处理流程"""
        lanes = self.detect_lanes(frame)
        obstacles = self.detect_obstacles(frame)
        return self.visualize(frame.copy(), lanes, obstacles)
