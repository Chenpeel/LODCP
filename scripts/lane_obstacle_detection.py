import cv2
import numpy as np
from yolov5 import YOLOv5

class LaneObstacleDetector:
    def __init__(self, model_path):
        self.model = YOLOv5(model_path)
        self.lane_color = (0, 255, 0)
        self.obstacle_color = (0, 0, 255)

    def detect_lanes(self, image):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny边缘检测
        edges = cv2.Canny(blur, 50, 150)
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)

        return lines

    def process_frame(self, frame):
        # 检测车道线
        lanes = self.detect_lanes(frame)
        if lanes is not None:
            for line in lanes:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), self.lane_color, 2)

        # 检测障碍物
        results = self.model.predict(frame)
        detections = results.pandas().xyxy[0]

        for _, det in detections.iterrows():
            if det['confidence'] > 0.5:  # 置信度阈值
                x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.obstacle_color, 2)
                cv2.putText(frame, f"{det['name']}: {det['confidence']:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.obstacle_color, 1)

        return frame, detections
