from deep_sort import DeepSort
import numpy as np

class ObstacleTracker:
    def __init__(self):
        self.tracker = DeepSort(
            model_path='deep_sort/mars-small128.pb',
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0
        )
        self.history = {}  # 存储目标历史轨迹

    def update(self, detections, frame):
        # 转换检测结果为DeepSORT格式
        bboxes = []
        confidences = []
        class_ids = []

        for _, det in detections.iterrows():
            bboxes.append([det['xmin'], det['ymin'], det['xmax'], det['ymax']])
            confidences.append(det['confidence'])
            class_ids.append(det['class'])

        if len(bboxes) > 0:
            bboxes = np.array(bboxes)
            confidences = np.array(confidences)
            class_ids = np.array(class_ids)

            # 更新跟踪器
            tracks = self.tracker.update(bboxes, confidences, class_ids, frame)

            # 更新历史轨迹
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr()
                class_id = track.class_id

                if track_id not in self.history:
                    self.history[track_id] = {
                        'positions': [],
                        'class_id': class_id,
                        'timestamps': []
                    }

                self.history[track_id]['positions'].append(bbox)
                self.history[track_id]['timestamps'].append(time.time())

                # 保留最近30个轨迹点
                if len(self.history[track_id]['positions']) > 30:
                    self.history[track_id]['positions'].pop(0)
                    self.history[track_id]['timestamps'].pop(0)

        return tracks
