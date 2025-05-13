import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from main_py.lane_area_detect import lane_area_detect
from main_py.ttc import ttc_predict
from main_py.fast_scnn import FastSCNN
from collections import deque, defaultdict

# 初始化模型
yolo = YOLO("models/best.pt")
deepsort = DeepSort(max_age=30)
fast_scnn = FastSCNN("models/best.pth")

video = cv2.VideoCapture("/resource/input7.mp4")
w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
fps = video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

ROI_POLYGON = np.array(
    [
        [
            (0, h),
            (0, int(h * 0.5)),
            (int(w * 0.4), int(h * 0.2)),
            (int(w * 0.6), int(h * 0.2)),
            (w, int(h * 0.5)),
            (w, h),
        ]
    ],
    dtype=np.int32,
)
ROI_MASK = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(ROI_MASK, ROI_POLYGON, 1)

prev_bboxes = {}
lane_history = {"left": deque(maxlen=20), "right": deque(maxlen=20)}
height_history = defaultdict(lambda: deque(maxlen=5))
ttc_history = defaultdict(lambda: deque(maxlen=5))


def average_lane_points(lane_points_list):
    if not lane_points_list:
        return None
    arr = np.array(lane_points_list)
    mean_pts = np.mean(arr, axis=0)
    return mean_pts.astype(np.int32)


def box_in_roi(box, roi_mask):
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if 0 <= cx < roi_mask.shape[1] and 0 <= cy < roi_mask.shape[0]:
        return roi_mask[cy, cx] > 0
    return False


def expand_polyline(pts, outward=True, expand_width=50):
    expanded = []
    n = len(pts)
    for i in range(n):
        if i == 0:
            dx = pts[i + 1][0] - pts[i][0]
            dy = pts[i + 1][1] - pts[i][1]
        elif i == n - 1:
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
        else:
            dx = pts[i + 1][0] - pts[i - 1][0]
            dy = pts[i + 1][1] - pts[i - 1][1]
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        nx = -dy / norm
        ny = dx / norm
        if not outward:
            nx, ny = -nx, -ny
        ex = int(pts[i][0] + nx * expand_width)
        ey = int(pts[i][1] + ny * expand_width)
        expanded.append([ex, ey])
    return np.array(expanded, dtype=np.int32)


frame_idx = 0
FRAME_INTERVAL = 1.0 / fps if fps > 0 else 1.0 / 30

while True:
    ret, frame = video.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    current_time = frame_idx * FRAME_INTERVAL

    # 1. 传统车道线检测，支持端点范围
    lanes = lane_area_detect(
        frame,
        left_anchor_range=((0, h - 1), (int(0.25 * w), h - 1)),
        right_anchor_range=((int(0.75 * w), h - 1), (w - 1, h - 1)),
    )
    for side in ["left", "right"]:
        if lanes[side] is not None:
            lane_history[side].append(lanes[side])

    # 2. FastSCNN语义分割，仅保留ROI区域
    seg_mask = fast_scnn.predict(frame)
    seg_mask = seg_mask * ROI_MASK

    # 3. YOLOv5目标检测
    yolo_results = yolo(frame, conf=0.6)
    detections = []
    for r in yolo_results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = r
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))

    # 4. DeepSORT目标跟踪
    tracks = deepsort.update_tracks(detections, frame=frame)
    track_infos = []
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = t.to_ltrb()
        vx, vy = t.last_velocity if hasattr(t, "last_velocity") else (0, 0)
        cls = t.det_class if hasattr(t, "det_class") else 0
        prev_bbox = prev_bboxes.get(t.track_id, [x1, y1, x2, y2])
        track_infos.append(
            {
                "track_id": t.track_id,
                "bbox": [x1, y1, x2, y2],
                "prev_bbox": prev_bbox,
                "vx": vx,
                "vy": vy,
                "class_id": cls,
            }
        )
        prev_bboxes[t.track_id] = [x1, y1, x2, y2]

    # 只保留ROI内目标
    track_infos_roi = [t for t in track_infos if box_in_roi(t["bbox"], ROI_MASK)]

    # 更新高度历史
    for t in track_infos_roi:
        x1, y1, x2, y2 = map(int, t["bbox"])
        h_box = y2 - y1
        height_history[t["track_id"]].append(h_box)

    # 6. TTC碰撞预测（只对ROI内目标），三级风险因子
    ttc_results = ttc_predict(
        track_infos_roi,
        lanes,
        seg_mask,
        height_history,
        risk_factor_in_lane=1.1,
        risk_factor_adjacent=0.66,
        risk_factor_outside=0.28,
        fps=int(fps) if fps > 0 else 30,
    )

    # 新增：TTC平滑递减
    for ttc_result in ttc_results:
        tid = ttc_result["track_id"]
        new_ttc = ttc_result["ttc"]
        hist = ttc_history[tid]
        if hist:
            prev_ttc = hist[-1]
            if new_ttc > prev_ttc:
                new_ttc = min(new_ttc, prev_ttc + FRAME_INTERVAL)
            else:
                new_ttc = max(new_ttc, prev_ttc - FRAME_INTERVAL)
        hist.append(new_ttc)
        ttc_result["ttc"] = new_ttc
        # 全局碰撞时间戳
        ttc_result["collision_time"] = current_time + new_ttc

    # 7. 可视化
    vis = frame.copy()
    # 画ROI区域
    cv2.polylines(vis, ROI_POLYGON, isClosed=True, color=(0, 255, 0), thickness=2)
    # 画平滑后的车道线
    draw_lane = True
    frame_area = frame.shape[0] * frame.shape[1] * 0.4
    for t in track_infos_roi:
        x1, y1, x2, y2 = map(int, t["bbox"])
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if box_area / frame_area > 0.77:
            draw_lane = False
            break
    if draw_lane:
        for side in ["left", "right"]:
            if lane_history[side]:
                avg_pts = average_lane_points(list(lane_history[side]))
                if avg_pts is not None:
                    cv2.polylines(vis, [avg_pts], False, (0, 255, 255), 2)
        expand_width = 50
        if lane_history["left"] and lane_history["right"]:
            left_pts = average_lane_points(list(lane_history["left"]))
            right_pts = average_lane_points(list(lane_history["right"]))
            if left_pts is not None and right_pts is not None:
                left_expanded = expand_polyline(
                    left_pts, outward=True, expand_width=expand_width
                )
                right_expanded = expand_polyline(
                    right_pts, outward=False, expand_width=expand_width
                )
                area_poly = np.vstack(
                    [left_pts, left_expanded[::-1], right_expanded, right_pts[::-1]]
                )
                area_poly = area_poly.reshape((-1, 1, 2))
                blue_mask = np.zeros_like(vis)
                cv2.fillPoly(blue_mask, [area_poly], (255, 0, 0))
                vis = cv2.addWeighted(vis, 1, blue_mask, 0.3, 0)
    # 画分割掩码
    color_mask = np.zeros_like(vis)
    color_mask[seg_mask == 1] = (0, 255, 0)
    vis = cv2.addWeighted(vis, 1, color_mask, 0.3, 0)
    # 画检测框和跟踪ID
    for t in track_infos_roi:
        x1, y1, x2, y2 = map(int, t["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f'ID:{t["track_id"]}', (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
    # 画TTC和风险分数
    for ttc in ttc_results:
        x1, y1, x2, y2 = map(int, ttc["bbox"])
        color = (
            (0, 0, 255)
            if ttc["risk_score"] > 0.5
            else (0, 255, 255) if ttc["risk_score"] > 0.2 else (0, 255, 0)
        )
        cv2.putText(
            vis,
            f'TTC:{ttc["ttc"]:.2f}s R:{ttc["risk_score"]:.2f} T:{ttc["collision_time"]:.2f}s',
            (x1, y2 + 15),
            0,
            0.4,
            color,
            1,
        )

    out.write(vis)
    cv2.imshow("Result", vis)
    if cv2.waitKey(1) == 27:
        break

    frame_idx += 1

video.release()
out.release()
cv2.destroyAllWindows()
