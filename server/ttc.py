import numpy as np


def point_lane_position(x, y, left_line, right_line):
    if left_line is None or right_line is None:
        return "unknown"
    idx = np.argmin(np.abs(left_line[:, 1] - y))
    x_left = left_line[idx, 0]
    x_right = right_line[idx, 0]
    if x_left < x < x_right:
        return "in_lane"
    elif x <= x_left:
        return "left"
    elif x >= x_right:
        return "right"
    else:
        return "unknown"


def ttc_predict(
    tracks,
    lanes,
    seg_mask,
    height_history=None,
    risk_factor_in_lane=1.1,
    risk_factor_adjacent=0.66,
    risk_factor_outside=0.27,
    fps=60,
    ttc_history=None,
    ttc_missing_max=10,
):
    MIN_TTC = 0.001
    MAX_TTC = 3.0

    def get_real_height(class_id):
        if class_id == 0:
            return 1.7
        else:
            return 1.5

    left_line = lanes.get("left")
    right_line = lanes.get("right")

    # 用于遮挡期间TTC递减补偿
    if ttc_history is None:
        ttc_history = {}

    # 标记本帧出现的track_id
    current_ids = set([t["track_id"] for t in tracks])

    results = []
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        prev_bbox = t.get("prev_bbox", [x1, y1, x2, y2])
        h = y2 - y1
        prev_h = prev_bbox[3] - prev_bbox[1]
        class_id = t.get("class_id", 2)
        real_height = get_real_height(class_id)

        # 距离估算
        distance = real_height / (h + 1e-3)
        prev_distance = real_height / (prev_h + 1e-3) if prev_h > 0 else distance

        # 速度估算（多帧平滑）
        if (
            height_history is not None
            and t["track_id"] in height_history
            and len(height_history[t["track_id"]]) >= 2
        ):
            h_list = list(height_history[t["track_id"]])
            d_list = [real_height / (hh + 1e-3) for hh in h_list]
            relative_speed = (d_list[-1] - d_list[0]) / ((len(d_list) - 1) / fps)
        else:
            relative_speed = (distance - prev_distance) * fps

        # 车道线分级
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        lane_pos = point_lane_position(cx, cy, left_line, right_line)

        # TTC计算（目标靠近时才有意义，远离或静止时TTC为MAX_TTC）
        if lane_pos == "in_lane" and relative_speed < -1e-3:
            ttc = distance / (-relative_speed)
        else:
            ttc = MAX_TTC

        ttc = np.clip(ttc, MIN_TTC, MAX_TTC)

        # 遮挡期间递减补偿
        tid = t["track_id"]
        if tid in ttc_history:
            prev_ttc, missing_count = ttc_history[tid]
            if tid in current_ids:
                # 目标可见，正常更新
                if ttc > prev_ttc:
                    ttc = min(ttc, prev_ttc + 1.0 / fps)
                else:
                    ttc = max(ttc, prev_ttc - 1.0 / fps)
                ttc_history[tid] = (ttc, 0)
            else:
                # 目标短暂消失，递减补偿
                if missing_count < ttc_missing_max:
                    ttc = max(prev_ttc - 1.0 / fps, MIN_TTC)
                    ttc_history[tid] = (ttc, missing_count + 1)
                else:
                    ttc = MAX_TTC
                    ttc_history[tid] = (ttc, ttc_missing_max)
        else:
            ttc_history[tid] = (ttc, 0)

        # 区域风险因子
        if lane_pos == "in_lane":
            risk_factor = risk_factor_in_lane
        elif lane_pos in ("left", "right"):
            risk_factor = risk_factor_adjacent
        else:
            risk_factor = risk_factor_outside

        # 归一化风险分数
        collision_prob = 3 - np.exp(ttc / MAX_TTC)
        risk_score = (
            risk_factor * collision_prob if risk_factor * collision_prob < 1 else 0.99
        )

        results.append(
            {
                "track_id": tid,
                "bbox": t["bbox"],
                "ttc": ttc,
                "risk_score": risk_score,
                "lane_pos": lane_pos,
                "risk_factor": risk_factor,
                "collision_prob": collision_prob,
            }
        )
    return results
