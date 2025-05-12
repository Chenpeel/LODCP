import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

def lane_area_detect(
    frame,
    left_anchor=None,
    right_anchor=None,
    roi_ratio=0.6,
    num_points=30
):
    height, width = frame.shape[:2]

    # 颜色检测范围
    color_ranges = {
        'white': ([0, 0, 200], [255, 30, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255])
    }

    morph_kernel = {
        'small': np.ones((3, 3), np.uint8),
        'medium': np.ones((5, 5), np.uint8),
        'large': np.ones((15, 15), np.uint8)
    }

    roi_vertices = np.array([
        [(0, height), (0, int(height*0.8)), (int(width*0.4), int(height*0.4)),
         (int(width*0.6), int(height*0.4)), (width, int(height*0.8)), (width, height)]
    ], dtype=np.int32)

    dog_scales = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]

    hough_params = {
        'rho': 1,
        'theta': np.pi/180,
        'threshold': 20,
        'minLineLength': 20,
        'maxLineGap': 300
    }

    # 1. CLAHE增强
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. 颜色分割
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    masks = []
    for lower, upper in color_ranges.values():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masks.append(mask)
    color_mask = masks[0]
    for mask in masks[1:]:
        color_mask = cv2.bitwise_or(color_mask, mask)

    # 3. 灰度图
    mask_rgb = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)

    # 4. DoG + Canny 边缘检测
    combined_edges = np.zeros_like(gray)
    for sigma1, sigma2 in dog_scales:
        g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
        dog = g1.astype(np.float64) - g2.astype(np.float64)
        dog_normalized = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, dog_binary = cv2.threshold(dog_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined_edges = cv2.bitwise_or(combined_edges, dog_binary)
    edges_dog = cv2.medianBlur(combined_edges, 3)
    edges_canny = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_or(edges_dog, edges_canny)

    # 5. 形态学增强
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel['large'])
    _, enhanced_edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)
    closed_edges = cv2.morphologyEx(enhanced_edges, cv2.MORPH_CLOSE, morph_kernel['medium'], iterations=2)
    opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, morph_kernel['small'])
    final_edges = cv2.dilate(opened_edges, morph_kernel['small'], iterations=1)
    _, tophat_binary = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    combined_edges = cv2.bitwise_or(final_edges, tophat_binary)

    # 6. ROI掩码
    roi_mask = np.zeros_like(combined_edges)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(combined_edges, roi_mask)

    # 7. 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=hough_params['rho'],
        theta=hough_params['theta'],
        threshold=hough_params['threshold'],
        minLineLength=hough_params['minLineLength'],
        maxLineGap=hough_params['maxLineGap']
    )

    left_pts, right_pts = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.4:
                continue
            if slope < 0:
                left_pts += [(x1, y1), (x2, y2)]
            else:
                right_pts += [(x1, y1), (x2, y2)]

    def fit_ray_with_fixed_point(pts, fixed_x, fixed_y, y_min, y_max, num=30):
        if len(pts) < 2:
            return None
        pts = np.array(pts)
        y = pts[:, 1]
        x = pts[:, 0]
        A = (y - fixed_y).reshape(-1, 1)
        k, _, _, _ = np.linalg.lstsq(A, x - fixed_x, rcond=None)
        k = k[0]
        y_vals = np.linspace(y_min, y_max, num)
        x_vals = k * (y_vals - fixed_y) + fixed_x
        return np.stack([x_vals, y_vals], axis=1).astype(np.int32)

    # 支持自定义端点
    if left_anchor is None:
        left_fixed_x, left_fixed_y = int(width * 0.1), height - 1
    else:
        left_fixed_x, left_fixed_y = left_anchor
    if right_anchor is None:
        right_fixed_x, right_fixed_y = int(width * 0.9), height - 1
    else:
        right_fixed_x, right_fixed_y = right_anchor

    y_min = int(height * roi_ratio)
    y_max = height - 1

    left_line = fit_ray_with_fixed_point(left_pts, left_fixed_x, left_fixed_y, y_min, y_max, num_points)
    right_line = fit_ray_with_fixed_point(right_pts, right_fixed_x, right_fixed_y, y_min, y_max, num_points)

    # 检查交叉，若交叉则返回None
    if left_line is not None and right_line is not None:
        for lpt, rpt in zip(left_line, right_line):
            if lpt[0] >= rpt[0]:
                left_line = None
                right_line = None
                break

    return {"left": left_line, "right": right_line}
