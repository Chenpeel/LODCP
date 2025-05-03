import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fast_scnn import EnhancedFastSCNN

# 图像路径
img_path = '/Users/alpha/Downloads/selfRepo/lodcp/docs/thesis/figures/result.png'

# 结果保存路径
result_path = '/Users/alpha/Downloads/selfRepo/lodcp/docs/thesis/figures/compare.png'

# 模型路径
model_path = '/Users/alpha/Downloads/selfRepo/lodcp/models/fastscnn/weights/best.pth'

# 定义预处理变换
def get_inference_transform():
    return A.Compose([
        A.Resize(320, 800),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# 加载模型
def load_model(model_path, num_classes=2):
    model = EnhancedFastSCNN(num_classes=num_classes)

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    # 从检查点提取模型状态字典
    if "model_state_dict" in checkpoint:
        print("找到model_state_dict，正在加载...")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("直接加载状态字典...")
        model.load_state_dict(checkpoint)

    model.eval()
    return model

# 进行推理
def inference(model, image, transform):
    # 保存原始图像尺寸以便后处理
    orig_h, orig_w = image.shape[:2]

    # 应用转换
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed["image"].unsqueeze(0)  # 添加batch维度

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 获取预测结果
    prob = torch.softmax(output, dim=1)
    pred = torch.argmax(prob, dim=1).squeeze(0).cpu().numpy()

    # 调整预测结果尺寸至原图大小
    pred_resized = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)

    return pred_resized

# 提取图像中的红色车道线
def extract_lane_lines(image):
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 红色线的HSV范围
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 创建掩码
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 形态学操作，连接断线
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    # 将红色线分为左右两侧
    height, width = image.shape[:2]
    mid_x = width // 2

    left_lane = mask_red.copy()
    left_lane[:, mid_x:] = 0  # 清除右半部分

    right_lane = mask_red.copy()
    right_lane[:, :mid_x] = 0  # 清除左半部分

    return left_lane, right_lane

# 水平平移掩码
def shift_mask_horizontally(mask, shift_distance):
    rows, cols = mask.shape
    M = np.float32([[1, 0, shift_distance], [0, 1, 0]])
    shifted_mask = cv2.warpAffine(mask, M, (cols, rows))
    return shifted_mask

# 创建二级预警区
def create_warning_zones(original_img, left_lane, right_lane, shift_distance=50):
    # 创建左右车道线的拷贝
    left_line = left_lane.copy()
    right_line = right_lane.copy()

    # 向左平移左侧线
    shifted_left = shift_mask_horizontally(left_line, -shift_distance)

    # 向右平移右侧线
    shifted_right = shift_mask_horizontally(right_line, shift_distance)

    # 创建左侧预警区域 (原始左线与左移左线之间)
    left_warning_zone = np.zeros_like(left_line)
    for y in range(left_line.shape[0]):
        left_xs = np.where(left_line[y, :] > 0)[0]
        shifted_left_xs = np.where(shifted_left[y, :] > 0)[0]

        if len(left_xs) > 0 and len(shifted_left_xs) > 0:
            min_left_x = np.min(left_xs)
            max_shifted_left_x = np.max(shifted_left_xs)

            if max_shifted_left_x < min_left_x:  # 确保区域正确
                left_warning_zone[y, max_shifted_left_x:min_left_x] = 255

    # 创建右侧预警区域 (原始右线与右移右线之间)
    right_warning_zone = np.zeros_like(right_line)
    for y in range(right_line.shape[0]):
        right_xs = np.where(right_line[y, :] > 0)[0]
        shifted_right_xs = np.where(shifted_right[y, :] > 0)[0]

        if len(right_xs) > 0 and len(shifted_right_xs) > 0:
            max_right_x = np.max(right_xs)
            min_shifted_right_x = np.min(shifted_right_xs)

            if max_right_x < min_shifted_right_x:  # 确保区域正确
                right_warning_zone[y, max_right_x:min_shifted_right_x] = 255

    # 合并左右预警区域
    warning_zone = cv2.bitwise_or(left_warning_zone, right_warning_zone)

    return warning_zone, shifted_left, shifted_right

# 创建车道区域蒙版（左右车道线之间的区域）
def create_lane_area(image_shape, left_lane, right_lane):
    # 创建空白蒙版
    lane_area = np.zeros(image_shape[:2], dtype=np.uint8)

    height, width = image_shape[:2]

    # 对每一行进行处理
    for y in range(height):
        # 找到这一行中左右车道线的位置
        left_xs = np.where(left_lane[y, :] > 0)[0]
        right_xs = np.where(right_lane[y, :] > 0)[0]

        # 如果这一行同时有左右车道线
        if len(left_xs) > 0 and len(right_xs) > 0:
            # 获取左侧线的最右边位置和右侧线的最左边位置
            right_of_left = np.max(left_xs)
            left_of_right = np.min(right_xs)

            # 如果左线在右线左边（正常情况）
            if right_of_left < left_of_right:
                # 填充左右车道线之间的区域
                lane_area[y, right_of_left:left_of_right] = 255

    return lane_area

# 可视化结果
def visualize_results(original_img, prediction, warning_zone, lane_area):
    # 创建可视化图像
    vis_img = original_img.copy()

    # 提取可行驶区域 (class=1)，并限制在车道区域内
    drivable_area = (prediction == 1) & (lane_area > 0)

    # 创建绿色蒙版 (使用半透明绿色表示可行驶区域)
    green_mask = np.zeros_like(original_img)
    green_mask[drivable_area] = [0, 255, 0]  # 绿色

    # 创建红色蒙版 (使用半透明红色表示预警区域)
    red_mask = np.zeros_like(original_img)
    red_mask[warning_zone > 0] = [0, 0, 255]  # 红色

    # 将绿色蒙版与原始图像叠加
    alpha_green = 0.3  # 绿色透明度
    vis_img = cv2.addWeighted(vis_img, 1, green_mask, alpha_green, 0)

    # 将红色蒙版与结果图像叠加
    alpha_red = 0.4  # 红色透明度
    vis_img = cv2.addWeighted(vis_img, 1, red_mask, alpha_red, 0)

    return vis_img

# 显示结果
def display_images(original, processed, title="处理结果对比"):
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 主流程
def main():
    try:
        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像文件: {img_path}")

        # 提取车道线
        print("提取车道线...")
        left_lane, right_lane = extract_lane_lines(img)

        # 保存提取的车道线(用于调试)
        debug_img = img.copy()
        debug_img[left_lane > 0] = [0, 0, 255]  # 左侧红色
        debug_img[right_lane > 0] = [255, 0, 0]  # 右侧蓝色(用于调试区分)
        cv2.imwrite('debug_lane_lines.png', debug_img)

        # 创建车道区域（左右车道线之间的区域）
        lane_area = create_lane_area(img.shape, left_lane, right_lane)
        cv2.imwrite('lane_area.png', lane_area)  # 保存用于调试

        # 创建预警区
        print("创建预警区域...")
        warning_zone, shifted_left, shifted_right = create_warning_zones(img, left_lane, right_lane, shift_distance=40)

        # 加载模型
        print("正在加载模型...")
        model = load_model(model_path)
        print("模型加载成功!")

        # 定义变换
        transform = get_inference_transform()

        # 推理
        print("正在进行模型推理...")
        prediction = inference(model, img, transform)
        print("推理完成!")

        # 可视化结果
        print("生成可视化结果...")
        result_img = visualize_results(img, prediction, warning_zone, lane_area)

        # 保存结果
        cv2.imwrite(result_path, result_img)
        print(f"结果已保存至: {result_path}")

        # 显示结果的统计信息
        road_pixels = np.sum((prediction == 1) & (lane_area > 0))
        lane_pixels = np.sum(lane_area > 0)
        road_percentage = (road_pixels / lane_pixels) * 100 if lane_pixels > 0 else 0
        print(f"车道内可行驶区域占车道面积的 {road_percentage:.2f}%")

        # 显示对比结果
        display_images(img, result_img, "")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
