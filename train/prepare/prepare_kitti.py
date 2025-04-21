import cv2
import numpy as np
from pathlib import Path
import re

def extract_base_name(label_name):
    """从标注文件名提取对应的原始图像名"""
    # 处理 um_road_000000.png → um_000000.png
    # 处理 um_lane_000000.png → um_000000.png
    # 处理 umm_road_000000.png → umm_000000.png
    # 处理 uu_road_000000.png → uu_000000.png
    pattern = r'^(um|umm|uu)_(?:road|lane)_(\d+)\.png$'
    match = re.match(pattern, label_name)
    if match:
        prefix = match.group(1)
        number = match.group(2)
        return f"{prefix}_{number.zfill(6)}.png"
    return None

def create_road_mask(label_path, output_dir):
    """将KITTI标注图像转换为二值掩码"""
    label = cv2.imread(str(label_path))
    if label is None:
        print(f"无法读取标注文件: {label_path}")
        return None

    # 创建可行驶道路掩码 (品红色区域)
    road_mask = np.all(label == [255, 0, 255], axis=-1).astype(np.uint8) * 255

    # 生成正确的掩码文件名
    base_name = extract_base_name(label_path.name)
    if not base_name:
        print(f"无法解析文件名格式: {label_path.name}")
        return None

    mask_path = output_dir / f"mask_{base_name}"
    cv2.imwrite(str(mask_path), road_mask)
    return mask_path

def verify_file_mapping(root_dir):
    """验证图像和标注文件的对应关系"""
    root = Path(root_dir)
    image_files = {f.name for f in (root/"image_2").glob("*.png")}
    label_files = {f.name for f in (root/"gt_image_2").glob("*.png")}

    missing_images = set()
    for label in label_files:
        base = extract_base_name(label)
        if base and base not in image_files:
            missing_images.add(label)

    if missing_images:
        print(f"警告: 发现 {len(missing_images)} 个标注文件没有对应的原始图像")
        for f in sorted(missing_images)[:3]:
            print(f"  - {f}")
    else:
        print("所有标注文件都有对应的原始图像")

def process_kitti_dataset(root_dir):
    """批量处理KITTI数据集"""
    root = Path(root_dir)
    output_dir = root / "masks"
    output_dir.mkdir(exist_ok=True)

    # 先验证文件对应关系
    verify_file_mapping(root_dir)

    # 处理所有标注图像
    success_count = 0
    for label_path in sorted((root/"gt_image_2").glob("*.png")):
        if create_road_mask(label_path, output_dir):
            success_count += 1

    print(f"成功生成 {success_count}/{len(list((root/'gt_image_2').glob('*.png')))} 个掩码文件")

if __name__ == "__main__":
    data_root = "data/kitti-road/data_road/training"
    process_kitti_dataset(data_root)
