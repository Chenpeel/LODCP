import yaml
import os
import glob
import cv2
import numpy as np
import random

# 你的YAML路径
yaml_path = "/Users/alpha/Downloads/selfRepo/lodcp/data/bdd100k-yolo/bdd100k.yaml"

# 读取YAML文件
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# 提取训练图片文件夹路径
train_dir = data.get('train', None)
if train_dir is None:
    print("没有找到train路径")
    exit()

# 读取训练图片路径（只列出图片文件, 支持jpg/png）
image_files = glob.glob(os.path.join(train_dir, '**', '*.jpg'), recursive=True)
image_files += glob.glob(os.path.join(train_dir, '**', '*.png'), recursive=True)

# 随机抽取部分图片作为校准样本（比如100张）
num_samples = min(300, len(image_files))
sample_files = random.sample(image_files, num_samples)

# 加载图片并生成校准样本数组
calib_data = []

for img_path in sample_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    # resize为模型输入大小（假设为320x320）
    img_resized = cv2.resize(img, (320, 320))
    # 转为float32，归一化
    img_resized = img_resized.astype(np.float32) / 255.0
    calib_data.append(img_resized)

# 转为numpy数组
calib_np = np.array(calib_data)

# 保存为.npy文件
np.save("/Users/alpha/Downloads/selfRepo/lodcp/data/calibration_samples.npy", calib_np)
print("校准数据已保存：calibration_samples.npy")
