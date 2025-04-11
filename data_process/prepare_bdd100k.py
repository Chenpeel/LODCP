import os
import json
import pandas as pd
from tqdm import tqdm
import shutil
import yaml
import random

def convert_bdd100k_to_yolo(bdd_dir, output_dir, data_fraction=1.0):
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    # BDD100K类别映射到YOLO
    category_map = {
        "pedestrian": 0, "rider": 1, "car": 2, "truck": 3,
        "bus": 4, "train": 5, "motorcycle": 6, "bicycle": 7,
        "traffic light": 8, "traffic sign": 9
    }

    # 处理训练集
    print("Processing training set...")
    with open(os.path.join(bdd_dir, "labels", "det_v2_train_release.json")) as f:
        train_anns = json.load(f)

    # 按比例抽样
    if data_fraction < 1.0:
        train_anns = random.sample(train_anns, int(len(train_anns) * data_fraction))
        print(f"Sampled {len(train_anns)} training annotations")

    for ann in tqdm(train_anns):
        img_name = ann["name"]
        img_src = os.path.join(bdd_dir, "bdd100k", "bdd100k", "images", "100k", "train", img_name)
        img_dst = os.path.join(output_dir, "images", "train", img_name)
        label_file = os.path.join(output_dir, "labels", "train", img_name.replace(".jpg", ".txt"))

        # 复制图像
        if not os.path.exists(img_src):
            print(f"Warning: Image {img_src} not found, skipping...")
            continue

        shutil.copy(img_src, img_dst)

        # 创建标签文件 - 处理labels为None的情况
        if ann["labels"] is None:
            # 创建空标签文件
            open(label_file, 'w').close()
            continue

        with open(label_file, "w") as f:
            for obj in ann["labels"]:
                if obj["category"] in category_map:
                    # 转换坐标到YOLO格式 (center_x, center_y, width, height)
                    x1 = obj["box2d"]["x1"]
                    y1 = obj["box2d"]["y1"]
                    x2 = obj["box2d"]["x2"]
                    y2 = obj["box2d"]["y2"]

                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2

                    # 归一化
                    center_x /= 1280.0
                    center_y /= 720.0
                    width /= 1280.0
                    height /= 720.0

                    f.write(f"{category_map[obj['category']]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    # 处理验证集 (同上)
    print("Processing validation set...")
    with open(os.path.join(bdd_dir, "labels", "det_v2_val_release.json")) as f:
        val_anns = json.load(f)

    # 按比例抽样
    if data_fraction < 1.0:
        val_anns = random.sample(val_anns, int(len(val_anns) * data_fraction))
        print(f"Sampled {len(val_anns)} validation annotations")

    for ann in tqdm(val_anns):
        img_name = ann["name"]
        img_src = os.path.join(bdd_dir, "bdd100k", "bdd100k", "images", "100k", "val", img_name)
        img_dst = os.path.join(output_dir, "images", "val", img_name)
        label_file = os.path.join(output_dir, "labels", "val", img_name.replace(".jpg", ".txt"))

        if not os.path.exists(img_src):
            print(f"Warning: Image {img_src} not found, skipping...")
            continue

        shutil.copy(img_src, img_dst)

        if ann["labels"] is None:
            open(label_file, 'w').close()
            continue

        with open(label_file, "w") as f:
            for obj in ann["labels"]:
                if obj["category"] in category_map:
                    x1 = obj["box2d"]["x1"]
                    y1 = obj["box2d"]["y1"]
                    x2 = obj["box2d"]["x2"]
                    y2 = obj["box2d"]["y2"]

                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2

                    center_x /= 1280.0
                    center_y /= 720.0
                    width /= 1280.0
                    height /= 720.0

                    f.write(f"{category_map[obj['category']]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    # 创建数据集配置文件
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(category_map),
        'names': list(category_map.keys())
    }

    with open(os.path.join(output_dir, "bdd100k.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    print(f"Dataset conversion complete. YOLO format dataset saved to {output_dir}")

if __name__ == "__main__":
    data_fraction = 0.5
    convert_bdd100k_to_yolo("data/bdd100k-dataset", "data/bdd100k-yolo", data_fraction=data_fraction)
