import os
import yaml
from yolov5 import train
import torch

# 1. 准备数据集配置
def create_bdd100k_yaml():
    yaml_content = {
        'path': 'data/bdd100k-yolo',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 5,  # 类别数
        'names': ['road', 'lane', 'vehicle', 'pedestrian', 'traffic sign']
    }
    os.makedirs('data', exist_ok=True)
    with open('data/bdd100k.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

create_bdd100k_yaml()

# 2. 训练配置
def train_model():
    args = {
        'weights': 'yolov5s.pt',
        'data': 'data/bdd100k.yaml',
        'epochs': 300,
        'batch_size': 160,
        'imgsz': 640,
        'device': '0',  # GPU设备ID
        'workers': 8,
        'project': 'runs/train',
        'name': 'v5s',
        'exist_ok': True,  # 允许覆盖现有实验

        # 优化器配置
        'optimizer': 'SGD',  #'SGD', 'Adam', 'AdamW'
        'lr0': 0.01,        # 初始学习率
        'momentum': 0.873,  # SGD动量
        'weight_decay': 0.0005,

        # 学习率调度
        'cos_lr': True,     # 余弦退火调度
        'lrf': 0.001,        # 最终学习率 = lr0 * lrf

        # 数据增强
        'rect': True,       # 矩形训练
        'augment': True,    # 启用Mosaic等增强

        # 日志和保存
        'save_period': 10,  # 每10个epoch保存一次权重
        'bbox_interval': -1,  # 验证阶段绘制预测框
        'artifact_alias': 'latest',

        # 高级配置
        'patience': 50,    # 早停等待epoch数
        'freeze': [0],     # 冻结前n层(0表示不冻结)
    }

    # 3. 开始训练
    train.run(**args)

    # 4. 训练后处理
    print("Training completed!")
    print(f"Results saved to runs/train/{args['name']}")

if __name__ == '__main__':
    train_model()
