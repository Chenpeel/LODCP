import os
import yaml
import csv
from yolov5 import train
from yolov5.utils.general import colorstr
from pathlib import Path

# 1. 准备数据集配置
def create_bdd100k_yaml():
    yaml_content = {
        'path': 'data/bdd100k-yolo',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 5,
        'names': ['road', 'lane', 'vehicle', 'pedestrian', 'traffic sign']
    }
    os.makedirs('data', exist_ok=True)
    with open('data/bdd100k.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

create_bdd100k_yaml()

# 2. 自定义回调类（替代lambda函数）
class CustomMetricLogger:
    def __init__(self, save_dir):
        self.save_path = Path(save_dir) / 'training_metrics.csv'
        self._init_csv_file()

    def _init_csv_file(self):
        headers = [
            'epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
            'metrics/precision', 'metrics/recall', 'metrics/mAP50', 'metrics/mAP50-95',
            'lr/pg0', 'lr/pg1', 'lr/pg2'
        ]
        with open(self.save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_metrics(self, epoch, logs):
        row = [
            epoch,
            logs.get('train/box_loss'),
            logs.get('train/cls_loss'),
            logs.get('train/dfl_loss'),
            logs.get('val/box_loss'),
            logs.get('val/cls_loss'),
            logs.get('val/dfl_loss'),
            logs.get('metrics/precision'),
            logs.get('metrics/recall'),
            logs.get('metrics/mAP50'),
            logs.get('metrics/mAP50-95'),
            logs.get('lr/pg0', 0),
            logs.get('lr/pg1', 0),
            logs.get('lr/pg2', 0)
        ]
        with open(self.save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(colorstr('green', 'bold', f'Metrics saved to {self.save_path}'))

# 3. 训练配置
def train_model():
    # 初始化指标记录器
    save_dir = 'runs/train/v5s'
    os.makedirs(save_dir, exist_ok=True)
    metric_logger = CustomMetricLogger(save_dir)

    args = {
        'weights': 'yolov5s.pt',
        'data': 'data/bdd100k.yaml',
        'epochs': 300,
        'batch_size': 160,
        'imgsz': 640,
        'device': '0',
        'workers': 8,
        'project': 'runs/train',
        'name': 'v5s',
        'exist_ok': True,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'momentum': 0.873,
        'weight_decay': 0.0005,
        'cos_lr': True,
        'lrf': 0.001,
        'rect': True,
        'augment': True
    }

    # 4. 开始训练（通过修改全局callbacks添加记录器）
    original_callbacks = train.get_callbacks()

    def wrapped_callbacks():
        callbacks = original_callbacks()
        callbacks.on_train_epoch_end = lambda epoch, logs: metric_logger.log_metrics(epoch, logs)
        return callbacks

    train.set_callbacks(wrapped_callbacks)

    print(colorstr('blue', 'bold', 'Starting training...'))
    train.run(**args)

if __name__ == '__main__':
    train_model()
