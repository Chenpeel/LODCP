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

# 2. 自定义回调函数（用于记录训练指标）
class MetricLogger:
    def __init__(self, save_dir):
        self.save_path = Path(save_dir) / 'training_metrics.csv'
        self._init_csv_file()

    def _init_csv_file(self):
        headers = [
            'epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
            'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
            'metrics/precision', 'metrics/recall', 'metrics/mAP50', 'metrics/mAP50-95',
            'lr/pg0', 'lr/pg1', 'lr/pg2'  # 不同参数组的学习率
        ]
        with open(self.save_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_metrics(self, epoch, train_metrics, val_metrics, lr_metrics):
        row = [
            epoch,
            train_metrics.get('box_loss', None),
            train_metrics.get('cls_loss', None),
            train_metrics.get('dfl_loss', None),
            val_metrics.get('box_loss', None),
            val_metrics.get('cls_loss', None),
            val_metrics.get('dfl_loss', None),
            val_metrics.get('precision', None),
            val_metrics.get('recall', None),
            val_metrics.get('mAP50', None),
            val_metrics.get('mAP50-95', None),
            *lr_metrics
        ]
        with open(self.save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(colorstr('green', 'bold', f'Metrics saved to {self.save_path}'))

# 3. 训练配置
def train_model():
    # 初始化指标记录器
    save_dir = 'runs/train/v5s_python'
    metric_logger = MetricLogger(save_dir)

    # 训练参数
    args = {
        'weights': 'yolov5s.pt',
        'data': 'data/bdd100k.yaml',
        'epochs': 300,
        'batch_size': 160,
        'imgsz': 640,
        'device': '0',
        'workers': 8,
        'project': 'runs/train',
        'name': 'v5s_python',
        'exist_ok': True,

        # 优化器配置
        'optimizer': 'SGD',
        'lr0': 0.01,
        'momentum': 0.873,
        'weight_decay': 0.0005,

        # 学习率调度
        'cos_lr': True,
        'lrf': 0.01,

        # 数据增强
        'rect': True,
        'augment': True,

        # 回调函数（关键修改点）
        'callbacks': {
            'on_train_epoch_end': lambda epoch, logs: metric_logger.log_metrics(
                epoch,
                train_metrics={
                    'box_loss': logs.get('train/box_loss'),
                    'cls_loss': logs.get('train/cls_loss'),
                    'dfl_loss': logs.get('train/dfl_loss')
                },
                val_metrics={
                    'box_loss': logs.get('val/box_loss'),
                    'cls_loss': logs.get('val/cls_loss'),
                    'dfl_loss': logs.get('val/dfl_loss'),
                    'precision': logs.get('metrics/precision'),
                    'recall': logs.get('metrics/recall'),
                    'mAP50': logs.get('metrics/mAP50'),
                    'mAP50-95': logs.get('metrics/mAP50-95')
                },
                lr_metrics=[
                    logs.get('lr/pg0', 0),  # 骨干网络学习率
                    logs.get('lr/pg1', 0),  # 检测头学习率
                    logs.get('lr/pg2', 0)   # 其他参数学习率
                ]
            )
        }
    }

    # 4. 开始训练
    print(colorstr('blue', 'bold', 'Starting training...'))
    train.run(**args)

if __name__ == '__main__':
    train_model()
