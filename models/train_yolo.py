import os
import yaml
from yolov5 import train
from yolov5.utils.general import colorstr
from pathlib import Path

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

def train_model():
    create_bdd100k_yaml()

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
        'augment': True,
        'save_period': 1,
    }

    print(colorstr('blue', 'bold', 'Starting training...'))
    train.run(**args)

    # 训练完成后自动转换日志为CSV
    convert_log_to_csv()

def convert_log_to_csv():
    log_file = next(Path('runs/train/v5s').glob('*_train.log'), None)
    if log_file:
        csv_file = log_file.with_suffix('.csv')
        with open(log_file) as f_in, open(csv_file, 'w') as f_out:
            f_out.write(f_in.read().replace(' ', ','))
        print(colorstr('green', f'Log converted to {csv_file}'))

if __name__ == '__main__':
    train_model()
