import os
import yaml
import torch
from yolov5 import train
from yolov5.utils.general import colorstr, LOGGER
from yolov5.utils.downloads import attempt_download
from pathlib import Path
import warnings

# 配置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像站
warnings.filterwarnings("ignore", message="torch.cuda.amp.autocast.*")  # 临时忽略警告

def train_model():
    # 训练参数配置
    args = {
        'weights': 'yolov5s.pt',
        'data': 'data/bdd100k-yolo/bdd100k.yaml',
        'epochs': 400,
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

    # 确保模型文件存在
    if not os.path.exists(args['weights']):
        LOGGER.info(f"{colorstr('yellow', 'bold', 'Downloading weights...')}")
        attempt_download(args['weights'])

    # 检查数据配置文件
    if not os.path.exists(args['data']):
        raise FileNotFoundError(f"Data config file {args['data']} not found!")

    print(colorstr('blue', 'bold', '\nStarting training with parameters:'))
    for k, v in args.items():
        print(f"{colorstr('green', k)}: {colorstr('white', str(v))}")

    try:
        # 开始训练
        train.run(**args)

        # 训练完成后处理日志
        convert_log_to_csv()

    except Exception as e:
        LOGGER.error(f"{colorstr('red', 'bold', 'Training failed:')}")
        LOGGER.error(e)
        raise

def convert_log_to_csv():
    """将训练日志转换为CSV格式"""
    try:
        log_dir = Path('runs/train/v5s')
        log_file = next(log_dir.glob('*_train.log'), None)

        if log_file and log_file[0].exists():
            csv_file = log_file[0].with_suffix('.csv')

            # 读取日志并转换格式
            with open(log_file[0], 'r') as f_in:
                lines = [line.strip().split() for line in f_in if line.strip()]

            # 写入CSV
            with open(csv_file, 'w') as f_out:
                for line in lines:
                    f_out.write(','.join(line) + '\n')

            print(colorstr('green', f'\nLog converted to: {csv_file}'))
        else:
            print(colorstr('yellow', 'No training log file found!'))

    except Exception as e:
        LOGGER.warning(f"{colorstr('yellow', 'Log conversion failed:')}")
        LOGGER.warning(e)

if __name__ == '__main__':
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    # 设置PyTorch自动混合精度（兼容新旧版本）
    if hasattr(torch, 'amp'):
        torch.autocast('cuda')  # PyTorch 2.0+
    else:
        torch.cuda.amp.autocast()  # 旧版本兼容

    train_model()
