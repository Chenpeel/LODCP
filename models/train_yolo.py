import os
import torch
import warnings
from yolov5 import train
from yolov5.utils.general import colorstr, LOGGER
from yolov5.utils.downloads import attempt_download
from pathlib import Path
from PIL import ImageFont
import yaml

# 修复 ImageFont.getsize 问题
try:
    ImageFont.getsize
except AttributeError:
    def _getsize(font, text):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    ImageFont.getsize = _getsize

# 配置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略所有用户警告

def train_model():
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
        LOGGER.info(f"{colorstr('yellow', 'Downloading weights...')}")
        attempt_download(args['weights'])

    # 检查数据配置
    if not os.path.exists(args['data']):
        raise FileNotFoundError(f"Data config {args['data']} not found!")

    # 加载数据配置
    with open(args['data'], 'r') as f:
        data_config = yaml.safe_load(f)

    LOGGER.info(colorstr('blue', 'bold', '\nTraining parameters:'))
    for k, v in args.items():
        LOGGER.info(f"{k}: {colorstr('white', str(v))}")

    # 使用新的 autocast 调用方式
    try:
        with torch.amp.autocast('cuda'):
            train.run(**args)
        convert_log_to_csv()
    except Exception as e:
        LOGGER.error(f"{colorstr('red', 'Training failed:')} {e}")
        raise

def convert_log_to_csv():
    try:
        log_dir = Path('runs/train/v5s')
        log_file = next(log_dir.glob('*_train.log'), None)

        if log_file and log_file.exists():
            csv_file = log_file.with_suffix('.csv')
            with open(log_file, 'r') as f_in, open(csv_file, 'w') as f_out:
                for line in f_in:
                    if line.strip():
                        f_out.write(','.join(line.split()) + '\n')
            LOGGER.info(colorstr('green', f'Log converted to: {csv_file}'))
    except Exception as e:
        LOGGER.warning(f"Log conversion failed: {e}")

if __name__ == '__main__':
    # 检查CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    # 设置更安全的线程数
    torch.set_num_threads(4)

    train_model()
