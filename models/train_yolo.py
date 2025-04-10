
import os
import yaml
from yolov5 import train

# 准备数据集配置
def create_yaml(data_dir, classes):
    yaml_content = {
        'path': data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open('dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

# 使用BDD100K数据集
classes = ['road', 'lane', 'vehicle', 'pedestrian', 'traffic sign']
create_yaml('data/bdd100k-dataset', classes)

# 训练参数配置
train.run(
    data='dataset.yaml',
    imgsz=640,
    batch_size=16,
    epochs=100,
    weights='yolov5s.pt',
    project='runs/train',
    name='yolov5s_bdd100k',
)
