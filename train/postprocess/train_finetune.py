import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/alpha/Downloads/selfRepo/lodcp")
YOLOv5_ROOT = Path("/Users/alpha/Downloads/cloneRepo/yolov5")

sys.path.insert(0, str(YOLOv5_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo import Model

model = Model('yolov5s.yaml').load('pruned_model.pt')
model.train(data='', epochs=50, prune_ratio=0.3)
