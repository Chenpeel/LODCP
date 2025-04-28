# Train YOLO Model

```bash
pip install ultralytics
yolo train model=yolo**.pt data=coco.yaml epochs=100 batch=16 workers=8 cache=ram hyp=hyp.yml
...
```
