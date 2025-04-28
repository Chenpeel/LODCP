# Post Process YOLO Model
Use Ultralytics Command

```bash
pip install ultralytics
yolo export model=yolo**.pt format=tflite data=coco.yaml int8=true
...
```
