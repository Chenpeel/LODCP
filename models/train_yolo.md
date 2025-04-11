
```bash
git clone https://github.com/Chenpeel/LODCP.git lodcp

cd lodcp

conda create -n lodcp python=3.11

conda activate lodcp

pip install -r requirements.txt

python3 data_process/getdataset.py

python3 data_process/prepare_bdd100k.py

yolo train model=yolov5s.pt data=data/bdd100k-yolo/bdd100k.yaml epochs=400 batch=160 imgsz=640 device=0 workers=8 project=runs/train name=v5s exist_ok=True optimizer=SGD lr0=0.001 momentum=0.899 weight_decay=0.0006 cos_lr=True lrf=0.001 rect=True augment=True save_period=1 warmup_epochs=20 warmup_momentum=0.8 warmup_bias_lr=0.1 --half
```
