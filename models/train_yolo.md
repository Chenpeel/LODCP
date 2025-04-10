
```bash
git clone https://github.com/Chenpeel/LODCP.git lodcp

cd lodcp

conda create -n lodcp python=3.11

conda activate lodcp

pip install -r requirements.txt

python3 data_process/getdataset.py

python3 data_process/prepare.py

yolo train \
    data=data/bdd100k-yolo/bdd100k.yaml \
    model=yolov5s.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=6 \
    project=runs/train \
    name=bdd100k_yolov5s
```
