import os
import kagglehub
import shutil
import warnings
warnings.simplefilter("ignore")

data_dir = "data"

def prepare_dir():
    # 创建数据目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

def task1():
    kitti_dest = os.path.join(data_dir, "kitti-segmentation")
    if not os.path.exists(kitti_dest):
        print("正在下载KITTI分割数据集...")
        kitti_path = kagglehub.dataset_download("sakshaymahna/kittiroadsegmentation")
        shutil.move(kitti_path, kitti_dest)
    else:
        print("KITTI分割数据集已存在")
def task2():
    bdd_dest = os.path.join(data_dir, "bdd100k-dataset")
    if not os.path.exists(bdd_dest):
        print("正在下载BDD100K数据集...")
        bdd_path = kagglehub.dataset_download("awsaf49/bdd100k-dataset")
        shutil.move(bdd_path, bdd_dest)
    else:
        print("BDD100K数据集已存在")
def task3():
    kitti_dest = os.path.join(data_dir, "kitti-road")
    if not os.path.exists(kitti_dest):
        print("正在下载KITTI道路数据集...")
        kitti_path = kagglehub.dataset_download("sumanyughoshal/kitti-road-dataset")
        shutil.move(kitti_path, kitti_dest)
    else:
        print("KITTI道路数据集已存在")

if __name__ == "__main__":
    prepare_dir()
    task2()
