import os
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw
from tqdm import tqdm  # 进度条支持
import yaml  # 替代JSON读取.env
from concurrent.futures import ThreadPoolExecutor
import gc  # 垃圾回收

# 加载配置
def load_config(config_path="config.yml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

class DatasetProcessor:
    def __init__(self, batch_size=100, num_workers=4, target_size=(512, 256)):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size

    def process_kitti(self, kitti_path, cache_dir=None):
        """
        分批处理KITTI数据集，支持缓存
        Args:
            kitti_path: KITTI数据集根目录
            cache_dir: 预处理结果缓存目录
        Yields:
            (images, masks) 批次数据
        """
        if cache_dir and os.path.exists(os.path.join(cache_dir, "kitti_cache.npz")):
            data = np.load(os.path.join(cache_dir, "kitti_cache.npz"))
            yield data["images"], data["masks"]
            return

        image_dir = os.path.join(kitti_path, "data_road/training/image_2")
        label_dir = os.path.join(kitti_path, "data_road/training/gt_image_2")

        img_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        total_files = len(img_files)

        # 分批处理
        for i in tqdm(range(0, total_files, self.batch_size), desc="Processing KITTI"):
            batch_files = img_files[i:i+self.batch_size]
            images, masks = [], []

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(
                    lambda f: self._process_kitti_image(image_dir, label_dir, f),
                    batch_files
                ))

            for img, mask in results:
                if img is not None and mask is not None:
                    images.append(img)
                    masks.append(mask)

            if images:
                images = np.array(images)
                masks = np.array(masks)

                if cache_dir and i == 0:  # 只缓存第一批作为示例
                    os.makedirs(cache_dir, exist_ok=True)
                    np.savez(os.path.join(cache_dir, "kitti_cache.npz"),
                            images=images, masks=masks)

                yield images, masks
                del images, masks
                gc.collect()

    def _process_kitti_image(self, image_dir, label_dir, img_file):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            return None, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, f"{base_name}_road.png")

        if os.path.exists(label_path):
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            return img, mask
        return None, None

    def process_bdd100k(self, bdd100k_path, json_file="det_v2_train_release.json"):
        """
        分批处理BDD100K数据集
        Args:
            bdd100k_path: 数据集根目录
            json_file: 标注文件名
        Yields:
            (images, masks) 批次数据
        """
        label_path = os.path.join(bdd100k_path, "labels", json_file)
        with open(label_path) as f:
            annotations = json.load(f)
        if "train" in json_file.lower():
            split = "train"
        elif "val" in json_file.lower():
            split = "val"
        else:
            split = "train"

        img_dir = os.path.join(bdd100k_path, "bdd100k", "bdd100k", "images", "100k", split)
        total_anns = len(annotations)

        # 分批处理
        for i in tqdm(range(0, total_anns, self.batch_size), desc="Processing BDD100K"):
            batch_anns = annotations[i:i+self.batch_size]
            images, masks = [], []

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(
                    lambda ann: self._process_bdd_annotation(ann, img_dir),
                    batch_anns
                ))

            for img, mask in results:
                if img is not None and mask is not None:
                    images.append(img)
                    masks.append(mask)

            if images:
                yield np.array(images), np.array(masks)
                del images, masks
                gc.collect()

    def _process_bdd_annotation(self, ann, img_dir):
        img_path = os.path.join(img_dir, ann["name"])
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.target_size)

            mask = Image.new("L", self.target_size, 0)
            draw = ImageDraw.Draw(mask)

            for label in ann["labels"]:
                if label["category"] == "drivable area":
                    for poly in label.get("poly2d", []):
                        # 坐标缩放至target_size
                        scaled_vertices = [
                            (x * self.target_size[0] / ann["width"],
                             y * self.target_size[1] / ann["height"])
                            for x, y in poly["vertices"]
                        ]
                        draw.polygon(scaled_vertices, fill=255)

            return np.array(img), np.array(mask)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None, None

def combine_and_save_datasets(kitti_gen, bdd_gen, output_dir):
    """
    合并数据集并保存为多个小文件
    Args:
        kitti_gen: KITTI数据生成器(可为None)
        bdd_gen: BDD100K数据生成器(可为None)
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_num = 0

    # 处理KITTI数据(如果有)
    if kitti_gen is not None:
        for kitti_imgs, kitti_masks in kitti_gen:
            np.savez(os.path.join(output_dir, f"kitti_batch_{batch_num}.npz"),
                    images=kitti_imgs, masks=kitti_masks)
            batch_num += 1

    # 处理BDD100K数据(如果有)
    if bdd_gen is not None:
        for bdd_imgs, bdd_masks in bdd_gen:
            np.savez(os.path.join(output_dir, f"bdd_batch_{batch_num}.npz"),
                    images=bdd_imgs, masks=bdd_masks)
            batch_num += 1

    # 创建合并后的数据加载器
    class MergedDataset:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.batch_files = sorted(
                [f for f in os.listdir(output_dir) if f.endswith(".npz")],
                key=lambda x: int(x.split("_")[2].split(".")[0]),
            )
        def __iter__(self):
            for batch_file in self.batch_files:
                data = np.load(os.path.join(self.output_dir, batch_file))
                yield data["images"], data["masks"]

    return MergedDataset(output_dir)

if __name__ == "__main__":
    # 配置参数
    target_size =tuple(list(config.get("target_size"))[::-1])

    processor = DatasetProcessor(
        batch_size=config.get("batch_size", 289),
        num_workers=config.get("num_workers", 8),
        target_size=target_size
    )
    # 创建数据生成器
    kitti_gen = processor.process_kitti(
        config["kitti_path"],
        cache_dir=config.get("cache_dir"))

    train_bdd_gen = processor.process_bdd100k(
        config["bdd100k_path"],
        json_file=config.get("bdd_json_train", "det_v2_train_release.json"))

    # 合并并保存数据集
    train_dir = config.get("train_dir", "processed_data/train")
    train_dataset = combine_and_save_datasets(kitti_gen, train_bdd_gen, train_dir)

    val_bdd_gen = processor.process_bdd100k(
        config["bdd100k_path"],
        json_file=config.get("bdd_json_val", "det_v2_val_release.json"))
    val_dir = config.get("val_dir", "processed_data/val")
    val_dataset = combine_and_save_datasets(None, val_bdd_gen, val_dir)
