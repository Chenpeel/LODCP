import os
import numpy as np
import json
from PIL import Image, ImageDraw
from tqdm import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor
import gc

def load_config(config_path="config.yml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

class DatasetProcessor:
    def __init__(self, batch_size=100, num_workers=4, target_size=(512, 256)):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size

    def process_bdd100k(self, bdd100k_path, json_file="det_v2_train_release.json"):
        """
        处理BDD100K数据集
        Args:
            bdd100k_path: 数据集根目录
            json_file: 标注文件名
        Yields:
            (images, masks) 批次数据
        """
        label_path = os.path.join(bdd100k_path, "labels", json_file)
        try:
            with open(label_path) as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"Error loading annotation file {label_path}: {str(e)}")
            raise

        if "train" in json_file.lower():
            split = "train"
        elif "val" in json_file.lower():
            split = "val"
        else:
            split = "train"

        img_dir = os.path.join(bdd100k_path, "bdd100k", "bdd100k", "images", "100k", split)
        total_anns = len(annotations)

        # 先过滤掉无效的标注
        valid_annotations = []
        for ann in annotations:
            if not isinstance(ann, dict) or "name" not in ann or "labels" not in ann:
                print(f"Invalid annotation format: {ann}")
                continue
            if not isinstance(ann["labels"], list):
                print(f"Invalid labels format in {ann['name']}")
                continue
            valid_annotations.append(ann)

        print(f"Original annotations: {len(annotations)}, Valid annotations: {len(valid_annotations)}")

        for i in tqdm(range(0, len(valid_annotations), self.batch_size), desc="Processing BDD100K"):
            batch_anns = valid_annotations[i:i+self.batch_size]
            images, masks = [], []

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(
                    lambda ann: self._process_bdd_annotation(ann, img_dir),
                    batch_anns
                ))

            # 过滤掉None结果
            valid_results = [r for r in results if r[0] is not None and r[1] is not None]

            if not valid_results:
                continue

            batch_images, batch_masks = zip(*valid_results)
            images.extend(batch_images)
            masks.extend(batch_masks)

            if images:
                yield np.array(images), np.array(masks)
                del images, masks
                gc.collect()

    def _process_bdd_annotation(self, ann, img_dir):
        img_path = os.path.join(img_dir, ann["name"])
        try:
            # 检查图片是否存在
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                return None, None

            # 加载图片
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(self.target_size)
                img_array = np.array(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                return None, None

            # 创建mask
            mask = Image.new("L", self.target_size, 0)
            draw = ImageDraw.Draw(mask)

            has_drivable_area = False
            for label in ann["labels"]:
                if not isinstance(label, dict):
                    continue
                if label.get("category") == "drivable area":
                    if not isinstance(label.get("poly2d", []), list):
                        continue
                    for poly in label.get("poly2d", []):
                        if not isinstance(poly, dict) or not isinstance(poly.get("vertices", []), list):
                            continue
                        try:
                            scaled_vertices = [
                                (x * self.target_size[0] / ann["width"],
                                y * self.target_size[1] / ann["height"])
                                for x, y in poly["vertices"]
                            ]
                            draw.polygon(scaled_vertices, fill=255)
                            has_drivable_area = True
                        except Exception as e:
                            print(f"Error processing polygon in {img_path}: {str(e)}")
                            continue

            if not has_drivable_area:
                print(f"No drivable area found in {img_path}")
                return None, None

            mask_array = np.array(mask)
            if len(mask_array.shape) == 3:
                mask_array = mask_array[..., 0]

            return img_array, mask_array
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None, None

def save_dataset(gen, output_dir):
    """
    保存数据集为多个小文件
    Args:
        gen: 数据生成器
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_num = 0

    for imgs, masks in gen:
        if len(imgs) == 0:
            continue

        np.savez(os.path.join(output_dir, f"batch_{batch_num}.npz"),
                images=imgs, masks=masks)
        batch_num += 1

    class BDDDataset:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.batch_files = sorted(
                [f for f in os.listdir(output_dir) if f.endswith(".npz")],
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
        def __iter__(self):
            for batch_file in self.batch_files:
                data = np.load(os.path.join(self.output_dir, batch_file))
                yield data["images"], data["masks"]

    return BDDDataset(output_dir)

if __name__ == "__main__":
    target_size = tuple(list(config.get("target_size"))[::-1])

    processor = DatasetProcessor(
        batch_size=config.get("batch_size", 300),
        num_workers=config.get("num_workers", 8),
        target_size=target_size
    )

    # 处理训练集
    train_bdd_gen = processor.process_bdd100k(
        config["bdd100k_path"],
        json_file=config.get("bdd_json_train", "det_v2_train_release.json"))
    train_dir = config.get("train_dir", "processed_data/train")
    train_dataset = save_dataset(train_bdd_gen, train_dir)

    # 处理验证集
    val_bdd_gen = processor.process_bdd100k(
        config["bdd100k_path"],
        json_file=config.get("bdd_json_val", "det_v2_val_release.json"))
    val_dir = config.get("val_dir", "processed_data/val")
    val_dataset = save_dataset(val_bdd_gen, val_dir)
