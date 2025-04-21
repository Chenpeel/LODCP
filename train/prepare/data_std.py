import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. 自定义数据集类
class KittiRoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # 建立图像-掩码映射关系
        self.samples = []
        missing_masks = []

        for img_path in sorted(self.img_dir.glob("*.png")):
            mask_path = self.mask_dir / f"mask_{img_path.name}"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                missing_masks.append(img_path.name)

        if missing_masks:
            print(f"警告: 共缺失 {len(missing_masks)} 个掩码文件")
            print("示例如下:")
            for name in missing_masks[:3]:
                print(f"  - {name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # 读取图像和掩码（增加错误处理）
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图像文件: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码文件: {mask_path}")

        mask = (mask > 0).astype(np.uint8)  # 二值化

        if self.transform:
            try:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            except Exception as e:
                raise RuntimeError(f"数据增强出错 (文件: {img_path.name}): {str(e)}")

        return img, mask.long()

# 2. 数据增强定义
def get_train_transform():
    return A.Compose([
        A.Resize(320, 800),  # Fast-SCNN推荐尺寸
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(320, 800),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# 3. 数据集划分
def prepare_datasets(data_root="data/kitti-road/data_road/training", val_ratio=0.2):
    full_dataset = KittiRoadDataset(
        img_dir=f"{data_root}/image_2",
        mask_dir=f"{data_root}/masks",
        transform=None
    )

    # 计算划分大小
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    # 随机划分
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )

    # 为子集添加不同的变换
    train_set.dataset.transform = get_train_transform()
    val_set.dataset.transform = get_val_transform()

    return train_set, val_set
