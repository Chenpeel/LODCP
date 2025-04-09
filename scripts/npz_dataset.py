import numpy as np
from glob import glob
import tensorflow as tf
import albumentations as A
import os
from tqdm import tqdm

class NpzDataset:
    def __init__(self, npz_dir, batch_size=8, target_size=(256, 512), augment=False):
        self.npz_files = sorted(glob(os.path.join(npz_dir, "*.npz")))
        if not self.npz_files:
            raise ValueError(f"No .npz files found in directory: {npz_dir}")

        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(scale=0.1, rotate=10, p=0.3),
        ])
        self._count_samples()

    def _count_samples(self):
        self.total_samples = 0
        for npz_file in tqdm(self.npz_files, desc="Counting samples"):
            try:
                with np.load(npz_file) as data:
                    if 'images' not in data or 'masks' not in data:
                        print(f"⚠️ Warning: Missing 'images' or 'masks' in {npz_file}, skipping...")
                        continue
                    self.total_samples += len(data['images'])
            except Exception as e:
                print(f"⚠️ Error counting samples in {npz_file}: {str(e)}")
                continue

    def _validate_and_normalize_shapes(self, images, masks):
        """简化后的形状验证和标准化"""
        # 确保images是4维 (batch, height, width, channels)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1) if images.shape[-1] <= 4 else images[..., np.newaxis]

        # 确保masks是4维 (batch, height, width, 1)
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ..., np.newaxis]
        elif len(masks.shape) == 3:
            masks = masks[..., np.newaxis] if masks.shape[-1] != 1 else masks[np.newaxis, ...]

        return images, masks

    def _process_batch(self, images, masks):
        """处理单个批次数据"""
        try:
            # 标准化形状
            images, masks = self._validate_and_normalize_shapes(images, masks)

            # 确保图像有3个通道
            if images.shape[-1] == 1:  # 如果是灰度图，复制为3通道
                images = np.repeat(images, 3, axis=-1)
            elif images.shape[-1] == 4:  # 如果是RGBA，只取RGB
                images = images[..., :3]

            # 归一化
            images = images.astype('float32') / 255.0
            masks = masks.astype('float32') / 255.0

            # 数据增强
            if self.augment:
                augmented_images = []
                augmented_masks = []
                for i in range(len(images)):
                    augmented = self.augmentation(
                        image=images[i],
                        mask=masks[i].squeeze()  # 移除通道维度给albumentations
                    )
                    augmented_images.append(augmented['image'])
                    augmented_masks.append(augmented['mask'][..., np.newaxis])  # 重新添加通道维度

                images = np.array(augmented_images)
                masks = np.array(augmented_masks)

            # 调整尺寸
            images = tf.image.resize(images, self.target_size)
            masks = tf.image.resize(masks, self.target_size)

            # 确保masks是单通道
            if masks.shape[-1] != 1:
                masks = masks[..., :1]

            return images, masks[..., 0]  # 返回时移除masks的通道维度
        except Exception as e:
            print(f"⚠️ Error processing batch: {str(e)}")
            raise

    def _generator(self):
        for npz_file in tqdm(self.npz_files, desc="Processing files"):
            try:
                with np.load(npz_file) as data:
                    if 'images' not in data or 'masks' not in data:
                        print(f"⚠️ Missing data in {npz_file}, skipping...")
                        continue

                    images = data['images']
                    masks = data['masks']

                    # 分批处理
                    num_samples = len(images)
                    for i in range(0, num_samples, self.batch_size):
                        batch_images = images[i:i+self.batch_size]
                        batch_masks = masks[i:i+self.batch_size]

                        if len(batch_images) == 0:
                            continue

                        yield self._process_batch(batch_images, batch_masks)
            except Exception as e:
                print(f"⚠️ Error processing {npz_file}: {str(e)}")
                continue

    def to_tf_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(None, *self.target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *self.target_size), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=output_signature
        )

        return dataset.unbatch().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    @property
    def steps_per_epoch(self):
        return int(np.ceil(self.total_samples / self.batch_size))
