import numpy as np
from glob import glob
import tensorflow as tf

class NpzDataset:
    def __init__(self, npz_dir, batch_size=8, target_size=(256, 512)):
        """
        Args:
            npz_dir: 包含npz文件的目录
            batch_size: 批大小
            target_size: 目标尺寸 (height, width)
        """
        self.npz_files = sorted(glob(f"{npz_dir}/*.npz"))
        self.batch_size = batch_size
        self.target_size = target_size

        # 预计算数据集大小
        self._count_samples()

    def _count_samples(self):
        """预计算总样本数"""
        self.total_samples = 0
        for npz_file in self.npz_files:
            with np.load(npz_file) as data:
                self.total_samples += len(data['images'])

    def __iter__(self):
        for npz_file in self.npz_files:
            data = np.load(npz_file)
            images = data['images'].astype('float32') / 255.0  # 归一化
            masks = data['masks'].astype('float32') / 255.0

            # 确保尺寸匹配
            if images.shape[1:3] != self.target_size:
                images = tf.image.resize(images, self.target_size)
                masks = tf.image.resize(masks[..., np.newaxis], self.target_size)[..., 0]

            # 如果模型需要单通道但输入是RGB
            if images.shape[-1] == 3:
                # 转换为灰度 (保持3通道兼容性，实际可按需修改)
                pass

            # 分批生成
            num_samples = len(images)
            for i in range(0, num_samples, self.batch_size):
                batch_images = images[i:i+self.batch_size]
                batch_masks = masks[i:i+self.batch_size]
                yield batch_images, batch_masks

    @property
    def steps_per_epoch(self):
        """计算每个epoch的步数"""
        return int(np.ceil(self.total_samples / self.batch_size))
