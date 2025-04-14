import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFastSCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 增强的下采样分支
        self.lds = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 通道数提升
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # 新增层
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 通道注意力模块
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid()
        )

        # 轻量级分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        # 下采样
        x = self.lds(x)  # [B, 128, H/4, W/4]

        # 注意力加权
        attn = self.attention(x)
        x = x * attn

        # 上采样
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        # 分类输出
        return self.classifier(x)
