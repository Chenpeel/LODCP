import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 学习分支 (Learning to Downsample)
        self.lds = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # 全局特征提取
        self.global_feature = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 特征融合与分类
        self.classifier = nn.Sequential(
            nn.Conv2d(64 + 48, num_classes, 1)  # 跳跃连接
        )

    def forward(self, x):
        # 下采样
        lds_out = self.lds(x)  # [B, 48, H/4, W/4]

        # 全局特征
        global_feat = self.global_feature(lds_out)  # [B, 64, H/4, W/4]

        # 上采样 + 跳跃连接
        global_feat = F.interpolate(global_feat, scale_factor=4, mode='bilinear', align_corners=True)
        lds_out = F.interpolate(lds_out, scale_factor=4, mode='bilinear', align_corners=True)

        # 分类输出
        out = self.classifier(torch.cat([global_feat, lds_out], dim=1))
        return out
