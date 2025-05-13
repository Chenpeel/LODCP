import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms


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
            nn.ReLU(),
        )

        # 通道注意力模块
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 128, 1),
            nn.Sigmoid(),
        )

        # 轻量级分类头
        self.classifier = nn.Sequential(nn.Conv2d(128, num_classes, 1))

    def forward(self, x):
        # 下采样
        x = self.lds(x)  # [B, 128, H/4, W/4]

        # 注意力加权
        attn = self.attention(x)
        x = x * attn

        # 上采样
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)

        # 分类输出
        return self.classifier(x)


class FastSCNN:
    def __init__(self, model_path, num_classes=2):
        self.model = EnhancedFastSCNN(num_classes)
        checkpoint = torch.load(model_path, map_location="cpu")
        # 兼容只保存了state_dict和保存了完整checkpoint的情况
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def predict(self, frame):
        inp = self.transform(frame).unsqueeze(0)
        with torch.no_grad():
            out = self.model(inp)[0]
        mask = out.argmax(0).cpu().numpy()
        mask = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return mask
