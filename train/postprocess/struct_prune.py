import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from copy import deepcopy
import numpy as np
import yaml
import json

PROJECT_ROOT = Path("/Users/alpha/Downloads/selfRepo/lodcp")
YOLOv5_ROOT = Path("/Users/alpha/Downloads/cloneRepo/yolov5")
sys.path.insert(0, str(YOLOv5_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from models.yolo import Model
from models.common import Conv, C3

model_path = f"{PROJECT_ROOT}/models/yolov5su/weights/best.pt"

def load_yolov5s_model(weights_path=None):
    """加载YOLOv5s模型"""
    # 加载模型配置但不传递anchors参数
    model = Model(cfg="models/yolov5s.yaml", ch=3, nc=2)  # 从配置文件创建模型

    # 如果提供了权重路径，加载权重
    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu")  # 加载检查点
        csd = ckpt["model"].float().state_dict()  # 检查点状态字典
        model.load_state_dict(csd, strict=False)  # 加载

    # 手动设置anchors
    anchors = [
        [10, 13, 16, 30, 33, 23],  # P3/8
        [30, 61, 62, 45, 59, 119],  # P4/16
        [116, 90, 156, 198, 373, 326]  # P5/32
    ]
    model.yaml["anchors"] = anchors  # 直接设置yaml中的anchors
    model.nc = 2  # 类别数

    return model

def get_channels_to_prune(module, amount=0.3):
    """计算需要剪枝的通道索引"""
    if isinstance(module, (nn.Conv2d, Conv)):
        # 对于标准Conv2d或YOLOv5的Conv模块，计算卷积核的L2范数
        conv = module.conv if isinstance(module, Conv) else module
        norms = torch.norm(conv.weight.data, p=2, dim=(1, 2, 3))
    elif isinstance(module, nn.Linear):
        # 计算每个神经元的L2范数
        norms = torch.norm(module.weight.data, p=2, dim=1)
    else:
        return []

    # 按范数排序，选择最小的amount比例
    sorted_indices = torch.argsort(norms)
    num_to_prune = int(amount * len(norms))
    return sorted_indices[:num_to_prune].tolist()

def apply_real_structured_pruning(model, amount=0.3):
    """实际执行结构化剪枝（真正减少参数）"""
    model = deepcopy(model)
    prune_info = {}

    # 第一遍：收集所有需要剪枝的层及其通道
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, Conv, nn.Linear)):
            channels_to_prune = get_channels_to_prune(module, amount)
            if channels_to_prune:
                prune_info[name] = {
                    'module': module,
                    'channels': channels_to_prune,
                    'type': type(module).__name__
                }

    # 第二遍：实际修改模型结构
    for name, info in prune_info.items():
        module = info['module']
        channels_to_prune = info['channels']

        if info['type'] == 'Conv':
            # 处理YOLOv5的自定义Conv模块
            conv = module.conv
            bn = module.bn
            act = module.act

            # 创建新的Conv模块，减少输出通道
            new_out_channels = conv.out_channels - len(channels_to_prune)
            new_conv = Conv(
                conv.in_channels,
                new_out_channels,
                k=conv.kernel_size[0],
                s=conv.stride[0],
                p=conv.padding[0],
                g=conv.groups,
                act=True  # YOLOv5的Conv总是有激活函数
            )

            # 复制保留的权重
            keep_indices = [i for i in range(conv.out_channels) if i not in channels_to_prune]
            new_conv.conv.weight.data = conv.weight.data[keep_indices]
            if conv.bias is not None:
                new_conv.conv.bias.data = conv.bias.data[keep_indices]

            # 复制BN层参数
            new_conv.bn.weight.data = bn.weight.data[keep_indices]
            new_conv.bn.bias.data = bn.bias.data[keep_indices]
            new_conv.bn.running_mean.data = bn.running_mean.data[keep_indices]
            new_conv.bn.running_var.data = bn.running_var.data[keep_indices]

            # 替换原模块
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_conv)

        elif info['type'] == 'Conv2d':
            # 处理标准Conv2d模块
            new_out_channels = module.out_channels - len(channels_to_prune)
            new_conv = nn.Conv2d(
                module.in_channels,
                new_out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None
            )

            # 复制保留的权重
            keep_indices = [i for i in range(module.out_channels) if i not in channels_to_prune]
            new_conv.weight.data = module.weight.data[keep_indices]
            if module.bias is not None:
                new_conv.bias.data = module.bias.data[keep_indices]

            # 替换原模块
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_conv)

    return model

def test_model_size(original_model, pruned_model):
    """测试模型大小变化"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)

    print(f"原始模型参数数量: {original_params}")
    print(f"剪枝后模型参数数量: {pruned_params}")
    print(f"参数减少比例: {(original_params - pruned_params) / original_params * 100:.2f}%")

def check_channel_alignment(pruned_model):
    from models.yolo import Detect
    mismatch_log = []

    for name, module in pruned_model.named_modules():
        if isinstance(module, Detect):
            continue  # 跳过检测头

        if isinstance(module, nn.Conv2d):
            next_conv = None
            # 查找下一个卷积层
            for _, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    next_conv = child
                    break

            if next_conv and module.out_channels != next_conv.in_channels:
                mismatch_log.append(
                    f"通道不匹配: {name} (out={module.out_channels}) -> "
                    f"下一层 (in={next_conv.in_channels})"
                )

    if mismatch_log:
        print("⚠️ 发现通道不匹配:")
        for log in mismatch_log:
            print(log)
    else:
        print("✅ 所有卷积层通道对齐正常")

def check_residual_connections(pruned_model):
    from models.common import C3
    for name, module in pruned_model.named_modules():
        if isinstance(module, C3):
            main_branch = module.cv2.conv.out_channels
            shortcut = module.cv1.conv.out_channels
            if main_branch != shortcut:
                print(f"⚠️ 残差通道不匹配: {name} (主分支={main_branch}, 捷径分支={shortcut})")
            else:
                print(f"✅ {name} 残差通道正常: {main_branch}")

def main():
    # 1. 加载预训练的YOLOv5s模型
    model = load_yolov5s_model(model_path)
    print("原始模型加载完成")

    # 2. 应用结构化剪枝
    pruned_model = apply_real_structured_pruning(model, amount=0.48)
    print("结构化剪枝完成")
    check_channel_alignment(pruned_model)
    check_residual_connections(pruned_model)

    # 3. 测试模型大小变化
    test_model_size(model, pruned_model)

    # 4. 保存剪枝后的模型
    torch.save({
        'model': pruned_model.state_dict(),
        'ema': None,
        'updates': 0,
        'optimizer': None,
        'epoch': -1,
        'anchors': [  # 保存anchors信息
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
    }, 'pruned_model.pt')
    print("剪枝后模型已保存")


if __name__ == "__main__":
    main()
