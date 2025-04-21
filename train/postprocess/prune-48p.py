import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path

PROJECT_ROOT = Path("/Users/alpha/Downloads/selfRepo/lodcp")
YOLOv5_ROOT = Path("/Users/alpha/Downloads/cloneRepo/yolov5")

sys.path.insert(0, str(YOLOv5_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo import Model
from utils.torch_utils import prune as yolov5_prune

def count_zeros(tensor):
    """安全计算张量中的零值数量"""
    return torch.sum(tensor == 0).item()

def print_sparsity(model):
    """打印模型的稀疏度统计"""
    zero_params = total_params = 0
    for p in model.parameters():
        zero_params += count_zeros(p)
        total_params += p.numel()
    sparsity = zero_params / total_params if total_params > 0 else 0
    print(f"全局稀疏度: {sparsity:.2%} (零值参数 {zero_params}/{total_params})")

def aggressive_prune(model, target_sparsity=0.48):
    """
    - 浅层(前3层): 20%剪枝
    - 中间层: 50%剪枝
    - 特征金字塔层: 30%剪枝
    - 输出层: 20%剪枝
    """
    # 定义分层剪枝策略
    PRUNE_POLICY = {
        'model.0': 0.2,   # 浅层卷积
        'model.1': 0.2,
        'model.2': 0.2,
        'model.3': 0.5,   # 中间层
        'model.4': 0.5,
        'model.5': 0.5,
        'model.6': 0.5,
        'model.7': 0.5,
        'model.8': 0.5,
        'model.9': 0.5,
        'model.10': 0.3,  # 特征金字塔
        'model.13': 0.3,
        'model.14': 0.3,
        'model.17': 0.3,
        'model.18': 0.3,
        'model.20': 0.3,
        'model.21': 0.3,
        'model.23': 0.3,
        'model.24': 0.2   # 输出层
    }

    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 查找匹配的剪枝策略
            prune_amount = 0
            for layer_prefix, amount in PRUNE_POLICY.items():
                if name.startswith(layer_prefix):
                    prune_amount = amount
                    break

            if prune_amount > 0:
                params_to_prune.append((module, 'weight'))
                print(f"计划剪枝 {name}: 目标剪枝率 {prune_amount:.0%}")

    # 执行全局剪枝
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=target_sparsity
    )

    # 使剪枝永久化
    for module, _ in params_to_prune:
        prune.remove(module, 'weight')
        module.weight.requires_grad = True

    return model

def load_model_with_matching_keys(model, state_dict):
    """改进的权重加载函数"""
    model_sd = model.state_dict()
    matched, missing = 0, 0

    for k, v in model_sd.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            model_sd[k] = state_dict[k]
            matched += 1
        else:
            missing += 1

    model.load_state_dict(model_sd, strict=False)
    print(f"\n权重加载: 成功匹配 {matched}/{len(model_sd)} 参数")
    if missing > 0:
        print(f"警告: {missing} 个参数未加载（使用初始化值）")

    return model

def verify_model(model):
    """验证模型状态"""
    print("\n模型验证:")
    requires_grad = sum(p.requires_grad for p in model.parameters())
    print(f"可训练参数: {requires_grad}/{sum(1 for _ in model.parameters())}")

    # 检查Detect层
    detect_layer = model.model[-1]
    print(f"检测头数量: {len(detect_layer.m) if hasattr(detect_layer, 'm') else 0}")

def main():
    # 1. 初始化模型
    print("初始化模型...")
    model = Model(YOLOv5_ROOT / "models/yolov5s.yaml")
    print(f"原始模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print_sparsity(model)

    # 2. 加载权重
    weights_path = PROJECT_ROOT / "models/yolov5su/weights/best.pt"
    print(f"\n加载权重: {weights_path}")

    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        model = load_model_with_matching_keys(model, ckpt['model'].float().state_dict())
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 3. 执行激进的剪枝
    print("\n执行激进剪枝(目标48%)...")
    model = aggressive_prune(model, target_sparsity=0.48)

    # 4. 剪枝后验证
    verify_model(model)

    # 5. 计算实际剪枝率
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum(count_zeros(p) for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"零值参数: {zero_params:,}")
    print(f"有效参数: {total_params - zero_params:,}")
    actual_prune_rate = zero_params / total_params
    print(f"实际剪枝率: {actual_prune_rate:.1%}")

    # 如果未达到目标，进行补充剪枝
    if actual_prune_rate < 0.48:
        additional_prune = (0.48 - actual_prune_rate) / (1 - actual_prune_rate)
        print(f"\n未达到目标剪枝率，进行补充剪枝 {additional_prune:.1%}...")
        model = aggressive_prune(model, target_sparsity=additional_prune)

        # 重新计算
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum(count_zeros(p) for p in model.parameters())
        print(f"最终剪枝率: {zero_params/total_params:.1%}")

    # 6. 保存模型
    output_path = PROJECT_ROOT / "pruned_model_48percent.pt"
    torch.save({
        'model': model.state_dict(),
        'stride': int(model.stride.max()),
        'names': model.names,
        'prune_rate': zero_params / total_params
    }, output_path)
    print(f"\n✅ 剪枝模型已保存: {output_path}")
    # 7. 转换为可部署格式
    deploy_path = PROJECT_ROOT / "pruned_model_48percent_deploy.pt"
    torch.save({
        'model': model,
        'stride': int(model.stride.max()),
        'names': model.names
    }, deploy_path)
    print(f"✅ 部署格式已保存: {deploy_path}")
if __name__ == "__main__":
    main()
