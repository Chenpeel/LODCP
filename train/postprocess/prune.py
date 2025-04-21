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

def safe_prune(model, amount=0.3):
    """
    保护性剪枝函数
    不剪枝的关键层：
    - 前3层卷积（浅层特征提取）
    - Detect层（模型头部）
    - 特征金字塔关键层
    """
    # 不剪枝的关键层列表
    EXCLUDE_LAYERS = [
        'model.0', 'model.1', 'model.2',  # 前3层卷积
        'model.24',                       # Detect层
        'model.10', 'model.14', 'model.17' # 特征金字塔关键层
    ]

    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if not any(exclude in name for exclude in EXCLUDE_LAYERS):
                params_to_prune.append((module, 'weight'))

    print(f"\n将对 {len(params_to_prune)} 个卷积层执行剪枝（排除 {len(EXCLUDE_LAYERS)} 个关键层）")

    # 执行全局剪枝
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    # 使剪枝永久化并确保梯度可计算
    for module, _ in params_to_prune:
        prune.remove(module, 'weight')
        module.weight.requires_grad = True

    return model

def load_model_with_matching_keys(model, state_dict):
    """改进的权重加载函数，自动处理部分不兼容参数"""
    model_sd = model.state_dict()

    # 参数名映射（处理部分不兼容情况）
    param_mapping = {
        'model.24.cv2.0.0.conv.weight': 'model.24.m.0.weight',
        'model.24.cv2.1.0.conv.weight': 'model.24.m.1.weight',
        'model.24.cv2.2.0.conv.weight': 'model.24.m.2.weight'
    }

    matched, missing = 0, 0
    for k, v in model_sd.items():
        # 尝试原始键名
        if k in state_dict and state_dict[k].shape == v.shape:
            model_sd[k] = state_dict[k]
            matched += 1
        # 尝试映射键名
        elif any(src in k for src in param_mapping):
            mapped_key = next((src for src in param_mapping if src in k), None)
            if mapped_key and mapped_key in state_dict:
                model_sd[k] = state_dict[mapped_key]
                matched += 1
                print(f"参数映射: {mapped_key} -> {k}")
        else:
            missing += 1

    model.load_state_dict(model_sd, strict=False)
    print(f"\n权重加载: 成功匹配 {matched}/{len(model_sd)} 参数")
    if missing > 0:
        print(f"警告: {missing} 个参数未加载（使用初始化值）")

    return model

def verify_model(model, verbose=False):
    """全面验证模型状态"""
    print("\n" + "="*50)
    print("模型验证报告")
    print("="*50)

    # 1. 检查梯度状态
    requires_grad = sum(p.requires_grad for p in model.parameters())
    print(f"可训练参数: {requires_grad}/{sum(1 for _ in model.parameters())}")

    # 2. 检查Detect层完整性
    detect_layer = model.model[-1]
    print("\nDetect层状态:")
    print(f"Anchors: {detect_layer.anchors.shape if hasattr(detect_layer, 'anchors') else 'Missing'}")
    print(f"检测头数量: {len(detect_layer.m) if hasattr(detect_layer, 'm') else 0}")

    # 3. 详细稀疏度分析
    if verbose:
        print("\n各层稀疏度:")
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                sparsity = (param == 0).float().mean().item()
                print(f"{name:50} {sparsity:.1%}")

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
        print(f"模型版本: {ckpt.get('version', '未知')}")
        model = load_model_with_matching_keys(model, ckpt['model'].float().state_dict())
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 3. 剪枝前验证
    print("\n剪枝前验证:")
    verify_model(model, verbose=True)

    # 4. 执行保护性剪枝
    print("\n执行保护性剪枝...")
    model = safe_prune(model, amount=0.3)

    # 5. 剪枝后验证
    print("\n剪枝后验证:")
    verify_model(model, verbose=True)

    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum(count_zeros(p) for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"零值参数: {zero_params:,}")
    print(f"有效参数: {total_params - zero_params:,}")
    print(f"实际剪枝率: {zero_params/total_params:.1%}")

    # 6. 保存模型（兼容格式）
    output_path = PROJECT_ROOT / "pruned_model_protected.pt"
    torch.save({
        'model': model.state_dict(),
        'stride': int(model.stride.max()),
        'names': model.names,
        'anchors': model.model[-1].anchors if hasattr(model.model[-1], 'anchors') else None,
        'prune_info': {
            'method': 'protected_global',
            'amount': 0.3,
            'excluded_layers': ['model.0-2', 'model.10/14/17', 'model.24']
        }
    }, output_path)
    print(f"\n✅ 剪枝模型已保存: {output_path}")

    # 7. 转换为可部署格式
    deploy_path = PROJECT_ROOT / "pruned_model_compatible.pt"
    torch.save({
        'model': model,
        'stride': int(model.stride.max()),
        'names': model.names
    }, deploy_path)
    print(f"✅ 部署格式已保存: {deploy_path}")

if __name__ == "__main__":
    main()
