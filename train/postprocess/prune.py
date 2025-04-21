import os
import sys
import torch
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

def analyze_mismatch(model, state_dict):
    """分析不匹配的参数"""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    print("\n详细参数分析:")
    print(f"仅在模型中的参数({len(model_keys - ckpt_keys)}): {sorted(model_keys - ckpt_keys)[:3] + ['...']}")
    print(f"仅在检查点中的参数({len(ckpt_keys - model_keys)}): {sorted(ckpt_keys - model_keys)[:3] + ['...']}")

def load_model_with_matching_keys(model, state_dict):
    """只加载匹配的权重参数"""
    model_state_dict = model.state_dict()
    matched_keys = []
    shape_mismatch = []

    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                model_state_dict[k] = v
                matched_keys.append(k)
            else:
                shape_mismatch.append((k, v.shape, model_state_dict[k].shape))

    analyze_mismatch(model, state_dict)

    if shape_mismatch:
        print("\n形状不匹配的参数:")
        for k, ckpt_shape, model_shape in shape_mismatch[:3]:
            print(f"{k}: 检查点{ckpt_shape} ≠ 模型{model_shape}")
        if len(shape_mismatch) > 3:
            print(f"...还有{len(shape_mismatch)-3}个")

    model.load_state_dict(model_state_dict, strict=False)
    print(f"\n权重加载结果:")
    print(f"✅ 成功加载 {len(matched_keys)}/{len(model_state_dict)} 参数")
    return model

def global_prune_model(model, amount=0.3):
    """执行全局剪枝"""
    parameters_to_prune = [
        (module, 'weight')
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]

    print(f"\n正在对 {len(parameters_to_prune)} 个卷积层执行全局剪枝 ({amount*100:.0f}%)...")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    return model

def verify_pruning(model, target_sparsity=0.3, tolerance=0.05):
    """验证剪枝效果"""
    total = sum(p.numel() for p in model.parameters())
    zeros = sum(count_zeros(p) for p in model.parameters())
    actual_sparsity = zeros / total
    print(f"\n剪枝验证: 目标{target_sparsity:.0%} 实际{actual_sparsity:.2%}")
    if abs(actual_sparsity - target_sparsity) >= tolerance:
        print(f"警告: 实际稀疏度与目标相差超过{tolerance:.0%}")

def main():
    # 初始化模型
    print("初始化模型...")
    model = Model(YOLOv5_ROOT / "models/yolov5s.yaml")
    print(f"原始模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print_sparsity(model)

    # 加载权重
    weights_path = PROJECT_ROOT / "models/yolov5su/weights/best.pt"
    print(f"\n正在加载权重: {weights_path}")

    try:
        ckpt = torch.load(weights_path, map_location='cpu')
        print(f"模型版本: {ckpt.get('version', '未知')}")
        print(f"训练使用的YOLOv5版本: {ckpt.get('yolov5_version', '未知')}")
        model = load_model_with_matching_keys(model, ckpt['model'].float().state_dict())
    except Exception as e:
        print(f"❌ 加载权重失败: {e}")
        return

    # 剪枝操作
    print("\n剪枝前状态:")
    print_sparsity(model)

    model = global_prune_model(model, amount=0.3)
    verify_pruning(model)

    print("\n剪枝后状态:")
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum(count_zeros(p) for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"零值参数: {zero_params:,}")
    print(f"非零参数: {total_params - zero_params:,}")

    # 保存模型
    output_path = PROJECT_ROOT / "pruned_model.pt"
    torch.save({
        'model': model.state_dict(),
        'prune_info': {
            'method': 'global_unstructured',
            'amount': 0.3,
            'zero_params': zero_params,
            'total_params': total_params,
            'sparsity': zero_params / total_params
        }
    }, output_path)
    print(f"\n✅ 模型已保存至 {output_path}")

if __name__ == "__main__":
    main()
