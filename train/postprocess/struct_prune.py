import torch
import torch_pruning as tp
from pathlib import Path
import sys

# 添加路径
PROJECT_ROOT = Path("~/Downloads/selfRepo/lodcp")
YOLOv5_ROOT = Path("~/Downloads/cloneRepo/yolov5")
sys.path.insert(0, str(YOLOv5_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from models.experimental import attempt_load
from models.yolo import Model, Detect

def prune_and_save(weights_path, prune_ratio=0.48):
    # 1. 加载完整模型结构
    model=attempt_load(weights_path)
    model.eval()

    # 2. 构建剪枝器
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=torch.randn(1,3,640,640),
        importance=tp.importance.MagnitudeImportance(p=2),
        ch_sparsity=prune_ratio,
        global_pruning=True,
        ignored_layers=ignored_layers,
    )

    # 3. 执行剪枝
    pruner.step()

    # 4. 生成剪枝后的YAML
    pruned_yaml = generate_pruned_yaml(model)
    print(pruned_yaml)
    # 5. 正确保存完整模型
    pruned_weights = weights_path.replace(".pt", "_pruned.pt")
    torch.save({
        'model': model.state_dict(),
        'yaml': pruned_yaml
    }, pruned_weights)

    return model

def generate_pruned_yaml(model):
    """自动生成剪枝后的YAML配置"""
    # 基础配置
    yaml_str = f"nc: {model.yaml['nc']}\n"
    yaml_str += f"depth_multiple: {model.yaml['depth_multiple']}\n"
    yaml_str += f"width_multiple: {model.yaml['width_multiple']}\n"

    # 处理anchors（可选，有些版本可能没有）
    if 'anchors' in model.yaml:
        yaml_str += "anchors:\n"
        yaml_str += f"  - {model.yaml['anchors']}\n\n"

    # 获取backbone和head的结构
    backbone = []
    head = []

    # 遍历模型层
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # 获取卷积层信息
            layer_info = f"  - [-1, 1, Conv, [{module.out_channels}, {module.kernel_size[0]}, {module.stride[0]}]]"

            # 简单区分backbone和head（根据层名或位置）
            if 'model.0.' in name or 'model.1.' in name or 'model.2.' in name or 'model.3.' in name:
                backbone.append(layer_info)
            else:
                head.append(layer_info)

    # 构建完整YAML
    yaml_str += "backbone:\n" + "\n".join(backbone) + "\n\n"
    yaml_str += "head:\n" + "\n".join(head) + "\n"

    return yaml_str

if __name__ == "__main__":
    pruned_model = prune_and_save("models/yolov5su/weights/best.pt")

    # 导出为ONNX验证
    torch.onnx.export(
        pruned_model,
        torch.randn(1,3,640,640),
        "pruned_model.onnx",
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=12
    )
