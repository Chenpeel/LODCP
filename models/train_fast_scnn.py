import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from data_process.data_std import *
from models.fast_scnn import EnhancedFastSCNN
import torch.nn as nn
import os
import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import gc
import csv
from torch.optim.lr_scheduler import OneCycleLR

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# 确保打印所有GPU信息
print("="*50)
print("初始化训练脚本 - 检查GPU配置")
print("="*50)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"检测到GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print("="*50)

def calculate_iou(pred, target, num_classes=2):
    """计算IoU指标"""
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou.item())
    return np.mean(ious), ious

def train_fast_scnn():
    # 内存优化设置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f" {num_gpus} 个GPU")

    # 初始化工作目录
    workdir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(workdir, "saved", f"fastscnn_{timestamp}")
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

    # 准备数据
    print("\n准备数据集...")
    train_set, val_set = prepare_datasets()

    # 调整batch size以适应内存限制
    base_batch_size = 16  # 进一步减少batch size
    batch_size = base_batch_size * max(num_gpus, 1)
    num_workers = min(4, os.cpu_count() // max(num_gpus, 1))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(4, base_batch_size // 4),  # 验证集使用更小的batch size
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"训练集: {len(train_set)} 样本, 验证集: {len(val_set)} 样本")
    print(f"使用批次大小: {batch_size} (每个GPU: {base_batch_size})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedFastSCNN(num_classes=2)

    # 使用DataParallel包装模型
    if num_gpus > 1:
        print(f"\n正在使用DataParallel在 {num_gpus} 个GPU上训练")
        model = nn.DataParallel(model)
    model = model.to(device)

    # 打印模型设备分配
    print("\n模型设备分配:")
    print(f"模型主设备: {next(model.parameters()).device}")
    if num_gpus > 1:
        for i, (name, param) in enumerate(model.named_parameters()):
            if i == 0:
                print(f"第一个参数 '{name}' 所在设备: {param.device}")
                break

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=1000 * len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )

    # 修正GradScaler初始化
    scaler = torch.cuda.amp.GradScaler()  # 正确的初始化方式

    # 创建CSV文件并写入表头
    csv_file = os.path.join(save_dir, "training_log.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'mIoU', 'Road IoU', 'Lane IoU', 'Learning Rate'])

    # 训练循环
    best_mIoU = 0.0
    for epoch in range(1000):
        model.train()
        train_loss = 0.0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/1000 [训练]') as pbar:
            for images, masks in pbar:
                # 更高效的内存管理
                torch.cuda.empty_cache()

                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # 修正autocast使用
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()

                # 添加GPU内存使用信息
                gpu_mem = []
                for i in range(num_gpus):
                    allocated = torch.cuda.memory_allocated(i)/1024**3
                    gpu_mem.append(f"GPU{i}: {allocated:.2f}GB")
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'gpu': ', '.join(gpu_mem)
                })

        # 验证阶段
        model.eval()
        val_loss = 0.0
        total_mIoU = 0.0
        total_class_iou = [0.0, 0.0]

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='[验证]'):
                torch.cuda.empty_cache()
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()

                preds = torch.argmax(outputs, dim=1)
                batch_mIoU, batch_class_iou = calculate_iou(preds, masks)
                total_mIoU += batch_mIoU
                total_class_iou = [sum(x) for x in zip(total_class_iou, batch_class_iou)]

        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_mIoU = total_mIoU / len(val_loader)
        avg_class_iou = [iou / len(val_loader) for iou in total_class_iou]
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch+1}/1000 - Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"mIoU: {avg_mIoU:.4f} (Road: {avg_class_iou[0]:.4f}, Lane: {avg_class_iou[1]:.4f}) | "
              f"LR: {current_lr:.2e}")

        # 保存训练日志到CSV文件
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_mIoU, avg_class_iou[0], avg_class_iou[1], current_lr])

        # 保存最佳模型
        if avg_mIoU > best_mIoU:
            best_mIoU = avg_mIoU
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mIoU': best_mIoU,
            }, os.path.join(save_dir, "models", f"best_model_mIoU{avg_mIoU:.4f}.pth"))

    # 保存最终模型
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_mIoU': avg_mIoU,
    }, os.path.join(save_dir, "models", "final_model.pth"))

if __name__ == "__main__":
    # 提升系统限制
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
    except:
        pass

    train_fast_scnn()
