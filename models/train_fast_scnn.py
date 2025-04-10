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

def calculate_iou(pred, target, num_classes=2):
    """
    计算IoU指标（支持多类别）
    Args:
        pred: [B, H, W] 预测的类别标签
        target: [B, H, W] 真实标签
        num_classes: 类别数
    Returns:
        iou: 平均IoU
        class_iou: 各类别IoU
    """
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)  # 平滑处理
        ious.append(iou.item())
    return np.mean(ious), ious  # 返回mIoU和各类别IoU

def train_fast_scnn():
    # 初始化工作目录
    workdir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(workdir, "saved", f"fastscnn_{timestamp}")
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

    # 初始化记录器
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))
    csv_path = os.path.join(save_dir, "logs", "training_log.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'epoch', 'train_loss', 'val_loss',
            'mIoU', 'IoU_class0', 'IoU_class1',  # 记录各类别IoU
            'learning_rate', 'timestamp'
        ])

    # 准备数据
    print("准备数据集...")
    train_set, val_set = prepare_datasets()
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2)
    print(f"训练集: {len(train_set)} 样本, 验证集: {len(val_set)} 样本")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedFastSCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=500*len(train_loader))
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    best_mIoU = 0.0
    for epoch in range(500):
        model.train()
        train_loss = 0.0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/500 [训练]') as pbar:
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        total_mIoU = 0.0
        total_class_iou = [0.0, 0.0]  # 记录各类别IoU

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='[验证]'):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

                # 计算IoU
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

        # 记录到TensorBoard
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        writer.add_scalar('mIoU', avg_mIoU, epoch)
        writer.add_scalars('IoU_by_class', {
            'class_0': avg_class_iou[0],
            'class_1': avg_class_iou[1]
        }, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        # 记录到CSV
        with open(csv_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                epoch+1,
                avg_train_loss,
                avg_val_loss,
                avg_mIoU,
                avg_class_iou[0],  # 类别0 IoU
                avg_class_iou[1],  # 类别1 IoU
                current_lr,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        print(f"Epoch {epoch+1}/500 - Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"mIoU: {avg_mIoU:.4f} (Road: {avg_class_iou[0]:.4f}, Lane: {avg_class_iou[1]:.4f}) | "
              f"LR: {current_lr:.2e}")

        # 保存最佳模型
        if avg_mIoU > best_mIoU:
            best_mIoU = avg_mIoU
            torch.save(model.state_dict(),
                      os.path.join(save_dir, "models", f"best_model_mIoU{avg_mIoU:.4f}.pth"))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, "models", "final_model.pth"))
    writer.close()

if __name__ == "__main__":
    # 提升系统限制
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
    except:
        pass

    train_fast_scnn()
