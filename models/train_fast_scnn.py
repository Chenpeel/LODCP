import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_process.data_std import *
from models.fast_scnn import *
import torch.nn as nn
import os
import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import gc
import csv

def train_fast_scnn():
    # 准备工作目录
    workdir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(workdir, "saved", f"fastscnn_{timestamp}")  # 统一保存目录

    # 创建所有子目录
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)    # 模型保存目录
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)      # 日志保存目录

    # 初始化TensorBoard和CSV记录器
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))
    csv_path = os.path.join(save_dir, "logs", "training_log.csv")

    # 初始化CSV文件并写入表头
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'epoch',
            'train_loss',
            'val_loss',
            'learning_rate',
            'timestamp'
        ])

    # 准备数据
    print("准备数据集...")
    train_set, val_set = prepare_datasets()
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
    print(f"训练集: {len(train_set)} 样本, 验证集: {len(val_set)} 样本")

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = FastSCNN(num_classes=2).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 初始化最佳验证损失
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        # 训练阶段
        try:
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/50 [训练]') as train_pbar:
                for images, masks in train_pbar:
                    images, masks = images.to(device), masks.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        except Exception as e:
            print(f"训练出错: {str(e)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # 验证阶段
        model.eval()
        val_loss = 0.0
        try:
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/50 [验证]') as val_pbar:
                with torch.no_grad():
                    for images, masks in val_pbar:
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item()
                        val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        except Exception as e:
            print(f"验证出错: {str(e)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        # 记录到CSV
        with open(csv_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                epoch+1,
                avg_train_loss,
                avg_val_loss,
                current_lr,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        print(f"Epoch {epoch+1}/50 - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, "models", f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"新的最佳模型保存到 {best_model_path}")

        # 定期清理资源
        if (epoch + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 训练结束后保存最终模型
    final_model_path = os.path.join(save_dir, "models", "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"训练完成，最终模型保存到 {final_model_path}")

    # 关闭TensorBoard writer
    writer.close()

if __name__ == "__main__":
    # 设置文件描述符限制 (Mac/Linux)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
    except:
        pass

    train_fast_scnn()
