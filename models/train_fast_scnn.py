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
import gc

def train_fast_scnn():
    # 准备工作目录
    workdir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(workdir, "logs", f"fastscnn_{timestamp}")
    models_dir = os.path.join(workdir, "saved_models")  # 修改为更清晰的目录名
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 准备数据 - 减少num_workers并添加persistent_workers
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

    # 训练循环 - 添加资源清理
    for epoch in range(50):
        model.train()
        train_loss = 0.0

        # 添加资源清理和错误处理
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

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/50 - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 定期清理资源
        if (epoch + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 保存模型
    model_path = os.path.join(models_dir, "fast-scnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"训练完成，模型保存到 {model_path}")

if __name__ == "__main__":
    # 设置文件描述符限制 (Mac/Linux)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
    except:
        pass

    train_fast_scnn()
