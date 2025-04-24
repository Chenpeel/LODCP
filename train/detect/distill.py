import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import yaml
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 添加路径
sys.path = [p for p in sys.path if "yolov5" not in str(p).lower()]
PROJECT_ROOT = Path("/Users/alpha/Downloads/selfRepo/lodcp")
YOLOv5_ROOT = Path("/Users/alpha/Downloads/cloneRepo/yolov5")
sys.path.insert(0, str(YOLOv5_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo import Model
from utils.dataloaders import LoadImagesAndLabels
from utils.general import check_dataset, colorstr
from utils.loss import ComputeLoss
from utils.torch_utils import de_parallel
from utils.callbacks import Callbacks
from val import run as val_run


class DistillLoss(nn.Module):
    def __init__(self, model, hyp, distill_weight=0.1):
        super().__init__()
        self.criterion = ComputeLoss(model)
        self.distill_weight = distill_weight
        self.hyp = hyp

    def forward(self, student_outputs, teacher_outputs, targets):
        loss, loss_items = self.criterion(student_outputs, targets)
        distill_loss = torch.zeros(1, device=student_outputs[0].device)

        # 处理教师输出格式
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]

        for s_pred, t_pred in zip(student_outputs, teacher_outputs):
            if isinstance(t_pred, (list, tuple)):
                t_pred = t_pred[0] if len(t_pred) > 0 else None

            if t_pred is None:
                continue

            # 处理YOLOv5/v7/v8的输出格式 [tensor, list]
            if t_pred.dim() == 3:  # [bs, 6, 8400]
                bs, _, num_preds = t_pred.shape
                split_sizes = [80*80*3, 40*40*3, 20*20*3]

                if sum(split_sizes) == num_preds:
                    start = 0
                    for i, size in enumerate(split_sizes):
                        end = start + size
                        t_scale = t_pred[:, :, start:end].permute(0, 2, 1)  # [bs, size, 6]
                        grid_size = int((size//3)**0.5)
                        t_scale = t_scale.view(bs, 3, grid_size, grid_size, 6)

                        # 找到对应的学生输出
                        s_scale = student_outputs[i]
                        self._compute_distill_loss(s_scale, t_scale, distill_loss)
                        start = end
                else:
                    # LOGGER.warning(f"Unexpected teacher output shape {t_pred.shape}")
                    pass

            elif t_pred.dim() == 5:  # 标准YOLO格式 [bs, na, h, w, C]
                self._compute_distill_loss(student_outputs[0], t_pred, distill_loss)
            else:
                pass
                # LOGGER.warning(f"Unsupported teacher output dimension {t_pred.dim()}")

        total_loss = loss + self.distill_weight * distill_loss
        return total_loss, torch.cat((loss_items, distill_loss))

    def _compute_distill_loss(self, s_pred, t_pred, distill_loss):
        """计算单个尺度上的蒸馏损失"""
        if s_pred.shape != t_pred.shape:
            t_pred = F.interpolate(
                t_pred.permute(0,1,4,2,3).flatten(0,1),
                size=s_pred.shape[2:4],
                mode='nearest'
            ).view(*s_pred.shape)

        # 分类损失
        s_cls = s_pred[..., 5:7].sigmoid()
        t_cls = t_pred[..., 5:7].sigmoid()
        temp = 3.0
        distill_loss += F.kl_div(
            F.log_softmax(s_cls / temp, dim=-1),
            F.softmax(t_cls / temp, dim=-1),
            reduction='batchmean'
        ) * (temp ** 2)

        # 边界框损失
        s_box = s_pred[..., :4].sigmoid()
        t_box = t_pred[..., :4].sigmoid()
        distill_loss += F.mse_loss(s_box, t_box)

def train(opt):
    # 初始化配置
    data_dict = check_dataset(opt.data)
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    # 设置默认超参数
    hyp.setdefault('gradient_clip_val', 10.0)
    hyp.setdefault('size', opt.imgsz)

    # 数据加载
    train_loader = DataLoader(
        LoadImagesAndLabels(
            data_dict['train'],
            img_size=opt.imgsz,
            augment=True,
            hyp=hyp
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
        pin_memory=True
    )

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载教师模型
    teacher = torch.load(opt.teacher_weights, map_location=device)['model'].float().eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 初始化学生模型
    student = Model(f'{YOLOv5_ROOT}/models/yolov5n.yaml', ch=3, nc=data_dict['nc']).to(device)
    student.hyp = hyp
    if opt.student_weights:
        ckpt = torch.load(opt.student_weights, map_location=device)
        student.load_state_dict(ckpt['model'].state_dict(), strict=False)

    # 训练配置
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=hyp['lr0'],
        momentum=hyp['momentum'],
        weight_decay=hyp['weight_decay']
    )

    scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # 初始周期长度(epoch)
            T_mult=1,  # 周期倍增因子
            eta_min=hyp['lr0'] * hyp['lrf'],  # 最小学习率
        )

    # 添加热身阶段
    warmup_epochs = hyp.get('warmup_epochs', 3)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs * len(train_loader)
    )

    # 组合调度器：先热身，然后余弦退火
    from torch.optim.lr_scheduler import SequentialLR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_epochs * len(train_loader)]
    )
    # 训练循环
    best_fitness = 0.0
    for epoch in range(opt.epochs):
        student.train()
        mloss = torch.zeros(4, device=device)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Epoch {epoch}/{opt.epochs}')

        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device, dtype=torch.float32) / 255.0
            targets = targets.to(device)

            # 前向传播
            with torch.no_grad():
                teacher_pred = teacher(imgs)
            student_pred = student(imgs)

            # 计算损失
            loss, loss_items = DistillLoss(student, hyp)(student_pred, teacher_pred, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), hyp['gradient_clip_val'])
            optimizer.step()
            scheduler.step()

            # 更新指标
            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_postfix_str(
                f'loss: {mloss.mean().item():.4f} '
                f'(box: {mloss[0].item():.4f}, cls: {mloss[1].item():.4f}, '
                f'dfl: {mloss[2].item():.4f}, distill: {mloss[3].item():.4f})'
            )

        # 验证和保存
        scheduler.step()
        ckpt = {
            'epoch': epoch,
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'hyp': hyp
        }

        # 定期验证
        if epoch % 10 == 0 or epoch == opt.epochs - 1:
            metrics = val_run(
                data=opt.data,
                weights=None,
                batch_size=opt.batch_size * 2,
                imgsz=opt.imgsz,
                model=student,
                plots=epoch == opt.epochs - 1,
                callbacks=callbacks
            )

            # 保存最佳模型
            fitness = metrics[2]  # mAP@0.5
            if fitness > best_fitness:
                best_fitness = fitness
                torch.save(ckpt, f'runs/train/{opt.name}/best.pt')

        # 保存最新模型
        torch.save(ckpt, f'runs/train/{opt.name}/last.pt')
        callbacks.run('on_train_epoch_end', epoch=epoch, model=student)

    # 最终验证
    val_run(
        data=opt.data,
        weights=f'runs/train/{opt.name}/best.pt',
        batch_size=opt.batch_size * 2,
        imgsz=opt.imgsz,
        plots=True,
        save_txt=True,
        save_conf=True,
        save_json=True
    )

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/bdd100k-yolo/bdd100k.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='train/detect/hyp.yml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--imgsz', '--img-size', type=int, default=640, help='train, val image size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--teacher-weights', type=str, required=True, help='teacher model weights path')
    parser.add_argument('--student-weights', type=str, default='', help='student initial weights path')
    parser.add_argument('--name', default='exp', help='save to project/name')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
