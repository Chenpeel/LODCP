# DistillYOLO
# USE Ultralytics API

# This Example for YOLOv5su teach YOLOv5nu
import argparse
import os
import sys
import yaml
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("~/Downloads/selfRepo/lodcp")
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


class DistillYOLO:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # 加载配置
        with open(opt.hyp) as f:
            self.hyp = yaml.safe_load(f)

        # 设置保存路径
        self.save_dir = Path(f'runs/train/{opt.name}')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 加载教师模型
        print(f"Loading teacher model from {opt.teacher_weights}")
        self.teacher = YOLO(opt.teacher_weights)

        # 初始化学生模型
        if opt.student_weights:
            print(f"Loading student model from {opt.student_weights}")
            self.student = YOLO(opt.student_weights)
        else:
            print(f"Initializing new student model with YOLOv5n")
            self.student = YOLO('yolov5n.pt')

        # 设置蒸馏权重参数
        self.initial_distill_weight = 0.5  # 初始蒸馏权重
        self.final_distill_weight = 0.1    # 最终蒸馏权重
        self.current_epoch = 0             # 当前训练轮次
        self.max_epochs = opt.epochs       # 总训练轮次

    def get_distill_weight(self):
        """计算当前轮次的蒸馏权重（使用余弦退火策略）"""
        # 确保在有效范围内
        normalized_epoch = min(self.current_epoch, self.max_epochs - 1)

        # 应用余弦退火公式
        return self.final_distill_weight + 0.5 * (self.initial_distill_weight - self.final_distill_weight) * \
               (1 + math.cos(math.pi * normalized_epoch / self.max_epochs))

    def train(self):
        """使用YOLOv8 API训练并进行知识蒸馏"""
        # 创建带有知识蒸馏功能的学生模型
        student_model = self.create_distill_model()

        # 添加训练状态监控回调
        self.student.add_callback("on_train_epoch_end", self.epoch_callback)

        # 设置训练参数
        results = student_model.train(
            data=self.opt.data,
            epochs=self.opt.epochs,
            batch=self.opt.batch_size,
            imgsz=self.opt.imgsz,
            device=self.opt.device,
            project=str(self.save_dir.parent),
            name=self.opt.name,
            exist_ok=True,
            patience=100,  # 早停
            save=True,
            seed=0,
            cos_lr=self.opt.cos_lr,  # 使用余弦退火学习率
            save_dir=str(self.save_dir)  # 确保所有输出保存在正确目录
        )

        # 最终评估
        print("\nFinal evaluation of best model:")
        student_model.val()

        return results

    def epoch_callback(self, trainer):
        """每个训练轮次结束时更新计数器"""
        self.current_epoch += 1
        current_weight = self.get_distill_weight()
        print(f"Epoch {self.current_epoch}/{self.max_epochs} - Current distill weight: {current_weight:.4f}")

    def create_distill_model(self):
        """创建一个带有蒸馏功能的学生模型"""
        # 冻结教师模型
        teacher_model = self.teacher.model
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()

        # 获取学生模型
        student_model = self.student

        # 保存原始前向传播方法的引用
        original_forward = student_model.model.forward

        # 定义包含蒸馏的前向传播方法
        def forward_with_distillation(x, *args, **kwargs):
            # 在训练模式下进行蒸馏
            if student_model.model.training:
                # 学生模型的原始输出
                student_outputs = original_forward(x, *args, **kwargs)

                # 教师模型的输出 (不计算梯度)
                with torch.no_grad():
                    teacher_outputs = teacher_model(x, *args, **kwargs)

                # 获取当前的蒸馏权重
                current_distill_weight = self.get_distill_weight()

                # 如果在训练阶段，返回(student_outputs, teacher_outputs)
                # 否则仅返回student_outputs
                if isinstance(student_outputs, tuple) and len(student_outputs) > 0:
                    # 一些YOLO模型返回(pred, loss)格式
                    pred, loss = student_outputs

                    # 计算蒸馏损失
                    distill_loss = self.compute_distill_loss(student_outputs, teacher_outputs)

                    # 将蒸馏损失添加到原始损失中
                    if isinstance(loss, dict):
                        # 如果loss是字典，添加新的蒸馏损失项
                        loss['distill'] = distill_loss * current_distill_weight
                        # 更新总损失
                        if 'loss' in loss:
                            loss['loss'] += distill_loss * current_distill_weight
                    elif isinstance(loss, torch.Tensor):
                        # 如果loss是张量，直接加上蒸馏损失
                        loss += distill_loss * current_distill_weight

                    return pred, loss
                else:
                    # 某些情况下可能只返回预测值
                    return student_outputs
            else:
                # 在非训练模式下，使用原始前向传播
                return original_forward(x, *args, **kwargs)

        # 替换前向传播方法
        student_model.model.forward = forward_with_distillation
        print(f"Enabled knowledge distillation in student model (initial weight: {self.initial_distill_weight:.2f}, final weight: {self.final_distill_weight:.2f})")

        return student_model

    def compute_distill_loss(self, student_outputs, teacher_outputs):
        """计算知识蒸馏损失"""
        distill_loss = torch.zeros(1, device=self.device)

        # 提取预测结果
        if isinstance(student_outputs, tuple):
            student_preds = student_outputs[0]
        else:
            student_preds = student_outputs

        if isinstance(teacher_outputs, tuple):
            teacher_preds = teacher_outputs[0]
        else:
            teacher_preds = teacher_outputs

        # 处理YOLOv8的多尺度预测输出
        if isinstance(student_preds, list) and isinstance(teacher_preds, list):
            # 遍历每个检测层级
            for s_pred, t_pred in zip(student_preds, teacher_preds):
                # 确保形状匹配
                if s_pred.shape != t_pred.shape:
                    continue

                # 特征蒸馏 - 软目标蒸馏
                temp = 3.0  # 温度参数

                # 提取目标置信度
                s_conf = s_pred[..., 4].sigmoid()
                t_conf = t_pred[..., 4].sigmoid()
                # 计算KL散度损失
                conf_loss = F.kl_div(
                    F.log_softmax(s_conf / temp, dim=0),
                    F.softmax(t_conf / temp, dim=0),
                    reduction='batchmean',
                    log_target=False
                ) * (temp * temp)

                # 提取分类预测
                if s_pred.shape[-1] > 5:  # 如果有类别预测
                    s_cls = s_pred[..., 5:].sigmoid()
                    t_cls = t_pred[..., 5:].sigmoid()
                    # 计算KL散度损失
                    cls_loss = F.kl_div(
                        F.log_softmax(s_cls / temp, dim=-1),
                        F.softmax(t_cls / temp, dim=-1),
                        reduction='batchmean',
                        log_target=False
                    ) * (temp * temp)

                    distill_loss += cls_loss

                # 提取边界框预测
                s_box = s_pred[..., :4].sigmoid()
                t_box = t_pred[..., :4].sigmoid()
                box_loss = F.mse_loss(s_box, t_box)

                # 总蒸馏损失
                distill_loss += box_loss + conf_loss

        elif isinstance(student_preds, torch.Tensor) and isinstance(teacher_preds, torch.Tensor):
            # 单一输出情况
            s_pred = student_preds
            t_pred = teacher_preds

            # 应用相同的蒸馏方法
            temp = 3.0

            # 位置和置信度损失
            if min(s_pred.shape) > 0 and min(t_pred.shape) > 0:
                s_box = s_pred[..., :4]
                t_box = t_pred[..., :4]
                box_loss = F.mse_loss(s_box, t_box)

                distill_loss += box_loss

                # 类别损失 (如果存在)
                if s_pred.shape[-1] > 5:
                    s_cls = s_pred[..., 5:]
                    t_cls = t_pred[..., 5:]
                    cls_loss = F.kl_div(
                        F.log_softmax(s_cls / temp, dim=-1),
                        F.softmax(t_cls / temp, dim=-1),
                        reduction='batchmean',
                        log_target=False
                    ) * (temp * temp)

                    distill_loss += cls_loss

        return distill_loss


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
    parser.add_argument('--cos_lr', action='store_true', help='use cosine learning rate scheduler')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    distiller = DistillYOLO(opt)
    results = distiller.train()
    print(f"Training completed. Best results: {results}")
