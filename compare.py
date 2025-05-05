import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取三个CSV文件
standalone_df = pd.read_csv('models/v5nu/results.csv')
distilled_df = pd.read_csv('models/v5su_t_nu/results.csv')
teacher_df = pd.read_csv('models/v5su/results.csv')

teacher_color = 'blue'
distilled_color = 'red'
standalone_color = 'green'



# 设置绘图样式
plt.style.use('ggplot')
sns.set_palette("colorblind")
plt.rcParams.update({'font.size': 14})

# 创建一个图形，包含多个子图
fig = plt.figure(figsize=(20, 24))

# 1. mAP50-95 对比
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(teacher_df['epoch'], teacher_df['metrics/mAP50-95(B)'], label='Teacher',color=teacher_color)
ax1.plot(distilled_df['epoch'], distilled_df['metrics/mAP50-95(B)'], label='Distilled',color=distilled_color)
ax1.plot(standalone_df['epoch'], standalone_df['metrics/mAP50-95(B)'], label='Standalone',color=standalone_color)
ax1.set_title('mAP50-95 Comparison')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('mAP50-95')
ax1.legend()
ax1.grid(True)

# 2. Precision 对比
ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(teacher_df['epoch'], teacher_df['metrics/precision(B)'], label='Teacher',color=teacher_color)
ax2.plot(distilled_df['epoch'], distilled_df['metrics/precision(B)'], label='Distilled',color=distilled_color)
ax2.plot(standalone_df['epoch'], standalone_df['metrics/precision(B)'], label='Standalone',color=standalone_color)
ax2.set_title('Precision Comparison')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Precision')
ax2.legend()
ax2.grid(True)

# 3. Recall 对比
ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(teacher_df['epoch'], teacher_df['metrics/recall(B)'], label='Teacher',color=teacher_color)
ax3.plot(distilled_df['epoch'], distilled_df['metrics/recall(B)'], label='Distilled',color=distilled_color)
ax3.plot(standalone_df['epoch'], standalone_df['metrics/recall(B)'], label='Standalone',color=standalone_color)
ax3.set_title('Recall Comparison')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Recall')
ax3.legend()
ax3.grid(True)

# 4. Box Loss 对比
ax4 = fig.add_subplot(3, 2, 4)
ax4.plot(teacher_df['epoch'], teacher_df['val/box_loss'], label='Teacher',color=teacher_color)
ax4.plot(distilled_df['epoch'], distilled_df['val/box_loss'], label='Distilled',color=distilled_color)
ax4.plot(standalone_df['epoch'], standalone_df['val/box_loss'], label='Standalone',color=standalone_color)
ax4.set_title('Validation Box Loss Comparison')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Box Loss')
ax4.legend()
ax4.grid(True)

# 5. Class Loss 对比
ax5 = fig.add_subplot(3, 2, 5)
ax5.plot(teacher_df['epoch'], teacher_df['val/cls_loss'], label='Teacher',color=teacher_color)
ax5.plot(distilled_df['epoch'], distilled_df['val/cls_loss'], label='Distilled',color=distilled_color)
ax5.plot(standalone_df['epoch'], standalone_df['val/cls_loss'], label='Standalone',color=standalone_color)
ax5.set_title('Validation Class Loss Comparison')
ax5.set_xlabel('Epochs')
ax5.set_ylabel('Class Loss')
ax5.legend()
ax5.grid(True)

# 6. DFL Loss 对比
ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(teacher_df['epoch'], teacher_df['val/dfl_loss'], label='Teacher',color=teacher_color)
ax6.plot(distilled_df['epoch'], distilled_df['val/dfl_loss'], label='Distilled',color=distilled_color)
ax6.plot(standalone_df['epoch'], standalone_df['val/dfl_loss'], label='Standalone',color=standalone_color)
ax6.set_title('Validation DFL Loss Comparison')
ax6.set_xlabel('Epochs')
ax6.set_ylabel('DFL Loss')
ax6.legend()
ax6.grid(True)

plt.tight_layout()

# 添加一个额外的图表，显示最终性能指标的对比
fig2 = plt.figure(figsize=(14, 10))

# 确保所有模型使用最后一个可用的纪元数据
teacher_last = teacher_df.iloc[-1]
distilled_last = distilled_df.iloc[-1]
standalone_last = standalone_df.iloc[-1]

metrics = ['metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']
models = ['Teacher', 'Distilled', 'Standalone']
data = []

for metric in metrics:
    data.append([teacher_last[metric], distilled_last[metric], standalone_last[metric]])

data_array = np.array(data)

ax = fig2.add_subplot(111)
bar_width = 0.25
index = np.arange(len(metrics))

bar1 = ax.bar(index, data_array[:, 0], bar_width, label='Teacher',color=teacher_color)
bar2 = ax.bar(index + bar_width, data_array[:, 1], bar_width, label='Distilled',color=distilled_color)
bar3 = ax.bar(index + 2 * bar_width, data_array[:, 2], bar_width, label='Standalone',color=standalone_color)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Final Performance Comparison')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(['mAP50-95', 'Precision', 'Recall'])
ax.legend()

# 在柱状图上显示具体数值
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

plt.tight_layout()

# 绘制时间效率对比
fig3 = plt.figure(figsize=(14, 6))
ax = fig3.add_subplot(111)

# 计算每个epoch的平均时间
teacher_time_per_epoch = teacher_df['time'].diff().mean()
distilled_time_per_epoch = distilled_df['time'].diff().mean()
standalone_time_per_epoch = standalone_df['time'].diff().mean()

times = [teacher_time_per_epoch, distilled_time_per_epoch, standalone_time_per_epoch]
models = ['Teacher (YOLOv5su)', 'Distilled', 'Standalone (YOLOv5nu)']

ax.bar(models, times, color=[teacher_color,distilled_color,standalone_color])
ax.set_title('Average Time Per Epoch')
ax.set_xlabel('Model')
ax.set_ylabel('Time (seconds)')

# 在柱状图上显示具体数值
for i, v in enumerate(times):
    ax.text(i, v + 1, f'{v:.2f}s', ha='center')

plt.tight_layout()

# 显示所有图表
plt.show()

# 准备一个简单的性能总结
print("=== Performance Summary ===")
print(f"{'Model':<15} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 45)
print(f"{'Teacher':<15} {teacher_last['metrics/mAP50-95(B)']:<10.4f} {teacher_last['metrics/precision(B)']:<10.4f} {teacher_last['metrics/recall(B)']:<10.4f}")
print(f"{'Distilled':<15} {distilled_last['metrics/mAP50-95(B)']:<10.4f} {distilled_last['metrics/precision(B)']:<10.4f} {distilled_last['metrics/recall(B)']:<10.4f}")
print(f"{'Standalone':<15} {standalone_last['metrics/mAP50-95(B)']:<10.4f} {standalone_last['metrics/precision(B)']:<10.4f} {standalone_last['metrics/recall(B)']:<10.4f}")

# 计算模型收敛速度
print("\n=== Convergence Analysis ===")
target_map = 0.30  # 设置一个目标mAP值

for name, df in [('Teacher', teacher_df), ('Distilled', distilled_df), ('Standalone', standalone_df)]:
    map_values = df['metrics/mAP50-95(B)'].values
    epochs_to_converge = np.argmax(map_values >= target_map)
    if epochs_to_converge > 0:
        print(f"{name} reached mAP of {target_map:.2f} at epoch {epochs_to_converge}")
    else:
        print(f"{name} did not reach target mAP of {target_map:.2f}")

    # 打印最终达到的最大mAP值和所在epoch
    max_map = map_values.max()
    max_map_epoch = map_values.argmax() + 1  # epoch从1开始计数
    print(f"{name} reached maximum mAP of {max_map:.4f} at epoch {max_map_epoch}")
