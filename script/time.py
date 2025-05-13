import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据
data = {
    'Model': ['v5su', 'v5su_t_nu', 'v5nu'],
    'Preprocess (ms)': [0.240917, 0.242234, 0.235917],
    'Inference (ms)': [4.802626, 2.796444, 2.742226],
    'Postprocess (ms)': [1.365100, 1.629904, 1.670403],
    'Total Speed (ms)': [6.409195, 4.669180, 4.650240],
    'FPS': [156.025835, 214.170383, 215.042653]
}
df = pd.DataFrame(data)

# 设置图表风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# 创建分组条形图
x = np.arange(len(df['Model']))
width = 0.2

# 绘制各个处理阶段的时间
plt.bar(x - width, df['Preprocess (ms)'], width, label='Preprocess')
plt.bar(x, df['Inference (ms)'], width, label='Inference')
plt.bar(x + width, df['Postprocess (ms)'], width, label='Postprocess')

# 添加标签和图例
plt.xlabel('Model')
plt.ylabel('Time (ms)')
plt.title('Processing Time Comparison')
plt.xticks(x, df['Model'])
plt.legend()

# 添加数值标签
for i, v in enumerate(df['Preprocess (ms)']):
    plt.text(i - width, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(df['Inference (ms)']):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(df['Postprocess (ms)']):
    plt.text(i + width, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('processing_time_comparison.pdf', dpi=300)
plt.show()

# 创建FPS对比图
plt.figure(figsize=(8, 5))
bars = plt.bar(df['Model'], df['FPS'], color='skyblue', width=0.5)

# 添加标签和标题
plt.xlabel('Model')
plt.ylabel('Frames Per Second (FPS)')
plt.title('Frame Rate Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('fps_comparison.pdf', dpi=300)
plt.show()
