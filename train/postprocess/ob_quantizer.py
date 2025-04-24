import tensorflow as tf
import numpy as np

# 你的模型路径
saved_model_dir = "/Users/alpha/Downloads/selfRepo/lodcp/models/structpruned/weights/best_saved_model"

# 校准数据路径
calibration_data_path = "/Users/alpha/Downloads/selfRepo/lodcp/data/calibration_samples.npy"

# 载入校准数据
calibration_data = np.load(calibration_data_path)

# 定义代表性数据生成器
def representative_dataset():
    for image in calibration_data:
        yield [np.expand_dims(image, axis=0).astype(np.float32)]

# 创建转换器
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# 设置优化策略（启用量化）
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 指定代表性数据集
converter.representative_dataset = representative_dataset

# 设置目标类型（INT8）
converter.inference_input_type = tf.uint8   # 输入数据类型
converter.inference_output_type = tf.uint8  # 输出数据类型

# 导出量化模型
tflite_model = converter.convert()

# 保存模型
output_path = "/Users/alpha/Downloads/selfRepo/lodcp/models/structpruned/weights/best_full_int.tflite"
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"INT8量化成功，模型已保存：{output_path}")
