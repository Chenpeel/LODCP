import tensorflow as tf

def quantize_model(model_path, output_path, representative_dataset):
    # 加载原始模型
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    # 设置量化选项
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # 确保完全量化 (所有操作都量化为int8)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # 或 tf.int8
    converter.inference_output_type = tf.uint8  # 或 tf.int8

    # 转换模型
    quantized_model = converter.convert()

    # 保存量化后的模型
    with open(output_path, 'wb') as f:
        f.write(quantized_model)

    print(f"量化模型已保存到: {output_path}")

# 准备代表性数据集
def representative_dataset():
    # 使用你的数据准备代码
    from train.prepare.data_std import prepare_datasets
    train_set, _ = prepare_datasets()

    # 从训练集中抽取样本 (通常100-200个足够)
    for i in range(100):
        img, _ = train_set[i]
        img = img.numpy()  # 转换为numpy数组
        img = img[np.newaxis, ...]  # 添加batch维度
        yield [img.astype(np.float32)]  # 必须返回一个列表

# 量化float32模型
quantize_model(
    model_path="models/fastscnn/weights/saved_model/best_float32.tflite",
    output_path="models/fastscnn/weights/saved_model/best_int8.tflite",
    representative_dataset=representative_dataset
)

def validate_quantized_model(model_path):
    # 加载量化模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 获取输入输出详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n量化模型信息:")
    print(f"输入类型: {input_details[0]['dtype']}")
    print(f"输出类型: {output_details[0]['dtype']}")
    print(f"输入缩放参数: {input_details[0]['quantization']}")
    print(f"输出缩放参数: {output_details[0]['quantization']}")

# 验证量化模型
validate_quantized_model("models/fastscnn/weights/saved_model/best_int8.tflite")
