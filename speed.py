from ultralytics import YOLO
import pandas as pd

# 定义模型路径
model_paths = {
    "v5su": "models/v5su/weights/best.pt",
    "v5su_t_nu": "v5su_t_nu/weights/best.pt",
    "v5nu": "models/v5nu/weights/best.pt"
}

# 定义数据集路径
data_yaml = "/Users/alpha/Downloads/selfRepo/lodcp/data/bdd100k-yolo/bdd100k.yaml"

# 创建结果DataFrames
results_df = pd.DataFrame()

# 测试每个模型
for name, path in model_paths.items():
    print(f"Testing model: {name}")
    model = YOLO(path)

    # 使用val命令的profile选项并指定数据集
    results = model.val(data=data_yaml, profile=True)

    # 打印可用的键，帮助调试
    print(f"Available speed keys: {results.speed.keys()}")

    # 提取速度信息 - 使用更通用的方式获取性能数据
    speed_info = {
        "Model": name,
        "Preprocess (ms)": results.speed.get('preprocess', 0),
        "Inference (ms)": results.speed.get('inference', 0),
        "Postprocess (ms)": results.speed.get('postprocess', 0),
        "Total Speed (ms)": sum(results.speed.values()),
        "FPS": 1000 / sum(results.speed.values()) if sum(results.speed.values()) > 0 else 0
    }

    # 添加到结果DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([speed_info])], ignore_index=True)

# 显示结果
print("\n=== Performance Comparison ===")
print(results_df)

# 保存结果
results_df.to_csv("model_profile_comparison.csv", index=False)
print("Results saved to model_profile_comparison.csv")
