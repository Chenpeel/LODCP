from ultralytics import YOLO
from pathlib import Path
import torch

# 指定路径
model_path = '/Users/alpha/Downloads/selfRepo/lodcp/models/v5su_t_nu/weights/best.pt'
image_paths = [
    '/Users/alpha/Downloads/selfRepo/lodcp/data/bdd100k-dataset/bdd100k/bdd100k/images/100k/test/cabc30fc-eb673c5a.jpg',
    '/Users/alpha/Downloads/selfRepo/lodcp/data/bdd100k-dataset/bdd100k/bdd100k/images/100k/test/cade2084-8fb02395.jpg'
]
output_dir = '/Users/alpha/Downloads/selfRepo/lodcp/docs/thesis/figures/chapter4'

# 创建输出目录
Path(output_dir).mkdir(parents=True, exist_ok=True)

try:
    # 加载模型
    model = YOLO(model_path)

    # 处理图像并保存结果
    for img_path in image_paths:
        try:
            # 获取图像名称
            img_name = Path(img_path).stem

            # 执行推理
            results = model(img_path, conf=0.25, iou=0.45)

            # 保存检测结果图像
            for r in results:
                im_path = Path(output_dir) / f"{img_name}.jpg"
                r.save(filename=str(im_path))

            print(f"已保存检测结果至: {output_dir}/{img_name}.jpg")

        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")

except Exception as e:
    print(f"加载模型时出错: {e}")

    # 如果您的本地环境有YOLOv5版本的代码库，尝试直接使用
    try:
        import sys
        sys.path.append('/Users/alpha/Downloads/selfRepo/lodcp')

        # 读取原始图像
        import cv2

        # 加载模型 (使用PyTorch直接加载)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        for img_path in image_paths:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue

            # 预处理图像
            img = cv2.resize(img, (640, 640))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

            # 推理
            with torch.no_grad():
                pred = model(img)

            # 后处理并保存 (简化版，仅绘制边界框)
            img_name = Path(img_path).stem
            orig_img = cv2.imread(img_path)

            # 假设pred中包含边界框信息 [x1, y1, x2, y2, conf, cls]
            for det in pred[0]:
                if det.shape[0] > 0:
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 保存结果图像
            output_path = f"{output_dir}/{img_name}.jpg"
            cv2.imwrite(output_path, orig_img)
            print(f"已保存检测结果至: {output_path}")

    except Exception as e:
        print(f"备选方法也失败: {e}")
        print("请尝试安装最新版的ultralytics包: pip install ultralytics")
