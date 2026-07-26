import os
import cv2
from ultralytics import YOLO

# 1. 加载刚才训练好的最佳模型权重
model_path = r"runs/detect/runs/detect/crop_exp1/weights/best.pt"
model = YOLO(model_path)

# 2. 测试数据集验证集里的图片
val_img_dir = "./datasets/val/images"

# 获取验证集里的前 3 张图片进行测试
img_names = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:3]

print("开始测试验证集图片...")
for name in img_names:
    img_path = os.path.join(val_img_dir, name)
    
    # 预测图片（conf=0.25 是置信度阈值）
    results = model.predict(source=img_path, conf=0.25, save=True, project="runs/detect", name="test_results")
    print(f"图片 {name} 测试完成，识别到 {len(results[0].boxes)} 个目标。")

print(f"\n🎉 测试完成！结果图片已保存在: runs/detect/test_results")