import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载官方预训练权重（首次运行会自动从网络下载 yolov8m.pt）
    # YOLOv8m (Medium) 参数量和精度非常适合农作物及小目标识别
    model = YOLO('yolov8m.pt') 

    # 2. 获取当前 train.py 所在的绝对路径，确保配置文件路径绝对不出错
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'data.yaml')

    # 3. 开始模型训练
    results = model.train(
        data=data_yaml_path,     # 指定配置文件 path
        epochs=100,              # 训练 100 轮（随时可用 Ctrl+C 提前终止并保留最佳模型）
        imgsz=800,               # 图像输入尺寸 800x800（极大幅度提升无人机航拍等小目标精度）
        batch=8,                 # Batch size 设为 8，完美适配 RTX 4060 8GB 显存，防止 OOM
        workers=4,               # CPU 多线程加载数据
        device=0,                # 使用 RTX 4060 独立显卡 GPU 0
        amp=True,                # 开启 16 位混合精度训练，显著提升训练速度并省显存
        mosaic=1.0,              # 开启 100% Mosaic 拼接数据增强，大幅增强小样本和复杂场景泛化
        project='runs/detect',   # 训练日志与模型权重的保存目录
        name='crop_exp1',        # 本次实验保存的文件夹名称
        save=True,               # 自动保存权重文件
        plots=True               # 自动生成混淆矩阵、P-R曲线、损失函数折线图等结果图
    )