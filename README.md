# 智慧农田无人机巡检与植保地面站

> 基于 **PySide6 + YOLO + PyOpenGL** 的无人机植保地面站：输入农田 GPS 起点与范围，自动生成**弓字形全覆盖巡航航线**；多线程视频流接入 YOLO 实时检测病虫害，3D 视图同时渲染规划航线与实际飞行轨迹，巡检报告入库 SQLite。

## 功能特性

- **弓字形全覆盖航线规划**（Lawnmower Pattern）：输入起点 GPS、农田宽/长、喷幅行距（0.1m~10km）、巡航高度与速度，自动生成往复覆盖航线
- **GPS 坐标转换**：基于 pyproj 实现 WGS84 → Web Mercator（EPSG:4326→3857）投影，以起飞点为原点换算米制航线坐标
- **多线程视频流处理**：QThread 分离视频解码 / YOLO 推理 / UI 渲染，支持本地视频、摄像头与 **RTSP 网络流**三种输入源
- **YOLO 病虫害检测**：ultralytics 推理（视频流实时 + 静态图片两种模式），目标框实时绘制
- **3D 双航迹渲染**：pyqtgraph.opengl 三维展示**规划航线（天蓝）与实际飞行轨迹（黄色）**对比，直观反映执行偏差；实际轨迹按飞行速度动态推算
- **巡检数据持久化**：巡检报告、轨迹坐标、图片检测结果保存 SQLite，支持历史查询
- **飞行数据曲线**：PyQtGraph 实时绘制飞行数据图表

## 技术栈

PySide6 · Ultralytics YOLOv8 · OpenCV · PyQtGraph（含 OpenGL 3D）· Pyproj · SQLite3

## 快速开始

```bash
pip install PySide6 opencv-python ultralytics pyqtgraph pyopengl pyproj numpy

python main_ui.py
```

1. 设置起点 GPS 与农田范围参数 → 生成弓字形航线
2. 接入视频源（视频文件 / 摄像头 / RTSP）开始检测
3. 巡检结束保存报告，3D 视图查看规划 vs 实际航迹

## 项目结构

```
field/
├── main_ui.py        # 地面站主程序（UI + 航线规划 + 多线程检测 + 3D 渲染）
├── train.py          # YOLO 模型训练脚本
├── test_model.py     # 模型推理测试
├── fix_labels.py     # 数据集标注修复工具
├── data.yaml         # 数据集配置
├── datasets/ runs/   # 数据集与训练产出
└── inspection_data.db # 巡检数据（运行自动生成）
```

## License

仅供学习与交流使用。
