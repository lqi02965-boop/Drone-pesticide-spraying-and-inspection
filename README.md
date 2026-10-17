# 智慧农田无人机巡检与植保地面站

> 基于 **PySide6 + YOLO + OpenCV** 的无人机地面站桌面系统：病虫害视觉识别、农田巡航航线规划、3D 轨迹渲染三位一体。

## 功能特性

- **多源视觉检测**：整合 OpenCV 与 YOLO 模型，支持视频流实时检测与静态图像检测两种模式
- **多线程渲染**：视频解码、推理、界面渲染分离线程，高帧率画面 + 目标框实时绘制不卡顿
- **航线规划**：农田区块巡航路径规划，支持航线数据管理
- **3D 轨迹渲染**：PyOpenGL 三维展示无人机飞行轨迹，PyQtGraph 绘制飞行数据曲线
- **数据持久化**：SQLite3 存储巡检日志与轨迹点流，支持历史回放管理
- **坐标转换**：Pyproj 处理地理坐标系换算

## 技术栈

PySide6 · YOLO · OpenCV · PyQtGraph · PyOpenGL · Pyproj · SQLite3

## 快速开始

```bash
pip install PySide6 opencv-python ultralytics pyqtgraph PyOpenGL pyproj
python main.py
```

## License

仅供学习与交流使用。
