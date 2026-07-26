import sys
import os
import cv2
import time
import sqlite3
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                               QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, 
                               QMessageBox, QInputDialog, QFrame, QDialog)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from ultralytics import YOLO
from pyproj import Transformer  # 🌟 导入地理坐标转换库

# ================= 1. GPS 经纬度转三维相对坐标工具类 =================
class GPSConverter:
    """将 GPS 经纬度 (WGS-84) 转换为以起飞点为原点 (0,0) 的相对米制坐标 (X, Y)"""
    def __init__(self):
        # 4326 为 WGS-84 经纬度坐标系，3857 为 Web 墨卡托投影坐标系 (单位：米)
        self.transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        self.origin_x = None
        self.origin_y = None

    def gps_to_local_meters(self, lon: float, lat: float):
        """输入经纬度，返回相对原点的 X, Y 距离（单位：米）"""
        x_m, y_m = self.transformer.transform(lon, lat)
        
        # 将第 1 次接收到的 GPS 经纬度设为三维坐标系原点 (0,0)
        if self.origin_x is None or self.origin_y is None:
            self.origin_x = x_m
            self.origin_y = y_m

        rel_x = x_m - self.origin_x
        rel_y = y_m - self.origin_y
        return rel_x, rel_y

    def reset(self):
        """重置原点（开启新巡检时调用）"""
        self.origin_x = None
        self.origin_y = None

# ================= 2. 图传模式选择自定义对话框 =================
class StreamSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🍋 选择实时图传模式")
        self.setFixedSize(360, 200)
        self.selected_source = None
        self.is_custom_rtsp = False

        self.setStyleSheet("""
            QDialog { background-color: #f2f7f2; }
            QLabel { font-size: 15px; font-weight: bold; color: #2d5a27; }
            QPushButton {
                background-color: #ffffff; color: #2e7d32;
                border: 2px solid #a8e6cf; border-radius: 12px;
                padding: 12px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #a8e6cf; color: #1b4332; }
            QPushButton#BtnRTSP { border-color: #fdcb6e; color: #d63031; }
            QPushButton#BtnRTSP:hover { background-color: #ffeaa7; }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("请选择无人机视频信号接入方式：")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_camera = QPushButton("📷 本地摄像头测试 (Camera 0)")
        self.btn_camera.clicked.connect(self.select_camera)

        self.btn_rtsp = QPushButton("🌐 无人机 RTSP 网络图传")
        self.btn_rtsp.setObjectName("BtnRTSP")
        self.btn_rtsp.clicked.connect(self.select_rtsp)

        layout.addWidget(title)
        layout.addWidget(self.btn_camera)
        layout.addWidget(self.btn_rtsp)
        self.setLayout(layout)

    def select_camera(self):
        self.selected_source = 0
        self.accept()

    def select_rtsp(self):
        self.is_custom_rtsp = True
        self.accept()

# ================= 3. 多线程视频流处理与轨迹绘制 =================
class VideoLoaderThread(QThread):
    frame_signal = Signal(np.ndarray)
    trajectory_signal = Signal(float, float, float)  # 相对X(米), 相对Y(米), 高度Z(米)
    raw_gps_signal = Signal(float, float, float)     # 原始经度, 原始纬度, 相对高度
    status_signal = Signal(str)

    def __init__(self, video_source, model_path, is_live=False):
        super().__init__()
        self.video_source = video_source
        self.model_path = model_path
        self.is_live = is_live
        self._is_running = True
        self._is_paused = False
        
        # 实例化 GPS 转换器
        self.gps_converter = GPSConverter()

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False
        self.wait()

    def run(self):
        mode_str = "实时无人机图传" if self.is_live else "本地视频巡检"
        self.status_signal.emit(f"正在载入 [{mode_str}] 与 AI 模型...")
        
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.status_signal.emit(f"模型加载失败: {str(e)}")
            return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.status_signal.emit(f"无法打开视频源: {self.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / (fps if fps > 0 and not self.is_live else 30)
        self.status_signal.emit(f"[{mode_str}] 信号链接成功，开始巡检")

        # 模拟无人机 GPS 基准点（接入真实串口/SDK 时直接替换这里的变量即可）
        base_lon, base_lat = 116.397128, 39.908127
        frame_count = 0
        last_annotated_frame = None

        while cap.isOpened() and self._is_running:
            if self._is_paused:
                time.sleep(0.1)
                continue

            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                if not self.is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.status_signal.emit("实时信号中断，尝试重连中...")
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.video_source)
                    continue

            frame_count += 1

            # 🌟 真实 GPS 接入位置示例：
            # 假设无人机正在沿矩形飞行，动态生成经纬度和高度 (实际对接时换成串口/UDP读到的 real_lon, real_lat, real_alt)
            current_lon = base_lon + (np.sin(frame_count * 0.03) * 0.0002)
            current_lat = base_lat + (np.cos(frame_count * 0.03) * 0.0002)
            current_alt = 15.0 + np.sin(frame_count * 0.05) * 1.5

            # 🌟 1. 将经纬度转为相对原点的米制坐标 (X, Y)
            rel_x_m, rel_y_m = self.gps_converter.gps_to_local_meters(current_lon, current_lat)
            rel_z_m = current_alt

            # YOLO 推理（隔帧渲染提升帧率）
            if frame_count % 2 == 0 or last_annotated_frame is None:
                results = model.predict(frame, imgsz=800, conf=0.25, verbose=False)
                last_annotated_frame = results[0].plot()

            # 🌟 2. 发送图像帧和三维相对坐标(米)
            self.frame_signal.emit(last_annotated_frame)
            self.trajectory_signal.emit(rel_x_m, rel_y_m, rel_z_m)
            self.raw_gps_signal.emit(current_lon, current_lat, current_alt)

            if not self.is_live:
                elapsed_time = time.time() - start_time
                sleep_time = delay - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()

# ================= 4. 地面站 UI 主界面 =================
class UAVGroundStation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍃 智慧农田无人机巡检系统 · 小清新特别版")
        self.resize(1380, 820)

        self.model_path = r"runs/detect/runs/detect/crop_exp1/weights/best.pt"
        self.video_path = "test_drone.mp4"
        self.db_path = "inspection_data.db"
        self.is_live_mode = False
        self.loader_thread = None
        self.last_frame = None
        self.session_start_time = None  # 记录巡检开始时间

        self.init_db()
        self.initUI()

    def init_db(self):
        """初始化 SQLite 数据库表 (巡检主表 + 轨迹明细表)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. 巡检架次主表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspection_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    duration_sec REAL NOT NULL,
                    duration_str TEXT NOT NULL,
                    point_count INTEGER NOT NULL
                )
            """)

            # 2. 轨迹明细表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES inspection_sessions(session_id)
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print("SQLite 建表失败:", e)

    def initUI(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f2f7f2; }
            QWidget { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }
            #HeaderPanel { background-color: #ffffff; border-radius: 16px; border: 1px solid #e1efe2; }
            #VideoContainer, #ChartContainer { background-color: #ffffff; border-radius: 18px; border: 1px solid #dceddd; }
            QLabel#TitleLabel { font-size: 20px; font-weight: bold; color: #2d5a27; }
            QLabel#SubTitleLabel { font-size: 13px; color: #7a9a75; }
            QPushButton {
                background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9;
                border-radius: 12px; padding: 10px 14px; font-size: 13px; font-weight: 600;
            }
            QPushButton:hover { background-color: #c8e6c9; color: #1b5e20; }
            QPushButton#BtnStart { background-color: #a8e6cf; color: #1b4332; border: 1px solid #88d49e; }
            QPushButton#BtnStart:hover { background-color: #88d49e; }
            QPushButton#BtnLive { background-color: #ffeaa7; color: #d63031; border: 1px solid #fdcb6e; }
            QPushButton#BtnLive:hover { background-color: #fdcb6e; }
            QPushButton#BtnPause { background-color: #fff3bf; color: #f59f00; border: 1px solid #ffe066; }
            QPushButton#BtnPause:hover { background-color: #ffe066; }
            QPushButton#BtnSave { background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
            QPushButton#BtnSave:hover { background-color: #90caf9; color: #0d47a1; }
            QPushButton#BtnClear { background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
            QPushButton#BtnClear:hover { background-color: #ef9a9a; color: #b71c1c; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # 顶部标题栏
        header_panel = QFrame()
        header_panel.setObjectName("HeaderPanel")
        header_layout = QHBoxLayout(header_panel)
        header_layout.setContentsMargins(20, 10, 20, 10)

        title_vbox = QVBoxLayout()
        title_label = QLabel("🍋 智慧农业 · 无人机巡检地面站 🍋")
        title_label.setObjectName("TitleLabel")
        subtitle_label = QLabel("YOLOv8 农田作物智能识别系统 | GPS 投影 3D 相对航迹 (单位: 米)")
        subtitle_label.setObjectName("SubTitleLabel")
        title_vbox.addWidget(title_label)
        title_vbox.addWidget(subtitle_label)

        tag_label = QLabel("柚子特调清新风 · 🌱 绿色农田版")
        tag_label.setStyleSheet("background-color: #f7fee7; color: #65a30d; border: 1px solid #d9f99d; padding: 6px 12px; border-radius: 20px; font-weight: bold;")

        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        header_layout.addWidget(tag_label)

        # 主内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # 左侧：视频控制面板
        left_container = QFrame()
        left_container.setObjectName("VideoContainer")
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(12, 12, 12, 12)

        self.video_label = QLabel("🍃 画面就绪，等待启动巡检...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #f8faf8; color: #52796f; font-size: 15px; border-radius: 12px; border: 1px dashed #b7e4c7;")
        self.video_label.setMinimumSize(600, 450)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_select_video = QPushButton("📁 选择视频")
        self.btn_select_video.clicked.connect(self.select_video)

        self.btn_live_stream = QPushButton("🍋 实时图传")
        self.btn_live_stream.setObjectName("BtnLive")
        self.btn_live_stream.clicked.connect(self.start_live_stream)

        self.btn_start = QPushButton("▶ 启动巡检")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_pause = QPushButton("⏸ 暂停巡检")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.clicked.connect(self.pause_detection)

        self.btn_save = QPushButton("💾 保存数据")
        self.btn_save.setObjectName("BtnSave")
        self.btn_save.clicked.connect(self.save_data)

        self.btn_clear = QPushButton("🧹 清空巡检")
        self.btn_clear.setObjectName("BtnClear")
        self.btn_clear.clicked.connect(self.clear_data)

        btn_layout.addWidget(self.btn_select_video)
        btn_layout.addWidget(self.btn_live_stream)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_clear)

        left_layout.addWidget(self.video_label, stretch=1)
        left_layout.addSpacing(8)
        left_layout.addLayout(btn_layout)

        # 右侧：3D 轨迹视口
        right_container = QFrame()
        right_container.setObjectName("ChartContainer")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 10, 8, 8)
        right_layout.setSpacing(6)

        chart_title = QLabel(" 🛸 相对起飞点 3D 飞行航迹 (单位: 米 | X-红 Y-绿 Z-蓝)")
        chart_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2d5a27;")
        right_layout.addWidget(chart_title)

        # 实时 GPS 显示文本框
        self.gps_info_label = QLabel("📍 真实 GPS: 经度 --° | 纬度 --° | 相对高度 -- m")
        self.gps_info_label.setStyleSheet("font-size: 12px; color: #388e3c; background-color: #e8f5e9; padding: 4px 8px; border-radius: 6px;")
        right_layout.addWidget(self.gps_info_label)

        self.view3d = gl.GLViewWidget()
        self.view3d.setBackgroundColor('#121e17')
        self.view3d.setCameraPosition(distance=50, elevation=30, azimuth=45)

        grid = gl.GLGridItem()
        grid.setSize(x=100, y=100, z=0)
        grid.setSpacing(x=5, y=5, z=5)
        grid.setColor((180/255, 230/255, 190/255, 0.3))
        self.view3d.addItem(grid)

        self.axis_x = gl.GLLinePlotItem(color=(1.0, 0.25, 0.25, 1.0), width=6, antialias=True)
        self.axis_y = gl.GLLinePlotItem(color=(0.25, 1.0, 0.25, 1.0), width=6, antialias=True)
        self.axis_z = gl.GLLinePlotItem(color=(0.25, 0.65, 1.0, 1.0), width=6, antialias=True)
        self.view3d.addItem(self.axis_x)
        self.view3d.addItem(self.axis_y)
        self.view3d.addItem(self.axis_z)

        self.text_x = gl.GLTextItem(pos=(27, 0, 0), text='X(m)', color=(255, 100, 100, 255))
        self.text_y = gl.GLTextItem(pos=(0, 27, 0), text='Y(m)', color=(100, 255, 100, 255))
        self.text_z = gl.GLTextItem(pos=(0, 0, 27), text='Z(m)', color=(100, 200, 255, 255))
        self.view3d.addItem(self.text_x)
        self.view3d.addItem(self.text_y)
        self.view3d.addItem(self.text_z)

        self.pos_data = []
        self.plot3d = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=(255/255, 230/255, 100/255, 1.0), width=3, antialias=True)
        self.view3d.addItem(self.plot3d)

        self.drone_point = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1.0, 0.6, 0.2, 1.0), size=12)
        self.view3d.addItem(self.drone_point)

        right_layout.addWidget(self.view3d, stretch=1)

        content_layout.addWidget(left_container, stretch=55)
        content_layout.addWidget(right_container, stretch=45)

        root_layout.addWidget(header_panel)
        root_layout.addLayout(content_layout, stretch=1)
        main_widget.setLayout(root_layout)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择农田巡检视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.is_live_mode = False
            QMessageBox.information(self, "提示", f"已选择视频: {os.path.basename(file_path)}")

    def start_live_stream(self):
        dialog = StreamSourceDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            video_source = None
            if dialog.selected_source is not None:
                video_source = dialog.selected_source
            elif dialog.is_custom_rtsp:
                url, ok = QInputDialog.getText(
                    self, "🌐 输入 RTSP 视频流地址", 
                    "请输入无人机 RTSP / HTTP 图传网址:",
                    text="rtsp://192.168.1.100:554/live"
                )
                if ok and url.strip():
                    video_source = url.strip()

            if video_source is not None:
                self.stop_thread()
                self.is_live_mode = True
                self.pos_data = []
                self.session_start_time = datetime.now()
                self.loader_thread = VideoLoaderThread(video_source, self.model_path, is_live=True)
                self.loader_thread.frame_signal.connect(self.update_image)
                self.loader_thread.trajectory_signal.connect(self.update_3d_trajectory)
                self.loader_thread.raw_gps_signal.connect(self.update_raw_gps_text)
                self.loader_thread.status_signal.connect(self.update_status)
                self.loader_thread.start()

    def update_status(self, text):
        if not self.loader_thread or not self.loader_thread._is_paused:
            self.video_label.setText(text)

    def update_image(self, cv_img):
        self.last_frame = cv_img.copy()
        self.render_frame(cv_img)

    def render_frame(self, cv_img, is_paused=False):
        display_img = cv_img.copy()

        if self.is_live_mode and not is_paused:
            cv2.circle(display_img, (35, 35), 8, (0, 200, 255), -1)
            cv2.putText(display_img, "LIVE STREAM (UAV Direct)", (52, 41), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)

        h_img, w_img, _ = display_img.shape
        cv2.putText(display_img, "LEMON YUZU AI", (w_img - 220, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 240, 160), 2, cv2.LINE_AA)

        if is_paused:
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (w_img, h_img), (30, 50, 35), -1)
            cv2.addWeighted(overlay, 0.45, display_img, 0.55, 0, display_img)

            text = "PAUSED - 巡检已暂停"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.1
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (w_img - text_size[0]) // 2
            text_y = (h_img + text_size[1]) // 2

            bar_w, bar_h = 18, 55
            gap = 14
            center_x, center_y = w_img // 2, h_img // 2 - 35
            cv2.rectangle(display_img, (center_x - bar_w - gap//2, center_y - bar_h//2), 
                          (center_x - gap//2, center_y + bar_h//2), (180, 245, 120), -1)
            cv2.rectangle(display_img, (center_x + gap//2, center_y - bar_h//2), 
                          (center_x + bar_w + gap//2, center_y + bar_h//2), (180, 245, 120), -1)

            cv2.putText(display_img, text, (text_x, text_y + 40), font, font_scale, (200, 255, 150), thickness, cv2.LINE_AA)

        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def update_3d_trajectory(self, x_m, y_m, z_m):
        """更新 3D 轨迹视图（接收相对原点的距离，单位：米）"""
        self.pos_data.append([x_m, y_m, z_m])
        pts = np.array(self.pos_data)
        self.plot3d.setData(pos=pts)
        self.drone_point.setData(pos=np.array([[x_m, y_m, z_m]]))

        max_dist = max(abs(x_m), abs(y_m), abs(z_m), 25.0)
        axis_len = max_dist + 5.0  

        self.axis_x.setData(pos=np.array([[0, 0, 0], [axis_len, 0, 0]]))
        self.axis_y.setData(pos=np.array([[0, 0, 0], [0, axis_len, 0]]))
        self.axis_z.setData(pos=np.array([[0, 0, 0], [0, 0, axis_len]]))

        self.text_x.setData(pos=(axis_len + 2, 0, 0))
        self.text_y.setData(pos=(0, axis_len + 2, 0))
        self.text_z.setData(pos=(0, 0, axis_len + 2))

    def update_raw_gps_text(self, lon, lat, alt):
        """更新显示真实 GPS 信息的文本控件"""
        self.gps_info_label.setText(f"📍 真实 GPS: 经度 {lon:.6f}° | 纬度 {lat:.6f}° | 相对高度 {alt:.1f} m")

    def start_detection(self):
        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, "错误", f"找不到模型文件:\n{self.model_path}")
            return

        if self.loader_thread and self.loader_thread.isRunning():
            if self.loader_thread._is_paused:
                self.loader_thread.resume()
                return

        self.stop_thread()
        self.pos_data = []
        self.is_live_mode = False
        self.session_start_time = datetime.now()
        
        self.loader_thread = VideoLoaderThread(self.video_path, self.model_path, is_live=False)
        self.loader_thread.frame_signal.connect(self.update_image)
        self.loader_thread.trajectory_signal.connect(self.update_3d_trajectory)
        self.loader_thread.raw_gps_signal.connect(self.update_raw_gps_text)
        self.loader_thread.status_signal.connect(self.update_status)
        self.loader_thread.start()

    def pause_detection(self):
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.pause()
            if self.last_frame is not None:
                self.render_frame(self.last_frame, is_paused=True)

    def save_data(self):
        if not self.pos_data:
            QMessageBox.warning(self, "提示", "当前没有可保存的巡检轨迹数据！")
            return

        start_time = getattr(self, 'session_start_time', datetime.now())
        end_time = datetime.now()
        duration_sec = (end_time - start_time).total_seconds()
        
        minutes, seconds = divmod(int(duration_sec), 60)
        hours, minutes = divmod(minutes, 60)
        duration_str = f"{hours:02d}时{minutes:02d}分{seconds:02d}秒" if hours > 0 else f"{minutes:02d}分{seconds:02d}秒"

        session_id = start_time.strftime("Session_%Y%m%d_%H%M%S")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. 写入巡检架次主表
            cursor.execute("""
                INSERT INTO inspection_sessions 
                (session_id, start_time, end_time, duration_sec, duration_str, point_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S"),
                round(duration_sec, 2),
                duration_str,
                len(self.pos_data)
            ))
            
            # 2. 批量写入轨迹明细表
            records = [
                (session_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pt[0], pt[1], pt[2]) 
                for pt in self.pos_data
            ]
            cursor.executemany("""
                INSERT INTO trajectory_logs (session_id, timestamp, x, y, z)
                VALUES (?, ?, ?, ?, ?)
            """, records)
            
            conn.commit()
            conn.close()

            QMessageBox.information(
                self, "保存成功", 
                f"✅ 巡检报告及轨迹坐标已成功入库！\n\n"
                f"🏷 巡检架次: {session_id}\n"
                f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⏱ 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⏳ 巡检时长: {duration_str} ({duration_sec:.1f} 秒)\n"
                f"📍 采节点数: {len(self.pos_data)} 个\n\n"
                f"📁 数据库路径: {os.path.abspath(self.db_path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "数据库错误", f"保存至 SQLite 失败:\n{str(e)}")

    def clear_data(self):
        self.stop_thread()

        self.pos_data = []
        self.last_frame = None
        self.session_start_time = None
        self.plot3d.setData(pos=np.zeros((1, 3)))
        self.drone_point.setData(pos=np.zeros((1, 3)))

        self.axis_x.setData(pos=np.array([[0, 0, 0], [25, 0, 0]]))
        self.axis_y.setData(pos=np.array([[0, 0, 0], [0, 25, 0]]))
        self.axis_z.setData(pos=np.array([[0, 0, 0], [0, 0, 25]]))
        self.text_x.setData(pos=(27, 0, 0))
        self.text_y.setData(pos=(0, 27, 0))
        self.text_z.setData(pos=(0, 0, 27))

        self.gps_info_label.setText("📍 真实 GPS: 经度 --° | 纬度 --° | 相对高度 -- m")
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("🍃 画面已初始化，等待启动巡检...")

        QMessageBox.information(self, "提示", "巡检状态与数据已重置！")

    def stop_thread(self):
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UAVGroundStation()
    win.show()
    sys.exit(app.exec())