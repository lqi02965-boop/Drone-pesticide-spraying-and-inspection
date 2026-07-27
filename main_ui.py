import sys
import os
import cv2
import time
import sqlite3
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                               QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, 
                               QMessageBox, QInputDialog, QFrame, QDialog, QDoubleSpinBox, 
                               QLineEdit, QGroupBox, QFormLayout)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from ultralytics import YOLO
from pyproj import Transformer  # 地理坐标转换库

# ================= 1. GPS 经纬度转三维相对坐标工具类 =================
class GPSConverter:
    """将 GPS 经纬度 (WGS-84) 转换为以起飞点为原点 (0,0) 的相对米制坐标 (X, Y)"""
    def __init__(self):
        self.transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        self.origin_x = None
        self.origin_y = None

    def set_origin(self, lon: float, lat: float):
        self.origin_x, self.origin_y = self.transformer.transform(lon, lat)

    def gps_to_local_meters(self, lon: float, lat: float):
        """输入经纬度，返回相对原点的 X, Y 距离（单位：米）"""
        x_m, y_m = self.transformer.transform(lon, lat)
        
        if self.origin_x is None or self.origin_y is None:
            self.origin_x = x_m
            self.origin_y = y_m

        rel_x = x_m - self.origin_x
        rel_y = y_m - self.origin_y
        return rel_x, rel_y

    def reset(self):
        self.origin_x = None
        self.origin_y = None

# ================= 2. 农田弓字形自动巡航航线规划器 =================
class CoveragePathPlanner:
    """农田全覆盖往复航线规划器 (Lawnmower Pattern)"""
    def __init__(self, spray_width=4.0):
        self.spray_width = spray_width  # 喷幅 / 行距 (单位: 米)

    def generate_lawnmower_path(self, start_x=0.0, start_y=0.0, field_width=40.0, field_length=40.0, cruise_alt=15.0):
        """根据农田宽(X方向)、长(Y方向)和行距，生成弓字形往复航线"""
        waypoints = []
        min_x, max_x = start_x, start_x + field_width
        min_y, max_y = start_y, start_y + field_length

        y_steps = np.arange(min_y, max_y + 0.1, self.spray_width)
        direction = 1

        for y in y_steps:
            if direction == 1:
                waypoints.append((min_x, y, cruise_alt))
                waypoints.append((max_x, y, cruise_alt))
            else:
                waypoints.append((max_x, y, cruise_alt))
                waypoints.append((min_x, y, cruise_alt))
            direction *= -1

        return waypoints

# ================= 3. 图传模式选择自定义对话框 =================
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

# ================= 4. 多线程视频流处理与轨迹绘制 =================
class VideoLoaderThread(QThread):
    frame_signal = Signal(np.ndarray)
    trajectory_signal = Signal(float, float, float)  # 相对X(米), 相对Y(米), 高度Z(米)
    raw_gps_signal = Signal(float, float, float)     # 原始经度, 原始纬度, 相对高度
    status_signal = Signal(str)

    def __init__(self, video_source, model_path, is_live=False, auto_waypoints=None, base_gps=(116.397128, 39.908127), speed_m_s=5.0):
        super().__init__()
        self.video_source = video_source
        self.model_path = model_path
        self.is_live = is_live
        self.auto_waypoints = auto_waypoints  # 巡航航点数据
        self.base_lon, self.base_lat = base_gps
        self.speed_m_s = speed_m_s            # 飞行速度 m/s
        self._is_running = True
        self._is_paused = False
        
        self.gps_converter = GPSConverter()

        self.wp_idx = 0
        self.sub_step = 0

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False

    def run(self):
        mode_str = "自动巡航喷洒" if self.auto_waypoints else ("实时无人机图传" if self.is_live else "本地视频巡检")
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
        fps = fps if fps > 0 else 30.0
        delay = 1.0 / (fps if not self.is_live else 30.0)
        self.status_signal.emit(f"[{mode_str}] 模式建立成功，开始作业")

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

            # ================= 动态轨迹坐标计算（结合飞行速度） =================
            if self.auto_waypoints and len(self.auto_waypoints) > 1:
                p1 = self.auto_waypoints[self.wp_idx]
                p2 = self.auto_waypoints[min(self.wp_idx + 1, len(self.auto_waypoints) - 1)]
                
                segment_dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
                step_dist = self.speed_m_s * (1.0 / fps)
                total_steps = max(int(segment_dist / step_dist), 1)

                alpha = self.sub_step / float(total_steps)
                rel_x_m = p1[0] + (p2[0] - p1[0]) * alpha
                rel_y_m = p1[1] + (p2[1] - p1[1]) * alpha
                rel_z_m = p1[2] + (p2[2] - p1[2]) * alpha

                self.sub_step += 1
                if self.sub_step >= total_steps:
                    self.sub_step = 0
                    if self.wp_idx < len(self.auto_waypoints) - 2:
                        self.wp_idx += 1

                current_lon = self.base_lon + (rel_x_m / 111000.0)
                current_lat = self.base_lat + (rel_y_m / 111000.0)
                current_alt = rel_z_m

            elif self.is_live:
                current_lon = self.base_lon
                current_lat = self.base_lat
                current_alt = 15.0
                rel_x_m, rel_y_m = self.gps_converter.gps_to_local_meters(current_lon, current_lat)
                rel_z_m = current_alt

            else:
                speed_m_per_frame = self.speed_m_s * (1.0 / fps)
                rel_x_m = frame_count * speed_m_per_frame
                rel_y_m = (frame_count * 0.05) % 10.0
                rel_z_m = 15.0 + np.sin(frame_count * 0.05) * 0.5

                current_lon = self.base_lon + (rel_x_m / 111000.0)
                current_lat = self.base_lat + (rel_y_m / 111000.0)
                current_alt = rel_z_m

            if frame_count % 2 == 0 or last_annotated_frame is None:
                results = model.predict(frame, imgsz=800, conf=0.25, verbose=False)
                last_annotated_frame = results[0].plot()

            self.frame_signal.emit(last_annotated_frame)
            self.trajectory_signal.emit(rel_x_m, rel_y_m, rel_z_m)
            self.raw_gps_signal.emit(current_lon, current_lat, current_alt)

            if not self.is_live:
                elapsed_time = time.time() - start_time
                sleep_time = delay - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()

# ================= 5. 地面站 UI 主界面 =================
class UAVGroundStation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍃 智慧农田无人机巡检系统 · 视频与图像双模式版")
        self.resize(1480, 880)

        self.model_path = r"runs/detect/runs/detect/crop_exp1/weights/best.pt"
        self.video_path = "test_drone.mp4"
        self.image_path = None
        self.current_live_source = None
        self.db_path = "inspection_data.db"
        self.is_live_mode = False
        self.is_image_mode = False  # 新增：是否处于静态图片检测模式
        self.loader_thread = None
        self.last_frame = None
        self.session_start_time = None
        
        self.planned_waypoints = []

        self.init_db()
        self.initUI()

    def init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
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
            
            QLabel { color: #1b4332; font-size: 13px; font-weight: 600; }
            QLabel#TitleLabel { font-size: 20px; font-weight: bold; color: #1b4332; }
            QLabel#SubTitleLabel { font-size: 13px; color: #52796f; }
            
            QGroupBox { 
                font-size: 14px; font-weight: bold; color: #1b4332; 
                border: 2px solid #a8e6cf; border-radius: 10px; 
                margin-top: 8px; padding-top: 12px; 
            }
            
            QLineEdit, QDoubleSpinBox { 
                border: 2px solid #88d49e; 
                border-radius: 6px; 
                padding: 5px 8px; 
                background-color: #ffffff; 
                color: #1a3c1e; 
                font-size: 14px; 
                font-weight: bold; 
            }
            
            QLineEdit:focus, QDoubleSpinBox:focus { 
                border: 2px solid #2d6a4f; 
                background-color: #f8fff9; 
            }

            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 18px;
                background-color: #e8f5e9;
                border: 1px solid #a8e6cf;
                border-radius: 3px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #b7e4c7;
            }

            QPushButton {
                background-color: #e8f5e9; color: #1b4332; border: 1px solid #a8e6cf;
                border-radius: 10px; padding: 8px 12px; font-size: 13px; font-weight: bold;
            }
            QPushButton:hover { background-color: #c8e6c9; color: #081c15; }
            
            QPushButton#BtnStart { background-color: #a8e6cf; color: #1b4332; border: 1px solid #74c69d; }
            QPushButton#BtnStart:hover { background-color: #74c69d; }
            
            QPushButton#BtnImg { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #81c784; }
            QPushButton#BtnImg:hover { background-color: #c8e6c9; }

            QPushButton#BtnLive { background-color: #ffeaa7; color: #d63031; border: 1px solid #fdcb6e; }
            QPushButton#BtnLive:hover { background-color: #fdcb6e; }
            
            QPushButton#BtnAuto { background-color: #0288d1; color: #ffffff; border: 1px solid #0277bd; font-size: 14px; }
            QPushButton#BtnAuto:hover { background-color: #01579b; }
            
            QPushButton#BtnPause { background-color: #fff3bf; color: #d97706; border: 1px solid #ffe066; }
            QPushButton#BtnSave { background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
            QPushButton#BtnClear { background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header_panel = QFrame()
        header_panel.setObjectName("HeaderPanel")
        header_layout = QHBoxLayout(header_panel)
        header_layout.setContentsMargins(20, 10, 20, 10)

        title_vbox = QVBoxLayout()
        title_label = QLabel("🍋 智慧农业 · 自定义农田区域航线规划地面站 🍋")
        title_label.setObjectName("TitleLabel")
        subtitle_label = QLabel("自定义起点 GPS + 农田范围 + 弓字形算法全自动覆盖巡航")
        subtitle_label.setObjectName("SubTitleLabel")
        title_vbox.addWidget(title_label)
        title_vbox.addWidget(subtitle_label)

        tag_label = QLabel("植保作业模式 · 🗺 区域航线规划")
        tag_label.setStyleSheet("background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; padding: 6px 12px; border-radius: 20px; font-weight: bold;")

        header_layout.addLayout(title_vbox)
        header_layout.addStretch()
        header_layout.addWidget(tag_label)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        left_container = QFrame()
        left_container.setObjectName("VideoContainer")
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(12, 12, 12, 12)

        self.video_label = QLabel("🍃 画面就绪，请选择 [🎬 视频] 或 [📷 照片] 进行巡检检测 ...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #f8faf8; color: #2d6a4f; font-size: 15px; font-weight: bold; border-radius: 12px; border: 2px dashed #b7e4c7;")
        self.video_label.setMinimumSize(600, 450)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_select_video = QPushButton("🎬 选择视频")
        self.btn_select_video.clicked.connect(self.select_video)

        # 🌟 新增：图片上传检测按钮
        self.btn_select_image = QPushButton("📷 选择照片")
        self.btn_select_image.setObjectName("BtnImg")
        self.btn_select_image.clicked.connect(self.select_and_detect_image)

        self.btn_live_stream = QPushButton("🍋 实时图传")
        self.btn_live_stream.setObjectName("BtnLive")
        self.btn_live_stream.clicked.connect(self.start_live_stream)

        self.btn_start = QPushButton("▶ 启动巡检")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.clicked.connect(self.pause_detection)

        self.btn_save = QPushButton("💾 保存数据")
        self.btn_save.setObjectName("BtnSave")
        self.btn_save.clicked.connect(self.save_data)

        self.btn_clear = QPushButton("🧹 清空")
        self.btn_clear.setObjectName("BtnClear")
        self.btn_clear.clicked.connect(self.clear_data)

        btn_layout.addWidget(self.btn_select_video)
        btn_layout.addWidget(self.btn_select_image)  # 添加到界面布局
        btn_layout.addWidget(self.btn_live_stream)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_clear)

        left_layout.addWidget(self.video_label, stretch=1)
        left_layout.addSpacing(6)
        left_layout.addLayout(btn_layout)

        right_container = QFrame()
        right_container.setObjectName("ChartContainer")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(6)

        chart_title = QLabel(" 🛸 相对起飞点 3D 飞行航迹 (天蓝: 规划航线 | 黄色: 实际飞行)")
        chart_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #1b4332;")
        right_layout.addWidget(chart_title)

        self.gps_info_label = QLabel("📍 真实 GPS: 经度 --° | 纬度 --° | 相对高度 -- m")
        self.gps_info_label.setStyleSheet("font-size: 12px; color: #1b4332; font-weight: bold; background-color: #d8f3dc; padding: 6px 10px; border-radius: 6px;")
        right_layout.addWidget(self.gps_info_label)

        auto_group = QGroupBox("🗺 农田自定义区域与巡航参数")
        auto_vbox = QVBoxLayout(auto_group)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(6, 6, 6, 6)
        form_layout.setSpacing(8)

        self.edit_start_gps = QLineEdit("116.397128, 39.908127")
        self.edit_start_gps.setPlaceholderText("经度, 纬度 (例: 116.397128, 39.908127)")
        form_layout.addRow("起点经纬度:", self.edit_start_gps)

        size_layout = QHBoxLayout()
        self.spin_field_w = QDoubleSpinBox()
        self.spin_field_w.setDecimals(2)
        self.spin_field_w.setRange(1.0, 10000000.0)  # 1m ~ 1000万米
        self.spin_field_w.setValue(500.0)             # 默认 500m
        
        self.spin_field_h = QDoubleSpinBox()
        self.spin_field_h.setDecimals(2)
        self.spin_field_h.setRange(1.0, 10000000.0)  # 1m ~ 1000万米
        self.spin_field_h.setValue(400.0)             # 默认 400m

        size_layout.addWidget(QLabel("宽(m):"))
        size_layout.addWidget(self.spin_field_w)
        size_layout.addWidget(QLabel("长(m):"))
        size_layout.addWidget(self.spin_field_h)
        form_layout.addRow("区域尺寸:", size_layout)

        param_layout = QHBoxLayout()
        self.spin_spray_w = QDoubleSpinBox()
        self.spin_spray_w.setDecimals(2)
        self.spin_spray_w.setRange(0.1, 10000.0)     # 行距 0.1m ~ 1万米
        self.spin_spray_w.setValue(5.0)

        self.spin_cruise_alt = QDoubleSpinBox()
        self.spin_cruise_alt.setDecimals(2)
        self.spin_cruise_alt.setRange(1.0, 10000.0)   # 高度 1m ~ 1万米
        self.spin_cruise_alt.setValue(15.0)

        # 🌟 新增：飞行速度控制参数
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setDecimals(1)
        self.spin_speed.setRange(0.5, 30.0)
        self.spin_speed.setValue(5.0)

        param_layout.addWidget(QLabel("行距(m):"))
        param_layout.addWidget(self.spin_spray_w)
        param_layout.addWidget(QLabel("高度(m):"))
        param_layout.addWidget(self.spin_cruise_alt)
        param_layout.addWidget(QLabel("速度(m/s):"))
        param_layout.addWidget(self.spin_speed)
        form_layout.addRow("航线参数:", param_layout)

        auto_vbox.addLayout(form_layout)

        auto_btn_layout = QHBoxLayout()
        self.btn_plan = QPushButton("📐 生成自定义航线")
        self.btn_plan.clicked.connect(self.plan_auto_mission)

        self.btn_auto_start = QPushButton("🚀 开启自动巡航")
        self.btn_auto_start.setObjectName("BtnAuto")
        self.btn_auto_start.clicked.connect(self.start_auto_cruise)

        auto_btn_layout.addWidget(self.btn_plan)
        auto_btn_layout.addWidget(self.btn_auto_start)
        auto_vbox.addLayout(auto_btn_layout)

        right_layout.addWidget(auto_group)

        # 3D 视图定义
        self.view3d = gl.GLViewWidget()
        self.view3d.setBackgroundColor('#121e17')
        self.view3d.setCameraPosition(distance=150, elevation=30, azimuth=45)

        self.grid = gl.GLGridItem()
        self.grid.setSize(x=1000, y=1000, z=0)
        self.grid.setSpacing(x=50, y=50, z=10)
        self.grid.setColor((180/255, 230/255, 190/255, 0.3))
        self.view3d.addItem(self.grid)

        self.axis_x = gl.GLLinePlotItem(color=(1.0, 0.25, 0.25, 1.0), width=6, antialias=True)
        self.axis_y = gl.GLLinePlotItem(color=(0.25, 1.0, 0.25, 1.0), width=6, antialias=True)
        self.axis_z = gl.GLLinePlotItem(color=(0.25, 0.65, 1.0, 1.0), width=6, antialias=True)
        self.view3d.addItem(self.axis_x)
        self.view3d.addItem(self.axis_y)
        self.view3d.addItem(self.axis_z)

        self.text_x = gl.GLTextItem(pos=(100, 0, 0), text='X(m)', color=(255, 100, 100, 255))
        self.text_y = gl.GLTextItem(pos=(0, 100, 0), text='Y(m)', color=(100, 255, 100, 255))
        self.text_z = gl.GLTextItem(pos=(0, 0, 100), text='Z(m)', color=(100, 200, 255, 255))
        self.view3d.addItem(self.text_x)
        self.view3d.addItem(self.text_y)
        self.view3d.addItem(self.text_z)

        self.planned_plot3d = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=(0.2, 0.8, 1.0, 0.8), width=3, antialias=True)
        self.view3d.addItem(self.planned_plot3d)

        self.pos_data = []
        self.plot3d = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=(255/255, 230/255, 100/255, 1.0), width=4, antialias=True)
        self.view3d.addItem(self.plot3d)

        self.drone_point = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1.0, 0.6, 0.2, 1.0), size=12)
        self.view3d.addItem(self.drone_point)

        right_layout.addWidget(self.view3d, stretch=1)

        content_layout.addWidget(left_container, stretch=50)
        content_layout.addWidget(right_container, stretch=50)

        root_layout.addWidget(header_panel)
        root_layout.addLayout(content_layout, stretch=1)
        main_widget.setLayout(root_layout)

    def parse_gps_input(self):
        text = self.edit_start_gps.text().strip()
        try:
            parts = [float(x.strip()) for x in text.replace('，', ',').split(',')]
            if len(parts) == 2:
                return parts[0], parts[1]
        except Exception:
            pass
        return 116.397128, 39.908127

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择农田巡检视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.is_live_mode = False
            self.is_image_mode = False
            QMessageBox.information(self, "提示", f"已选择视频: {os.path.basename(file_path)}")

    # 🌟 新增：图片选择与 YOLO 检测处理逻辑
    def select_and_detect_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择农田拍照照片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return

        self.stop_thread()
        self.is_image_mode = True
        self.is_live_mode = False
        self.image_path = file_path

        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, "错误", f"找不到 AI 模型文件:\n{self.model_path}")
            return

        try:
            # 1. 载入图像
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.critical(self, "错误", "图片读取失败，请检查文件格式！")
                return

            # 2. 调用 YOLO 模型推理
            model = YOLO(self.model_path)
            results = model.predict(img, imgsz=800, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()

            # 3. 在 3D 图中标注拍摄位置（基于当前 GPS）
            lon, lat = self.parse_gps_input()
            alt = self.spin_cruise_alt.value()
            self.pos_data = [[0.0, 0.0, alt]]
            
            pts = np.array(self.pos_data)
            self.plot3d.setData(pos=pts)
            self.drone_point.setData(pos=pts)
            self.update_raw_gps_text(lon, lat, alt)

            # 4. 渲染检测结果图像到左侧看板
            self.last_frame = annotated_frame.copy()
            self.render_frame(annotated_frame)
            
            # 统计目标数量
            box_count = len(results[0].boxes) if results[0].boxes is not None else 0
            QMessageBox.information(self, "图片检测完成", f"📷 照片检测成功！\n\n识别发现目标数: {box_count} 个\n文件名称: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "检测失败", f"照片推理异常:\n{str(e)}")

    def plan_auto_mission(self):
        lon, lat = self.parse_gps_input()
        field_w = self.spin_field_w.value()
        field_h = self.spin_field_h.value()
        spray_w = self.spin_spray_w.value()
        cruise_alt = self.spin_cruise_alt.value()

        planner = CoveragePathPlanner(spray_width=spray_w)
        self.planned_waypoints = planner.generate_lawnmower_path(
            start_x=0.0, start_y=0.0, 
            field_width=field_w, field_length=field_h, 
            cruise_alt=cruise_alt
        )

        pts = np.array(self.planned_waypoints)
        self.planned_plot3d.setData(pos=pts)

        max_dim = max(field_w, field_h, cruise_alt, 100.0)
        grid_size = float(max_dim * 1.5)
        step = float(grid_size / 20.0)
        
        self.grid.setSize(x=grid_size, y=grid_size, z=0)
        self.grid.setSpacing(x=step, y=step, z=10)
        
        self.view3d.setCameraPosition(distance=max_dim * 1.8, elevation=30, azimuth=45)

        axis_len = max_dim * 1.2
        self.axis_x.setData(pos=np.array([[0, 0, 0], [axis_len, 0, 0]]))
        self.axis_y.setData(pos=np.array([[0, 0, 0], [0, axis_len, 0]]))
        self.axis_z.setData(pos=np.array([[0, 0, 0], [0, 0, axis_len]]))
        self.text_x.setData(pos=(axis_len + 2, 0, 0))
        self.text_y.setData(pos=(0, axis_len + 2, 0))
        self.text_z.setData(pos=(0, 0, axis_len + 2))

        area_m2 = field_w * field_h
        area_mu = area_m2 / 666.67

        QMessageBox.information(
            self, "航线规划成功", 
            f"✅ 农田自定义全覆盖巡航航线已生成！\n\n"
            f"📍 起点 GPS: 经度 {lon:.6f}°, 纬度 {lat:.6f}°\n"
            f"📐 农田范围: {field_w:.1f} m × {field_h:.1f} m ({area_m2:,.0f} ㎡ / 约 {area_mu:,.2f} 亩)\n"
            f"📏 行距: {spray_w} 米 | 🛸 飞行高度: {cruise_alt} 米\n"
            f"🚀 巡航速度: {self.spin_speed.value()} m/s\n"
            f"📍 总航点数量: {len(self.planned_waypoints)} 个\n\n"
            f"右侧 3D 图已自动调整渲染范围，点击 [🚀 开启自动巡航] 即可飞赴作业。"
        )

    def start_auto_cruise(self):
        if not self.planned_waypoints:
            self.plan_auto_mission()

        if self.loader_thread and self.loader_thread.isRunning():
            if self.loader_thread._is_paused:
                self.loader_thread.resume()
                return
            else:
                self.stop_thread()

        base_gps = self.parse_gps_input()
        source = self.current_live_source if (self.is_live_mode and self.current_live_source is not None) else self.video_path

        self.pos_data = []
        self.is_image_mode = False
        self.session_start_time = datetime.now()
        self.loader_thread = VideoLoaderThread(
            video_source=source, 
            model_path=self.model_path, 
            is_live=self.is_live_mode, 
            auto_waypoints=self.planned_waypoints, 
            base_gps=base_gps,
            speed_m_s=self.spin_speed.value()
        )
        self.loader_thread.frame_signal.connect(self.update_image)
        self.loader_thread.trajectory_signal.connect(self.update_3d_trajectory)
        self.loader_thread.raw_gps_signal.connect(self.update_raw_gps_text)
        self.loader_thread.status_signal.connect(self.update_status)
        self.loader_thread.start()

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
                    text="rtsp://192.168.137.30:8554/live"
                )
                if ok and url.strip():
                    video_source = url.strip()

            if video_source is not None:
                self.stop_thread()
                self.is_live_mode = True
                self.is_image_mode = False
                self.current_live_source = video_source
                self.pos_data = []
                self.session_start_time = datetime.now()
                base_gps = self.parse_gps_input()
                self.loader_thread = VideoLoaderThread(
                    video_source, self.model_path, is_live=True, base_gps=base_gps, speed_m_s=self.spin_speed.value()
                )
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
        elif self.is_image_mode:
            cv2.putText(display_img, "STATIC IMAGE MODE", (35, 41), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

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
        self.pos_data.append([x_m, y_m, z_m])
        pts = np.array(self.pos_data)
        self.plot3d.setData(pos=pts)
        self.drone_point.setData(pos=np.array([[x_m, y_m, z_m]]))

    def update_raw_gps_text(self, lon, lat, alt):
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
        self.is_image_mode = False
        self.session_start_time = datetime.now()
        base_gps = self.parse_gps_input()
        
        self.loader_thread = VideoLoaderThread(
            self.video_path, self.model_path, is_live=False, base_gps=base_gps, speed_m_s=self.spin_speed.value()
        )
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
            QMessageBox.warning(self, "提示", "当前没有可保存的巡检轨迹/图片检测数据！")
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
                f"⏳ 巡检时长: {duration_str}\n"
                f"📍 采节点数: {len(self.pos_data)} 个\n\n"
                f"📁 数据库路径: {os.path.abspath(self.db_path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "数据库错误", f"保存至 SQLite 失败:\n{str(e)}")

    def clear_data(self):
        self.stop_thread()

        self.pos_data = []
        self.planned_waypoints = []
        self.last_frame = None
        self.session_start_time = None
        self.is_image_mode = False
        
        self.planned_plot3d.setData(pos=np.zeros((1, 3)))
        self.plot3d.setData(pos=np.zeros((1, 3)))
        self.drone_point.setData(pos=np.zeros((1, 3)))

        self.gps_info_label.setText("📍 真实 GPS: 经度 --° | 纬度 --° | 相对高度 -- m")
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("🍃 画面已初始化，等待启动巡检...")

        QMessageBox.information(self, "提示", "巡检状态与数据已重置！")

    def stop_thread(self):
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.stop()
            if not self.loader_thread.wait(1000):
                self.loader_thread.terminate()
                self.loader_thread.wait()
            self.loader_thread = None

    def closeEvent(self, event):
        self.stop_thread()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UAVGroundStation()
    win.show()
    sys.exit(app.exec())