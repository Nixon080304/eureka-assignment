"""Minimal Qt controls, map, and RGBD previews."""

import math

import numpy as np
from PySide6 import QtWidgets as widgets
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QImage, QPainterPath, QPen, QPixmap


class MainWindow(widgets.QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.goal_key = None
        self.setWindowTitle("Eureka - minimal UI")
        self.resize(1100, 720)
        self.setStyleSheet("QWidget {background:#17232d; color:#eeeeee;}")
        root = widgets.QWidget()
        self.setCentralWidget(root)
        layout = widgets.QVBoxLayout(root)
        controls = widgets.QHBoxLayout()
        self.scene = widgets.QComboBox()
        self.scene.addItems(["01", "02", "03", "04", "05", "06"])
        self.seed = widgets.QSpinBox()
        self.seed.setRange(-2147483648, 2147483647)
        self.seed.setPrefix("Seed ")
        self.goals = widgets.QComboBox()
        self.load_button = widgets.QPushButton("Load scene")
        self.navigate_button = widgets.QPushButton("Navigate")
        self.stop_button = widgets.QPushButton("STOP")
        self.load_button.clicked.connect(self.load_scene)
        self.navigate_button.clicked.connect(self.navigate)
        self.stop_button.clicked.connect(self.stop_robot)
        for widget in (self.scene, self.seed, self.goals):
            controls.addWidget(widget)
        for button in (self.load_button, self.navigate_button, self.stop_button):
            controls.addWidget(button)
        layout.addLayout(controls)
        body = widgets.QHBoxLayout()
        self.map_scene = widgets.QGraphicsScene(self)
        self.map_canvas = widgets.QGraphicsView(self.map_scene)
        body.addWidget(self.map_canvas, 2)
        cameras = widgets.QVBoxLayout()
        self.rgb = widgets.QLabel("RGB")
        self.depth = widgets.QLabel("Depth: near bright, far dark")
        for label in (self.rgb, self.depth):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(240, 160)
            label.setSizePolicy(widgets.QSizePolicy.Ignored, widgets.QSizePolicy.Ignored)
            cameras.addWidget(label)
        body.addLayout(cameras, 1)
        layout.addLayout(body, 1)
        legend = "Grey walls | Orange obstacles | Blue trail | Teal path | Gold goals | y up"
        layout.addWidget(widgets.QLabel(legend))
        recording = widgets.QHBoxLayout()
        self.folder = widgets.QLineEdit("recordings")
        self.record_button = widgets.QPushButton("Start recording")
        self.record_button.clicked.connect(self.record)
        for widget in (self.folder, self.record_button):
            recording.addWidget(widget)
        layout.addLayout(recording)
        self.record_info = widgets.QLabel()
        self.record_info.setWordWrap(True)
        self.detail = widgets.QLabel()
        self.detail.setWordWrap(True)
        layout.addWidget(self.record_info)
        layout.addWidget(self.detail)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.refresh()

    def submit(self, action, **kwargs):
        try:
            self.controller.submit(action, **kwargs)
        except Exception as error:
            self.detail.setText(str(error))

    def load_scene(self):
        self.submit("load_scene", scene=self.scene.currentText(), seed=self.seed.value())

    def navigate(self):
        self.submit("navigate", name=self.goals.currentData())

    def stop_robot(self):
        self.submit("stop")

    def record(self):
        if self.controller.snapshot().recording.get("active"):
            self.submit("stop_recording")
        else:
            self.submit("start_recording", directory=self.folder.text().strip() or "recordings")

    def world_point(self, x, y):
        grid = self.snapshot.grid
        return QPointF(
            (x - grid.origin[0]) / grid.resolution,
            grid.height - (y - grid.origin[1]) / grid.resolution,
        )

    def draw_map(self, snapshot):
        self.map_scene.clear()
        grid = snapshot.grid
        if grid is None:
            return
        pixels = np.empty((*grid.occupied.shape, 3), dtype=np.uint8)
        pixels[:] = (22, 45, 54)
        pixels[grid.occupied.astype(bool)] = (147, 170, 179)
        if snapshot.dynamic is not None:
            pixels[snapshot.dynamic] = (255, 156, 85)
        image = QImage(
            pixels.data, grid.width, grid.height, pixels.strides[0], QImage.Format_RGB888
        ).copy()
        self.map_scene.addPixmap(QPixmap.fromImage(image))
        for points, color in ((snapshot.trajectory, "#679ac6"), (snapshot.path, "#52d8c6")):
            if points:
                path = QPainterPath(self.world_point(*points[0]))
                for point in points[1:]:
                    path.lineTo(self.world_point(*point))
                self.map_scene.addPath(path, QPen(QColor(color), 1.5))
        for goal in snapshot.goals:
            point = self.world_point(goal.x, goal.y)
            color = QColor("#52d8c6" if goal.name in snapshot.marked else "#f3cc76")
            self.map_scene.addEllipse(point.x() - 3, point.y() - 3, 6, 6, QPen(color, 1.5))
            label = self.map_scene.addSimpleText(goal.name)
            label.setBrush(color)
            label.setPos(point.x() + 4, point.y() - 15)
        if snapshot.pose:
            pose = snapshot.pose
            point = self.world_point(pose.x, pose.y)
            radius = 0.17 / grid.resolution
            bounds = QRectF(-radius, -radius, radius * 2, radius * 2)
            bounds.moveCenter(point)
            self.map_scene.addEllipse(bounds, QPen(Qt.white), QBrush(QColor("#258d87")))
            heading = point + QPointF(math.cos(pose.yaw), -math.sin(pose.yaw)) * radius * 1.5
            self.map_scene.addLine(QLineF(point, heading), QPen(Qt.white))
        self.map_scene.setSceneRect(0, 0, grid.width, grid.height)
        self.map_canvas.fitInView(self.map_scene.sceneRect(), Qt.KeepAspectRatio)

    def refresh(self):
        snapshot = self.controller.snapshot()
        self.snapshot = snapshot
        key = (snapshot.epoch, snapshot.goals, snapshot.marked)
        if key != self.goal_key:
            previous = (
                self.goals.currentData() if self.goal_key and key[0] == self.goal_key[0] else None
            )
            self.goals.clear()
            for goal in snapshot.goals:
                self.goals.addItem(
                    goal.name + (" ✓" if goal.name in snapshot.marked else ""), goal.name
                )
            self.goals.setCurrentIndex(max(0, self.goals.findData(previous)))
            if self.goal_key is None or snapshot.epoch != self.goal_key[0]:
                self.scene.setCurrentText(snapshot.scene)
            self.goal_key = key
        ready = snapshot.connected and snapshot.mode not in ("loading", "fault")
        self.load_button.setEnabled(snapshot.mode != "loading")
        self.navigate_button.setEnabled(ready and bool(snapshot.goals))
        self.stop_button.setEnabled(snapshot.connected)
        self.draw_map(snapshot)
        recording = snapshot.recording
        active = bool(recording.get("active"))
        self.record_button.setText("Stop recording" if active else "Start recording")
        self.record_button.setEnabled(snapshot.connected or active)
        self.folder.setEnabled(not active)
        self.record_info.setText(
            f"Frames: {recording.get('frames', 0)}  Dropped: {recording.get('dropped', 0)}  "
            f"{recording.get('error') or recording.get('path') or ''}"
        )
        pose = snapshot.pose
        position = (
            f"({pose.x:.2f}, {pose.y:.2f}) m, {math.degrees(pose.yaw):.1f}°" if pose else "No pose"
        )
        self.detail.setText(
            f"Scene {snapshot.scene}, seed {snapshot.seed} | {snapshot.mode}\n"
            f"{position} | {snapshot.detail}"
        )
        packet = snapshot.camera
        self.rgb.clear()
        self.depth.clear()
        if packet:
            depth = packet.depth_array()
            gray = (255 - np.clip(depth / 6000 * 255, 0, 255)).astype(np.uint8)
            gray[depth == 0] = 0
            height, width = gray.shape
            image = QImage(gray.data, width, height, width, QImage.Format_Grayscale8).copy()
            for label, frame in ((self.rgb, QImage.fromData(packet.jpeg)), (self.depth, image)):
                label.setPixmap(QPixmap.fromImage(frame).scaled(label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.timer.stop()
        self.controller.close()
        event.accept()
