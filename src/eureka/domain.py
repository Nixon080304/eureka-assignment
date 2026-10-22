"""Values shared by navigation, transport, recording, and presentation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    @classmethod
    def from_dict(cls, value):
        pose = cls(float(value["x"]), float(value["y"]), float(value["yaw"]))
        if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)):
            raise ValueError("Pose contains non-finite values")
        return pose

    def as_dict(self):
        return {"x": self.x, "y": self.y, "yaw": self.yaw}

    def distance(self, xy):
        return math.hypot(self.x - xy[0], self.y - xy[1])


@dataclass(frozen=True)
class Goal:
    name: str
    x: float
    y: float
    yaw: float | None = None

    @classmethod
    def from_dict(cls, value):
        p = value["pose"]
        if len(p) not in (2, 3) or not all(math.isfinite(float(v)) for v in p):
            raise ValueError("Invalid goal pose")
        return cls(
            str(value["name"]), float(p[0]), float(p[1]), float(p[2]) if len(p) == 3 else None
        )

    @property
    def xy(self):
        return self.x, self.y


@dataclass(frozen=True)
class MapGrid:
    occupied: np.ndarray
    resolution: float
    origin: tuple[float, float]

    @property
    def width(self):
        return self.occupied.shape[1]

    @property
    def height(self):
        return self.occupied.shape[0]

    def world_to_cell(self, x, y):
        return (
            self.height - 1 - math.floor((y - self.origin[1]) / self.resolution),
            math.floor((x - self.origin[0]) / self.resolution),
        )

    def cell_to_world(self, row, col):
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (self.height - row - 0.5) * self.resolution,
        )

    def contains(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width


@dataclass(frozen=True)
class CameraPacket:
    header: dict[str, Any]
    jpeg: bytes
    depth: bytes
    received_ns: int

    @property
    def pose(self):
        return Pose.from_dict(self.header["pose"])

    @property
    def t(self):
        return float(self.header["t"])

    @property
    def seq(self):
        return int(self.header["seq"])

    def depth_array(self):
        return np.frombuffer(self.depth, dtype="<u2").reshape(
            int(self.header["height"]), int(self.header["width"])
        )


@dataclass(frozen=True)
class Snapshot:
    connected: bool = False
    scene: str = ""
    epoch: int = 0
    seed: int | None = None
    mode: str = "disconnected"
    detail: str = "Connect to the simulator to begin."
    pose: Pose | None = None
    state: dict = field(default_factory=dict)
    grid: MapGrid | None = None
    dynamic: np.ndarray | None = None
    path: tuple[tuple[float, float], ...] = ()
    trajectory: tuple[tuple[float, float], ...] = ()
    goals: tuple[Goal, ...] = ()
    active_goal: Goal | None = None
    marked: tuple[str, ...] = ()
    camera: CameraPacket | None = None
    recording: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
