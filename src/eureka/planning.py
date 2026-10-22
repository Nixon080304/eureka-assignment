"""Conservative occupancy planning and capture-pose RGBD obstacle mapping."""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree

from eureka.domain import CameraPacket, MapGrid, Pose


class NoPath(ValueError):
    """The requested goal cannot be reached with the required clearance."""


def project_obstacles(packet: CameraPacket, calibration: dict, height: float = 0.09) -> np.ndarray:
    """Return world points within the robot's collision height band.

    Depth is optical-axis distance. The transform and pose belong to the
    capture, rather than the later time when the packet reaches the client.
    """
    depth = packet.depth_array().astype(np.float64) * float(calibration["depth_scale"])
    valid = (depth >= 0.05) & (depth <= float(calibration["max_depth"]))
    v, u = np.nonzero(valid)
    z = depth[v, u]
    camera = np.column_stack(
        (
            (u - calibration["cx"]) * z / calibration["fx"],
            (v - calibration["cy"]) * z / calibration["fy"],
            z,
        )
    )
    transform = np.asarray(calibration["T_base_camera"], dtype=float)
    base = camera @ transform[:3, :3].T + transform[:3, 3]
    base = base[(base[:, 2] >= 0.025) & (base[:, 2] <= height)]
    pose = packet.pose
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    world = base.copy()
    world[:, 0] = pose.x + c * base[:, 0] - s * base[:, 1]
    world[:, 1] = pose.y + s * base[:, 0] + c * base[:, 1]
    return world


class GridPlanner:
    """Own the static map, confirmed obstacles, and cached configuration space."""

    def __init__(
        self, grid: MapGrid, radius: float = 0.17, margin: float = 0.03, height: float = 0.09
    ):
        if radius <= 0 or height <= 0 or margin < 0 or grid.resolution <= 0:
            raise ValueError("Radius and resolution must be positive; margin must be nonnegative")
        if grid.occupied.ndim != 2 or not grid.occupied.size:
            raise ValueError("Occupancy map must be a nonempty two-dimensional array")
        self.grid = grid
        self.radius, self.height = radius, height
        self.margin = margin
        self.dynamic = np.zeros(grid.occupied.shape, dtype=bool)
        self.revision = 0
        self._evidence = np.zeros(grid.occupied.shape, dtype=np.uint8)
        self._last_camera: tuple[float, int] | None = None
        self._cache_revision = -1
        self._blocked = np.empty_like(self.dynamic)
        self._tree: cKDTree | None = None
        self._static_distance = (
            distance_transform_edt(~grid.occupied.astype(bool)) * grid.resolution
            if grid.occupied.any()
            else np.full(grid.occupied.shape, np.inf)
        )

    def _refresh(self) -> None:
        if self._cache_revision == self.revision:
            return
        occupied = self.grid.occupied.astype(bool) | self.dynamic
        r = self.grid.resolution
        distance = distance_transform_edt(np.pad(~occupied, 1, constant_values=False))[1:-1, 1:-1]
        self._blocked = distance * r <= self.radius + self.margin + r / math.sqrt(2)
        rows, cols = np.indices(occupied.shape)
        boundary = np.minimum.reduce(
            (
                (rows + 0.5) * r,
                (cols + 0.5) * r,
                (self.grid.height - rows - 0.5) * r,
                (self.grid.width - cols - 0.5) * r,
            )
        )
        self._blocked |= boundary < self.radius + self.margin + r / math.sqrt(2)
        points = np.argwhere(occupied)
        centers = np.column_stack(
            (
                self.grid.origin[0] + (points[:, 1] + 0.5) * r,
                self.grid.origin[1] + (self.grid.height - points[:, 0] - 0.5) * r,
            )
        )
        self._tree = cKDTree(centers) if len(centers) else None
        self._cache_revision = self.revision

    @property
    def blocked(self) -> np.ndarray:
        self._refresh()
        return self._blocked

    def _clearance(self, point: tuple[float, float]) -> float:
        self._refresh()
        x, y = point
        ox, oy = self.grid.origin
        r = self.grid.resolution
        clearance = min(x - ox, y - oy, ox + self.grid.width * r - x, oy + self.grid.height * r - y)
        if self._tree is not None:
            nearest, _ = self._tree.query(point)
            indices = self._tree.query_ball_point(point, nearest + r * math.sqrt(2))
            delta = np.maximum(np.abs(self._tree.data[indices] - point) - r / 2, 0)
            clearance = min(clearance, float(np.linalg.norm(delta, axis=1).min()))
        return clearance

    def segment_clear(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        """Check a swept disc, including a monotone escape from reduced margin."""
        if not all(math.isfinite(v) for v in (*start, *end)):
            return False
        clearance = self._segment_clearance(start, end)
        if clearance >= self.radius + self.margin:
            return True
        if clearance < self.radius + 0.002:
            return False
        r = self.grid.resolution
        count = max(1, math.ceil(math.dist(start, end) / (r / 4)))
        points = np.linspace(start, end, count + 1)
        cells = [self.grid.world_to_cell(*p) for p in points]
        if any(not self.grid.contains(*cell) for cell in cells):
            return False
        previous = self._clearance(start)
        if previous < self.radius + 0.002:
            return False
        escaped = False
        for point in points:
            clearance = self._clearance(tuple(point))
            if clearance < self.radius + 0.002:
                return False
            if clearance < self.radius + self.margin:
                if escaped or clearance + 1e-8 < previous:
                    return False
                if math.dist(start, point) > 0.6:
                    return False
            else:
                escaped = True
            previous = clearance
        return True

    def motion_clear(
        self,
        pose: Pose,
        linear: float,
        angular: float,
        velocity: tuple[float, float] = (0.0, 0.0),
        acceleration: tuple[float, float] = (1.0, 6.0),
        horizon: float = 0.25,
    ) -> bool:
        """Check the commanded arc and braking path from measured wheel velocity.

        Simulate the watchdog interval before braking at the published acceleration
        limits. Each chord receives an extra bound for curvature between steps.
        """
        x, y, yaw = pose.x, pose.y, pose.yaw
        v, w = velocity
        dt = 0.01
        command_steps = max(1, math.ceil(horizon / dt))
        for step in range(command_steps + 100):
            target_v, target_w = (linear, angular) if step < command_steps else (0.0, 0.0)
            v += float(np.clip(target_v - v, -acceleration[0] * dt, acceleration[0] * dt))
            w += float(np.clip(target_w - w, -acceleration[1] * dt, acceleration[1] * dt))
            next_x = x + v * dt * math.cos(yaw + w * dt / 2)
            next_y = y + v * dt * math.sin(yaw + w * dt / 2)
            curvature_error = abs(v * w) * dt * dt / 8
            if self._segment_clearance((x, y), (next_x, next_y)) < (
                self.radius + 0.001 + curvature_error
            ):
                return False
            x, y, yaw = next_x, next_y, yaw + w * dt
            if step >= command_steps and abs(v) < 1e-10 and abs(w) < 1e-10:
                return True
        return False

    def _segment_clearance(self, start: tuple[float, float], end: tuple[float, float]) -> float:
        self._refresh()
        a, b = np.asarray(start), np.asarray(end)
        r = self.grid.resolution
        origin = np.asarray(self.grid.origin)
        upper = origin + np.array([self.grid.width, self.grid.height]) * r
        clearance = float(
            min((a - origin).min(), (b - origin).min(), (upper - a).min(), (upper - b).min())
        )
        if self._tree is None or clearance < 0:
            return clearance
        half_length = float(np.linalg.norm(b - a)) / 2
        indices = self._tree.query_ball_point(
            (a + b) / 2, half_length + self.radius + self.margin + r / math.sqrt(2)
        )
        if not indices:
            return clearance
        centers = self._tree.data[indices]
        lower, higher = centers - r / 2, centers + r / 2
        delta = b - a
        length_squared = float(delta @ delta)
        endpoint_a = np.linalg.norm(np.maximum(np.abs(a - centers) - r / 2, 0), axis=1)
        endpoint_b = np.linalg.norm(np.maximum(np.abs(b - centers) - r / 2, 0), axis=1)
        minimum = np.minimum(endpoint_a, endpoint_b)
        if length_squared:
            near = np.zeros(len(centers))
            far = np.ones(len(centers))
            for axis in range(2):
                if abs(delta[axis]) < 1e-14:
                    outside = (a[axis] < lower[:, axis]) | (a[axis] > higher[:, axis])
                    near[outside] = np.inf
                else:
                    t1 = (lower[:, axis] - a[axis]) / delta[axis]
                    t2 = (higher[:, axis] - a[axis]) / delta[axis]
                    near = np.maximum(near, np.minimum(t1, t2))
                    far = np.minimum(far, np.maximum(t1, t2))
            minimum[near <= far] = 0
            for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                corners = centers + np.array([dx, dy]) * r / 2
                t = np.clip((corners - a) @ delta / length_squared, 0, 1)
                distances = np.linalg.norm(corners - (a + t[:, None] * delta), axis=1)
                minimum = np.minimum(minimum, distances)
        return min(clearance, float(minimum.min()))

    def plan(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Return safe waypoints, including the exact requested endpoint."""
        if not all(math.isfinite(v) for v in (*start, *goal)):
            raise NoPath("Start and goal must contain finite coordinates")
        source = self.grid.world_to_cell(*start)
        target = self.grid.world_to_cell(*goal)
        if not self.grid.contains(*source) or not self.grid.contains(*target):
            raise NoPath("Start or goal lies outside the map")
        blocked = self.blocked
        if blocked[target]:
            raise NoPath("Goal has insufficient obstacle clearance")
        if self._clearance(start) < self.radius + 0.002:
            raise NoPath("Start overlaps an obstacle or map boundary")
        if self.segment_clear(start, goal):
            return self._subdivide([start, goal])
        frontier = [(math.dist(source, target), 0.0, source)]
        costs = {source: 0.0}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        while frontier:
            _, cost, cell = heapq.heappop(frontier)
            if cost != costs[cell]:
                continue
            if cell == target:
                route = [cell]
                while route[-1] != source:
                    route.append(parents[route[-1]])
                points = [start] + [self.grid.cell_to_world(*c) for c in reversed(route[:-1])]
                points.append(goal)
                return self._subdivide(self._smooth(points))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                neighbor = cell[0] + dr, cell[1] + dc
                if not self.grid.contains(*neighbor):
                    continue
                if blocked[neighbor] and not blocked[cell]:
                    continue
                if dr and dc:
                    corners = (cell[0] + dr, cell[1]), (cell[0], cell[1] + dc)
                    if any(blocked[c] for c in corners):
                        continue
                point = start if cell == source else self.grid.cell_to_world(*cell)
                next_point = self.grid.cell_to_world(*neighbor)
                if blocked[cell] or blocked[neighbor]:
                    if math.dist(start, next_point) > 0.6:
                        continue
                    if not self.segment_clear(point, next_point):
                        continue
                new_cost = cost + math.hypot(dr, dc)
                if new_cost < costs.get(neighbor, math.inf):
                    costs[neighbor] = new_cost
                    parents[neighbor] = cell
                    heapq.heappush(
                        frontier, (new_cost + math.dist(neighbor, target), new_cost, neighbor)
                    )
        raise NoPath("No collision-free route connects the start and goal")

    def _smooth(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        result = [points[0]]
        index = 0
        while index < len(points) - 1:
            candidate = len(points) - 1
            while candidate > index + 1 and not self.segment_clear(
                points[index], points[candidate]
            ):
                candidate -= 1
            if not self.segment_clear(points[index], points[candidate]):
                raise NoPath("Grid route cannot be connected safely to the requested pose")
            result.append(points[candidate])
            index = candidate
        return result

    @staticmethod
    def _subdivide(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        result = [points[0]]
        for start, end in zip(points, points[1:]):
            count = max(1, math.ceil(math.dist(start, end) / 0.5))
            result.extend(
                (
                    start[0] + (end[0] - start[0]) * i / count,
                    start[1] + (end[1] - start[1]) * i / count,
                )
                for i in range(1, count + 1)
            )
        result[-1] = points[-1]
        return result

    def update_depth(self, packet: CameraPacket, calibration: dict) -> int:
        """Confirm spatially supported returns observed in two distinct captures."""
        identity = packet.t, packet.seq
        if self._last_camera is not None and identity <= self._last_camera:
            return 0
        self._last_camera = identity
        points = project_obstacles(packet, calibration, self.height)
        if not len(points):
            return 0
        cols = np.floor((points[:, 0] - self.grid.origin[0]) / self.grid.resolution).astype(int)
        rows = (
            self.grid.height
            - 1
            - np.floor((points[:, 1] - self.grid.origin[1]) / self.grid.resolution).astype(int)
        )
        valid = (rows >= 0) & (rows < self.grid.height) & (cols >= 0) & (cols < self.grid.width)
        counts = np.zeros_like(self._evidence, dtype=np.int32)
        np.add.at(counts, (rows[valid], cols[valid]), 1)
        candidates = (counts >= 2) & (self._static_distance > 0.04) & ~self.dynamic
        supported = candidates & (maximum_filter(self._evidence, size=3) >= 1)
        self._evidence[candidates] = np.minimum(self._evidence[candidates].astype(int) + 1, 255)
        added = int(supported.sum())
        if added:
            self.dynamic |= supported
            self.revision += 1
        return added
