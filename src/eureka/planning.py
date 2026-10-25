"""Conservative occupancy planning and capture-pose RGBD obstacle mapping."""

from __future__ import annotations

import heapq
import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree

from .domain import CameraPacket, JsonDict, MapGrid, Point, Pose, Velocity


class NoPath(ValueError):
    """The requested goal cannot be reached with the required clearance."""


def project_obstacles(
    packet: CameraPacket, calibration: JsonDict, height: float = 0.09
) -> NDArray[np.float64]:
    """Return world points within the robot's collision height band.

    Depth is optical-axis distance. The transform and pose belong to the
    capture, rather than the later time when the packet reaches the client.
    """
    depth = packet.depth_array().astype(np.float64)
    depth = depth * float(calibration["depth_scale"])
    valid = (depth >= 0.05) & (depth <= float(calibration["max_depth"]))
    v, u = np.nonzero(valid)
    z = depth[v, u]

    camera_x = (u - calibration["cx"]) * z / calibration["fx"]
    camera_y = (v - calibration["cy"]) * z / calibration["fy"]
    camera = np.column_stack((camera_x, camera_y, z))

    transform = np.asarray(calibration["T_base_camera"], dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    base = camera @ rotation.T + translation
    collision_height = (base[:, 2] >= 0.025) & (base[:, 2] <= height)
    base = base[collision_height]

    pose = packet.pose
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    world = base.copy()
    world[:, 0] = pose.x + cos_yaw * base[:, 0] - sin_yaw * base[:, 1]
    world[:, 1] = pose.y + sin_yaw * base[:, 0] + cos_yaw * base[:, 1]
    return world


class GridPlanner:
    """Own the static map, confirmed obstacles, and cached configuration space."""

    def __init__(
        self, grid: MapGrid, radius: float = 0.17, margin: float = 0.03, height: float = 0.09
    ) -> None:
        if radius <= 0 or height <= 0 or margin < 0 or grid.resolution <= 0:
            raise ValueError("Radius and resolution must be positive; margin must be nonnegative")
        if grid.occupied.ndim != 2 or not grid.occupied.size:
            raise ValueError("Occupancy map must be a nonempty two-dimensional array")
        self.grid = grid
        self.radius = radius
        self.height = height
        self.margin = margin
        self.dynamic = np.zeros(grid.occupied.shape, dtype=bool)
        self.revision = 0
        self._evidence = np.zeros(grid.occupied.shape, dtype=np.uint8)
        self._last_camera: tuple[float, int] | None = None
        self._cache_revision = -1
        self._blocked = np.empty_like(self.dynamic)
        self._tree: cKDTree | None = None
        if grid.occupied.any():
            free_cells = ~grid.occupied.astype(bool)
            self._static_distance = distance_transform_edt(free_cells) * grid.resolution
        else:
            self._static_distance = np.full(grid.occupied.shape, np.inf)

    def _refresh(self) -> None:
        if self._cache_revision == self.revision:
            return
        occupied = self.grid.occupied.astype(bool) | self.dynamic
        resolution = self.grid.resolution
        padded_free_cells = np.pad(~occupied, 1, constant_values=False)
        distance = distance_transform_edt(padded_free_cells)[1:-1, 1:-1]
        required_clearance = self.radius + self.margin + resolution / math.sqrt(2)
        self._blocked = distance * resolution <= required_clearance

        rows, cols = np.indices(occupied.shape)
        distance_from_top = (rows + 0.5) * resolution
        distance_from_left = (cols + 0.5) * resolution
        distance_from_bottom = (self.grid.height - rows - 0.5) * resolution
        distance_from_right = (self.grid.width - cols - 0.5) * resolution
        boundary = np.empty_like(distance_from_top)
        np.minimum.reduce(
            (distance_from_top, distance_from_left, distance_from_bottom, distance_from_right),
            out=boundary,
        )
        self._blocked |= boundary < required_clearance

        points = np.argwhere(occupied)
        center_x = self.grid.origin[0] + (points[:, 1] + 0.5) * resolution
        center_y = self.grid.origin[1] + (self.grid.height - points[:, 0] - 0.5) * resolution
        centers = np.column_stack((center_x, center_y))
        if len(centers):
            self._tree = cKDTree(centers)
        else:
            self._tree = None
        self._cache_revision = self.revision

    @property
    def blocked(self) -> NDArray[np.bool_]:
        self._refresh()
        return self._blocked

    def _clearance(self, point: Point) -> float:
        self._refresh()
        x, y = point
        ox, oy = self.grid.origin
        r = self.grid.resolution
        distance_from_left = x - ox
        distance_from_bottom = y - oy
        distance_from_right = ox + self.grid.width * r - x
        distance_from_top = oy + self.grid.height * r - y
        clearance = min(
            distance_from_left, distance_from_bottom, distance_from_right, distance_from_top
        )
        if self._tree is not None:
            nearest, _ = self._tree.query(point)
            indices = self._tree.query_ball_point(point, nearest + r * math.sqrt(2))
            centers = self._tree.data[indices]
            delta = np.maximum(np.abs(centers - point) - r / 2, 0)
            obstacle_distance = float(np.linalg.norm(delta, axis=1).min())
            clearance = min(clearance, obstacle_distance)
        return clearance

    def segment_clear(self, start: Point, end: Point) -> bool:
        """Check a swept disc, including a monotone escape from reduced margin."""
        for coordinate in (*start, *end):
            if not math.isfinite(coordinate):
                return False
        clearance = self._segment_clearance(start, end)
        if clearance >= self.radius + self.margin:
            return True
        if clearance < self.radius + 0.002:
            return False
        r = self.grid.resolution
        count = max(1, math.ceil(math.dist(start, end) / (r / 4)))
        points = np.linspace(start, end, count + 1)
        cells = []
        for point in points:
            cells.append(self.grid.world_to_cell(*point))
        for cell in cells:
            if not self.grid.contains(*cell):
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
        velocity: Velocity = (0.0, 0.0),
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
            if step < command_steps:
                target_v = linear
                target_w = angular
            else:
                target_v = 0.0
                target_w = 0.0

            linear_change = np.clip(target_v - v, -acceleration[0] * dt, acceleration[0] * dt)
            v += float(linear_change)
            angular_change = np.clip(target_w - w, -acceleration[1] * dt, acceleration[1] * dt)
            w += float(angular_change)
            next_x = x + v * dt * math.cos(yaw + w * dt / 2)
            next_y = y + v * dt * math.sin(yaw + w * dt / 2)
            curvature_error = abs(v * w) * dt * dt / 8
            clearance = self._segment_clearance((x, y), (next_x, next_y))
            if clearance < self.radius + 0.001 + curvature_error:
                return False
            x = next_x
            y = next_y
            yaw = yaw + w * dt
            if step >= command_steps and abs(v) < 1e-10 and abs(w) < 1e-10:
                return True
        return False

    def _segment_clearance(self, start: Point, end: Point) -> float:
        self._refresh()
        a = np.asarray(start)
        b = np.asarray(end)
        r = self.grid.resolution
        origin = np.asarray(self.grid.origin)
        upper = origin + np.array([self.grid.width, self.grid.height]) * r
        start_to_lower_edges = (a - origin).min()
        end_to_lower_edges = (b - origin).min()
        start_to_upper_edges = (upper - a).min()
        end_to_upper_edges = (upper - b).min()
        clearance = float(
            min(start_to_lower_edges, end_to_lower_edges, start_to_upper_edges, end_to_upper_edges)
        )
        if self._tree is None or clearance < 0:
            return clearance
        half_length = float(np.linalg.norm(b - a)) / 2
        midpoint = (a + b) / 2
        search_radius = half_length + self.radius + self.margin + r / math.sqrt(2)
        indices = self._tree.query_ball_point(midpoint, search_radius)
        if not indices:
            return clearance
        centers = self._tree.data[indices]
        lower = centers - r / 2
        higher = centers + r / 2
        delta = b - a
        length_squared = float(delta @ delta)

        delta_a = np.maximum(np.abs(a - centers) - r / 2, 0)
        delta_b = np.maximum(np.abs(b - centers) - r / 2, 0)
        endpoint_a = np.linalg.norm(delta_a, axis=1)
        endpoint_b = np.linalg.norm(delta_b, axis=1)
        minimum = np.minimum(endpoint_a, endpoint_b)
        if length_squared != 0:
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
                from_start_to_corners = corners - a
                projection = from_start_to_corners @ delta / length_squared
                t = np.clip(projection, 0, 1)
                closest_points = a + t[:, None] * delta
                distances = np.linalg.norm(corners - closest_points, axis=1)
                minimum = np.minimum(minimum, distances)
        return min(clearance, float(minimum.min()))

    def plan(self, start: Point, goal: Point) -> list[Point]:
        """Return safe waypoints, including the exact requested endpoint."""
        for coordinate in (*start, *goal):
            if not math.isfinite(coordinate):
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
                points = [start]
                for route_cell in reversed(route[:-1]):
                    points.append(self.grid.cell_to_world(*route_cell))
                points.append(goal)
                smoothed_points = self._smooth(points)
                return self._subdivide(smoothed_points)
            for dr, dc in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                neighbor = (cell[0] + dr, cell[1] + dc)
                if not self.grid.contains(*neighbor):
                    continue
                if blocked[neighbor] and not blocked[cell]:
                    continue
                if dr != 0 and dc != 0:
                    first_corner = (cell[0] + dr, cell[1])
                    second_corner = (cell[0], cell[1] + dc)
                    if blocked[first_corner] or blocked[second_corner]:
                        continue
                if cell == source:
                    point = start
                else:
                    point = self.grid.cell_to_world(*cell)
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

    def _smooth(self, points: list[Point]) -> list[Point]:
        result = [points[0]]
        index = 0
        while index < len(points) - 1:
            candidate = len(points) - 1
            while candidate > index + 1:
                if self.segment_clear(points[index], points[candidate]):
                    break
                candidate -= 1
            if not self.segment_clear(points[index], points[candidate]):
                raise NoPath("Grid route cannot be connected safely to the requested pose")
            result.append(points[candidate])
            index = candidate
        return result

    @staticmethod
    def _subdivide(points: list[Point]) -> list[Point]:
        result = [points[0]]
        for start, end in zip(points, points[1:]):
            count = max(1, math.ceil(math.dist(start, end) / 0.5))
            for i in range(1, count + 1):
                x = start[0] + (end[0] - start[0]) * i / count
                y = start[1] + (end[1] - start[1]) * i / count
                result.append((x, y))
        result[-1] = points[-1]
        return result

    def update_depth(self, packet: CameraPacket, calibration: JsonDict) -> int:
        """Confirm spatially supported returns observed in two distinct captures."""
        identity = (packet.t, packet.seq)
        if self._last_camera is not None and identity <= self._last_camera:
            return 0
        self._last_camera = identity
        points = project_obstacles(packet, calibration, self.height)
        if not len(points):
            return 0
        columns_from_origin = (points[:, 0] - self.grid.origin[0]) / self.grid.resolution
        rows_from_origin = (points[:, 1] - self.grid.origin[1]) / self.grid.resolution
        cols = np.floor(columns_from_origin).astype(int)
        rows = self.grid.height - 1 - np.floor(rows_from_origin).astype(int)
        valid = (rows >= 0) & (rows < self.grid.height) & (cols >= 0) & (cols < self.grid.width)
        counts = np.zeros_like(self._evidence, dtype=np.int32)
        np.add.at(counts, (rows[valid], cols[valid]), 1)
        candidates = (counts >= 2) & (self._static_distance > 0.04) & ~self.dynamic
        supported = candidates & (maximum_filter(self._evidence, size=3) >= 1)
        updated_evidence = self._evidence[candidates].astype(int) + 1
        self._evidence[candidates] = np.minimum(updated_evidence, 255)
        added = int(supported.sum())
        if added:
            self.dynamic |= supported
            self.revision += 1
        return added
