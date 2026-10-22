"""One owner for simulator sockets, navigation state, and command timing."""

from __future__ import annotations

import hashlib
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .domain import CameraPacket, Goal, Pose, Snapshot, wrap_angle
from .planning import GridPlanner, NoPath
from .recording import Recorder
from .transport import Simulator

DEFAULT_NAVIGATION_TIMEOUT = 300.0


class PoseEstimate:
    def __init__(self, robot: dict, initial: Pose | None = None):
        self.robot = robot
        self.pose = initial
        self.ticks = None
        self.t = None
        self.velocity = (0.0, 0.0)

    def update(self, state: dict) -> Pose:
        measured = Pose.from_dict(state["pose"])
        ticks = state["wheels"]["left_ticks"], state["wheels"]["right_ticks"]
        if self.pose is None:
            self.pose = measured
        if self.ticks is not None:
            scale = 2 * math.pi * self.robot["wheel_radius"] / self.robot["ticks_per_rev"]
            left, right = ((ticks[i] - self.ticks[i]) * scale for i in (0, 1))
            distance = (left + right) / 2
            angle = (right - left) / self.robot["wheel_base"]
            elapsed = float(state.get("t", 0.0)) - self.t if self.t is not None else 0.0
            if elapsed > 0:
                self.velocity = distance / elapsed, angle / elapsed
            p = self.pose
            self.pose = Pose(
                p.x + distance * math.cos(p.yaw + angle / 2),
                p.y + distance * math.sin(p.yaw + angle / 2),
                wrap_angle(p.yaw + angle),
            )
        self.ticks = ticks
        self.t = float(state.get("t", 0.0))
        p = self.pose
        alpha = 0.035
        self.pose = Pose(
            p.x + alpha * (measured.x - p.x),
            p.y + alpha * (measured.y - p.y),
            wrap_angle(p.yaw + alpha * wrap_angle(measured.yaw - p.yaw)),
        )
        return self.pose


def goal_satisfied(pose: Pose, goal: Goal, distance: float = 0.025, angle: float = 0.04) -> bool:
    return pose.distance(goal.xy) <= distance and (
        goal.yaw is None or abs(wrap_angle(goal.yaw - pose.yaw)) <= angle
    )


def wheel_speeds(linear: float, angular: float, robot: dict) -> tuple[float, float]:
    half = robot["wheel_base"] / 2
    radius = robot["wheel_radius"]
    left, right = (linear - angular * half) / radius, (linear + angular * half) / radius
    factor = max(1.0, abs(left) / robot["max_wheel_speed"], abs(right) / robot["max_wheel_speed"])
    return left / factor, right / factor


@dataclass(frozen=True)
class Retreat:
    origin: Pose
    target: tuple[float, float]
    deadline: float


class Controller:
    def __init__(
        self,
        host: str = "localhost",
        port_base: int = 5555,
        *,
        navigation_timeout: float = DEFAULT_NAVIGATION_TIMEOUT,
    ):
        if not math.isfinite(navigation_timeout) or navigation_timeout <= 0:
            raise ValueError("Navigation timeout must be finite and positive")
        self.navigation_timeout = navigation_timeout
        self.host, self.port_base = host, port_base
        self._commands = queue.Queue(maxsize=32)
        self._shutdown = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._thread = None
        self.sim = None
        self.recorder = None
        self.connected = False
        self.mode, self.detail = "disconnected", "Connect to the simulator to begin."
        self.scene, self.seed, self.epoch = "", None, 0
        self.grid, self.planner, self.pose, self.camera = None, None, None, None
        self.robot, self.calibration, self.map_info = {}, {}, {}
        self.state, self.goals, self.marked = {}, (), set()
        self._mark_attempts = set()
        self.active_goal = None
        self.path, self.trajectory = [], deque(maxlen=6000)
        self._samples = deque(maxlen=40)
        self._pose_history = deque(maxlen=250)
        self._state_at, self._camera_at, self._depth_at = 0.0, 0.0, 0.0
        self._sim_clock_offset = None
        self._control_at, self._publish_at, self._trajectory_at = 0.0, 0.0, 0.0
        self._scan_angle, self._scanned = 0.0, False
        self._verify_at = None
        self._retreat: Retreat | None = None
        self._progress_at, self._best_distance = 0.0, math.inf
        self._nav_at = 0.0
        self._plan_at = 0.0
        self._bumps, self._replans, self._recoveries = 0, 0, 0
        self._command_failures, self._command_drops = 0, 0
        self._bumper_before = False
        self._estimator = None
        self.events = deque(maxlen=500)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="eureka-control", daemon=True)
            self._thread.start()

    def submit(self, action: str, **kwargs) -> None:
        if action == "stop":
            self._stop.set()
            while True:
                try:
                    self._commands.get_nowait()
                except queue.Empty:
                    break
        try:
            self._commands.put_nowait((action, kwargs))
        except queue.Full as exc:
            self._stop.set()
            raise RuntimeError("Command queue is full; robot stop requested") from exc

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def close(self, timeout: float | None = None) -> None:
        self._stop.set()
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("Motors stopped; recording finalization is still running")

    def _event(self, event: str, **data) -> None:
        self.events.append({"event": event, "t": self.state.get("t"), **data})

    def _zero(self) -> None:
        if self.sim is not None:
            if self.sim.wheels():
                self._command_failures = 0
            self._control_at = time.monotonic()

    def _halt(self) -> None:
        self._zero()
        until = time.monotonic() + 0.28
        while time.monotonic() < until:
            self._zero()
            time.sleep(0.025)

    def _run(self) -> None:
        self.sim = Simulator(self.host, self.port_base)
        self.recorder = Recorder()
        try:
            while not self._shutdown.is_set():
                try:
                    if self._stop.is_set():
                        self._cancel("Stopped by operator.")
                        self._stop.clear()
                    try:
                        action, kwargs = self._commands.get_nowait()
                    except queue.Empty:
                        action = None
                    if action:
                        self._action(action, kwargs)
                    if self.connected:
                        self._receive()
                    now = time.monotonic()
                    if now - self._control_at >= 0.05:
                        self._control_at = now
                        self._control(now)
                    if now - self._publish_at >= 0.1:
                        self._publish_at = now
                        self._publish()
                except Exception as exc:
                    self._zero()
                    self.mode, self.detail = "fault", f"{type(exc).__name__}: {exc}"
                    self._event("fault", error=str(exc))
                    self._publish()
                self._shutdown.wait(0.004)
        finally:
            self._halt()
            self.connected = False
            self.sim.close()
            self.recorder.stop()
            self.mode, self.detail = "disconnected", "Disconnected. Motors stopped."
            self._publish()

    def _action(self, action: str, args: dict) -> None:
        if action in ("connect", "load_scene"):
            self._bootstrap(args if action == "load_scene" else None)
        elif action == "navigate":
            if not self.connected or self.pose is None:
                raise RuntimeError("Wait for a connected simulator and fresh pose")
            goal = next((g for g in self.goals if g.name == args["name"]), None)
            if goal is None:
                raise ValueError("Unknown named goal")
            self._cancel("Preparing navigation.")
            self.active_goal = goal
            self._recoveries = 0
            self._nav_at = self._progress_at = time.monotonic()
            self._best_distance = self.pose.distance(goal.xy)
            if goal_satisfied(self.pose, goal):
                self.mode, self.detail = "verifying", "Holding position to verify goal tolerance."
                self._verify_at = None
            elif not self._scanned:
                self.mode, self.detail = "scanning", "Scanning for obstacles missing from the map."
                self._scan_angle = 0.0
            else:
                self._plan()
        elif action == "stop":
            self._cancel("Stopped by operator.")
        elif action == "start_recording":
            if not self.connected or self.grid is None:
                raise RuntimeError("Connect before recording")
            self.recorder.start(
                Path(args.get("directory", "recordings")),
                {
                    "source_sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path(__file__).parent.glob("*.py"))
                    },
                    "scene": self.scene,
                    "seed": self.seed,
                    "epoch": self.epoch,
                    "robot": self.robot,
                    "camera": self.calibration,
                    "map": self.map_info,
                    "goals": [g.__dict__ for g in self.goals],
                },
            )
        elif action == "stop_recording":
            self._cancel("Stopped while finalizing camera recording.")
            self._halt()
            self.recorder.stop()
        elif action == "disconnect":
            self._cancel("Disconnected.")
            self._halt()
            self.recorder.stop()
            self.connected = False
            self.mode = "disconnected"
        else:
            raise ValueError(f"Unknown application command: {action}")

    def _cancel(self, detail: str) -> None:
        self._zero()
        self.mode = "idle" if self.connected else "disconnected"
        self.detail = detail
        self.active_goal, self.path, self._verify_at = None, [], None

    def _bootstrap(self, load: dict | None) -> None:
        self.mode, self.detail = "loading", "Stopping motors and loading simulator metadata."
        self._publish()
        self._halt()
        self.connected = False
        self.recorder.stop()
        self.marked.clear()
        self._mark_attempts.clear()
        initial = None
        self.seed = None
        if load is not None:
            result, _ = self.sim.call(
                "load_scene",
                timeout=5000,
                id=str(load.get("scene", "01")),
                seed=int(load.get("seed", 42)),
            )
            self.scene, self.seed = str(result["id"]), int(load.get("seed", 42))
            initial = Pose.from_dict(result["start"])
            self.marked.clear()
            self._mark_attempts.clear()
        self.robot, _ = self.sim.call("get_robot_info")
        self.calibration, _ = self.sim.call("get_camera_info")
        values, _ = self.sim.call("get_goals")
        self.goals = tuple(Goal.from_dict(g) for g in values)
        self.grid, self.map_info, self._map_png = self.sim.get_map()
        self.planner = GridPlanner(
            self.grid, radius=float(self.robot["radius"]), height=float(self.robot["height"])
        )
        self.sim.restart_streams()
        self._estimator = PoseEstimate(self.robot, initial)
        self.epoch += 1
        self.state, self.pose, self.camera = {}, None, None
        self.trajectory.clear()
        self._samples.clear()
        self._pose_history.clear()
        self.path, self.active_goal = [], None
        self._state_at = self._camera_at = self._depth_at = 0.0
        self._sim_clock_offset = None
        self._scanned, self._bumper_before = False, False
        self._bumps = self._replans = self._recoveries = 0
        self.connected = True
        self.mode, self.detail = "idle", "Ready. Choose a named goal."
        self._event("episode", scene=self.scene, epoch=self.epoch, seed=self.seed)

    def _receive_states(self) -> bool:
        states = self.sim.states()
        for state in states:
            if self.state and (
                state["scene"] != self.state["scene"]
                or (
                    state["t"] < self.state["t"] - 0.25
                    and (state["seq"] > self.state["seq"] or state["seq"] < self.state["seq"] - 10)
                )
            ):
                self._halt()
                self.marked.clear()
                self._mark_attempts.clear()
                self._bootstrap(None)
                self.detail = "External scene reset detected. Navigation stopped."
                return False
            if self.state and state["seq"] <= self.state["seq"]:
                continue
            old_pose = self.pose
            self.state = state
            self.scene = str(state["scene"])
            self.pose = self._estimator.update(state)
            self._pose_history.append((float(state["t"]), self.pose))
            now = time.monotonic()
            offset = now - float(state["t"])
            self._sim_clock_offset = (
                offset if self._sim_clock_offset is None else min(self._sim_clock_offset, offset)
            )
            self._state_at = min(now, float(state["t"]) + self._sim_clock_offset)
            self._samples.append((state["t"], Pose.from_dict(state["pose"])))
            self.recorder.offer_state(state)
            if self.mode == "scanning" and old_pose is not None:
                self._scan_angle += abs(wrap_angle(self.pose.yaw - old_pose.yaw))
            if self._state_at - self._trajectory_at > 0.1:
                self._trajectory_at = self._state_at
                self.trajectory.append((self.pose.x, self.pose.y))
            if state["bumper"] and not self._bumper_before:
                self._bumps += 1
                self._event("bumper", pose=self.pose.as_dict())
                if self.mode in ("following", "aligning", "verifying"):
                    self._recover()
                elif self.mode == "recovery":
                    self._zero()
                    self._retreat = None
                    self.mode, self.detail = (
                        "scanning",
                        "New contact during retreat. Scanning without translation.",
                    )
                    self._scan_angle = 0.0
            self._bumper_before = bool(state["bumper"])
        return True

    def _receive(self) -> None:
        if not self._receive_states():
            return
        packets = self.sim.cameras()
        for packet in packets:
            self.recorder.offer(packet)
            if self.state and packet.t < self.state["t"] - 1.0:
                continue
            if self.camera is not None and packet.seq <= self.camera.seq:
                continue
            self.camera = packet
            now = time.monotonic()
            offset = now - packet.t
            self._sim_clock_offset = (
                offset if self._sim_clock_offset is None else min(self._sim_clock_offset, offset)
            )
            self._camera_at = min(now, packet.t + self._sim_clock_offset)
        if packets and self.camera is not None and time.monotonic() - self._depth_at > 0.2:
            self._depth_at = time.monotonic()
            pose = self._capture_pose(self.camera.t, self.camera.pose)
            mapping_packet = CameraPacket(
                {**self.camera.header, "pose": pose.as_dict()},
                self.camera.jpeg,
                self.camera.depth,
                self.camera.received_ns,
            )
            added = self.planner.update_depth(mapping_packet, self.calibration)
            if added and self.mode == "following" and self.path:
                if not self.planner.segment_clear((self.pose.x, self.pose.y), self.path[0]):
                    self._plan()
        self._receive_states()

    def _capture_pose(self, capture_time: float, fallback: Pose) -> Pose:
        history = self._pose_history
        if not history or capture_time < history[0][0] or capture_time > history[-1][0] + 0.04:
            return fallback
        for (t0, p0), (t1, p1) in zip(history, list(history)[1:]):
            if t0 <= capture_time <= t1:
                ratio = (capture_time - t0) / (t1 - t0) if t1 > t0 else 0.0
                return Pose(
                    p0.x + ratio * (p1.x - p0.x),
                    p0.y + ratio * (p1.y - p0.y),
                    wrap_angle(p0.yaw + ratio * wrap_angle(p1.yaw - p0.yaw)),
                )
        return history[-1][1]

    def _plan(self) -> None:
        if self.pose is None or self.active_goal is None:
            return
        self.mode, self.detail = "planning", "Planning a path with robot clearance."
        self._publish()
        self._halt()
        self._receive()
        if self._stop.is_set() or self._shutdown.is_set() or self.active_goal is None:
            self._cancel("Stopped before planning.")
            return
        if self.state.get("bumper"):
            self._recover()
            return
        try:
            route = self.planner.plan((self.pose.x, self.pose.y), self.active_goal.xy)
        except NoPath as exc:
            self.mode, self.detail = "blocked", f"No safe path: {exc}"
            self.path = []
            self._event(
                "blocked",
                reason=str(exc),
                pose=self.pose.as_dict(),
                dynamic_cells=int(self.planner.dynamic.sum()),
            )
            return
        self.path = list(route[1:] if len(route) > 1 else route)
        self._replans += 1
        self._progress_at = time.monotonic()
        self._control_at = self._progress_at
        self._plan_at = self._progress_at
        self._best_distance = self.pose.distance(self.active_goal.xy)
        self.mode, self.detail = "waiting", "Waiting for fresh streams after planning."
        self._event(
            "path",
            goal=self.active_goal.name,
            pose=self.pose.as_dict(),
            points=list(self.path),
            dynamic_cells=int(self.planner.dynamic.sum()),
        )

    def _recover(self) -> None:
        self._zero()
        self._recoveries += 1
        self._retreat = None
        if self._recoveries > 3:
            self.mode, self.detail = "blocked", "Repeated contact. Stopped for operator review."
            return
        target = (
            self.pose.x - 0.06 * math.cos(self.pose.yaw),
            self.pose.y - 0.06 * math.sin(self.pose.yaw),
        )
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        visited = any(
            0.065 <= -(x - self.pose.x) * c - (y - self.pose.y) * s <= 0.15
            and abs(-(x - self.pose.x) * s + (y - self.pose.y) * c) <= 0.01
            for x, y in list(self.trajectory)[-150:]
        )
        if not visited or not self.planner.segment_clear((self.pose.x, self.pose.y), target):
            self.mode, self.detail = (
                "scanning",
                "Contact bearing unknown. Scanning without translation.",
            )
            self._scan_angle = 0.0
            self._scanned = False
            return
        self._retreat = Retreat(self.pose, target, time.monotonic() + 2.0)
        self.mode, self.detail = "recovery", "Retreating at most 6 cm on a checked segment."

    def _motion_clear(self, linear: float, angular: float) -> bool:
        velocity = self._estimator.velocity if self._estimator is not None else (0.0, 0.0)
        return self.planner.motion_clear(
            self.pose,
            linear,
            angular,
            velocity,
            (self.robot.get("max_linear_accel", 1.0), self.robot.get("max_angular_accel", 6.0)),
            self.robot.get("watchdog", 0.25) + max(0.0, time.monotonic() - self._state_at),
        )

    def _control(self, now: float) -> None:
        moving = self.mode in (
            "scanning",
            "waiting",
            "following",
            "aligning",
            "verifying",
            "recovery",
        )
        if not self.connected:
            return
        if not moving:
            self._zero()
            return
        if self._stop.is_set() or self._shutdown.is_set():
            self._cancel("Stopped by operator.")
            return
        if now - self._state_at > 0.20:
            self._cancel("State stream is stale. Select the goal again after reconnection.")
            self.mode = "fault"
            return
        if self.active_goal is not None and now - self._nav_at > self.navigation_timeout:
            self._cancel(f"Navigation exceeded {self.navigation_timeout:g} seconds.")
            self.mode = "blocked"
            return
        if self.mode == "waiting":
            self._zero()
            if (
                self._state_at >= self._plan_at
                and self._camera_at >= self._plan_at
                and now - self._state_at <= 0.10
                and now - self._camera_at <= 0.25
            ):
                self.mode = "following"
                self.detail = f"Following path to {self.active_goal.name}."
            elif now - self._plan_at > 3:
                self._cancel("Fresh streams did not arrive after planning. Motors remain stopped.")
                self.mode = "fault"
            return
        if self.mode == "scanning":
            if now - self._camera_at > 0.7:
                self._zero()
                self.detail = "Waiting for a fresh RGBD camera frame."
                return
            if self._scan_angle >= 2 * math.pi - 0.12:
                self._zero()
                self._scanned = True
                self._plan()
                return
            self._drive(0.0, 1.0)
            return
        if self.mode == "recovery":
            retreat = self._retreat
            progress = (
                0.0
                if retreat is None
                else (
                    (retreat.origin.x - self.pose.x) * math.cos(retreat.origin.yaw)
                    + (retreat.origin.y - self.pose.y) * math.sin(retreat.origin.yaw)
                )
            )
            velocity = self._estimator.velocity if self._estimator is not None else (0.0, 0.0)
            reverse_speed = max(0.04, -velocity[0])
            observation_age = max(0.0, now - self._state_at)
            stopping_distance = reverse_speed * (0.05 + observation_age) + reverse_speed**2 / (
                2 * self.robot.get("max_linear_accel", 1.0)
            )
            if retreat is None or progress + stopping_distance >= 0.06 or now >= retreat.deadline:
                self._zero()
                self._retreat = None
                self._plan()
            elif (
                not self.planner.segment_clear((self.pose.x, self.pose.y), retreat.target)
                or abs(wrap_angle(self.pose.yaw - retreat.origin.yaw)) > 0.03
                or not self._motion_clear(-0.04, 0.0)
            ):
                self._zero()
                self._retreat = None
                self.mode, self.detail = (
                    "scanning",
                    "Retreat no longer clear. Scanning without translation.",
                )
                self._scan_angle = 0.0
            else:
                self._drive(-0.04, 0.0)
            return
        goal = self.active_goal
        if goal is None:
            self._zero()
            return
        distance = self.pose.distance(goal.xy)
        if distance < self._best_distance - 0.025:
            self._best_distance, self._progress_at = distance, now
        if now - self._progress_at > 15 and self.mode == "following":
            self._plan()
            return
        if distance <= 0.020:
            error = 0.0 if goal.yaw is None else wrap_angle(goal.yaw - self.pose.yaw)
            if abs(error) > 0.030:
                self.mode, self.detail = "aligning", "Aligning final goal heading."
                self._drive(0.0, float(np.clip(2.2 * error, -0.8, 0.8)))
            else:
                self.mode, self.detail = "verifying", "Verifying stable position and heading."
                self._zero()
                self._verify(now)
            return
        self._verify_at = None
        self.mode = "following"
        self.detail = f"Following path to {goal.name}."
        if now - self._camera_at > 0.55:
            self._cancel("Camera stream is stale. Stopped before moving into unseen space.")
            self.mode = "fault"
            return
        while len(self.path) > 1 and self.pose.distance(self.path[0]) < 0.10:
            if not self.planner.segment_clear((self.pose.x, self.pose.y), self.path[1]):
                if not self.planner.segment_clear(self.path[0], self.path[1]):
                    self._zero()
                    self._plan()
                    return
                break
            self.path.pop(0)
        if not self.path:
            self._plan()
            return
        target = self.path[0]
        if not self.planner.segment_clear((self.pose.x, self.pose.y), target):
            self._zero()
            self._plan()
            return
        error = wrap_angle(
            math.atan2(target[1] - self.pose.y, target[0] - self.pose.x) - self.pose.yaw
        )
        angular = float(np.clip(2.6 * error, -1.3, 1.3))
        linear = min(0.20, 0.9 * self.pose.distance(target)) * max(0.0, math.cos(error))
        if abs(error) > 0.60:
            linear = 0.0
        if self.planner._clearance((self.pose.x, self.pose.y)) < (
            self.planner.radius + self.planner.margin + 0.005
        ):
            if abs(error) > 0.015:
                linear = 0.0
            else:
                angular = 0.0
                linear = min(linear, 0.06)
        if not self._motion_clear(linear, angular):
            self._zero()
            self.detail = "Stopping before a curved or braking trajectory leaves free space."
            return
        self._drive(linear, angular)

    def _drive(self, linear: float, angular: float) -> None:
        if self._stop.is_set() or self._shutdown.is_set():
            self._zero()
            return
        recovering = self._command_failures > 0
        speeds = (0.0, 0.0) if recovering else wheel_speeds(linear, angular, self.robot)
        if self.sim.wheels(*speeds):
            self._command_failures = 0
            return
        self._command_failures += 1
        self._command_drops += 1
        self.detail = "Command transport is busy. Sending stop before resuming."
        if self._command_failures >= 4:
            self.mode = "fault"
            self.detail = "Command transport unavailable. Motor watchdog will stop motion."

    def _verify(self, now: float) -> None:
        if not goal_satisfied(self.pose, self.active_goal):
            self._verify_at = None
            return
        if self._verify_at is None:
            self._verify_at = now
            return
        if now - self._verify_at < 0.6 or len(self._samples) < 15:
            return
        recent = [p for t, p in self._samples if t > self.state["t"] - 0.5]
        if len(recent) < 15:
            return
        mean = Pose(
            float(np.mean([p.x for p in recent])),
            float(np.mean([p.y for p in recent])),
            math.atan2(sum(math.sin(p.yaw) for p in recent), sum(math.cos(p.yaw) for p in recent)),
        )
        if not goal_satisfied(mean, self.active_goal, 0.030, 0.05):
            return
        name = self.active_goal.name
        self._halt()
        if name not in self.marked:
            if name in self._mark_attempts:
                self.mode = "fault"
                self.detail = "Previous goal mark outcome is unknown. Reset the scene to retry."
                return
            self._mark_attempts.add(name)
            self.detail = "Goal mark requested. A timeout leaves its outcome unknown."
            result, _ = self.sim.call("mark_goal", name=name)
            self.marked.add(name)
            self._event(
                "goal_marked",
                name=name,
                mark_t=result["t"],
                estimated_pose=self.pose.as_dict(),
                mean_pose=mean.as_dict(),
                position_error=self.pose.distance(self.active_goal.xy),
            )
        self.mode, self.detail = "arrived", f"Reached and marked {name}."
        self.path = []

    def _publish(self) -> None:
        now = time.monotonic()
        dynamic = None
        if self.planner is not None:
            dynamic = self.planner.dynamic.copy()
            dynamic.flags.writeable = False
        record = self.recorder.status() if self.recorder is not None else {}
        value = Snapshot(
            connected=self.connected,
            scene=self.scene,
            epoch=self.epoch,
            seed=self.seed,
            mode=self.mode,
            detail=self.detail,
            pose=self.pose,
            state=dict(self.state),
            grid=self.grid,
            dynamic=dynamic,
            path=tuple(self.path),
            trajectory=tuple(self.trajectory),
            goals=self.goals,
            active_goal=self.active_goal,
            marked=tuple(sorted(self.marked)),
            camera=self.camera,
            recording=record,
            stats={
                "state_age": now - self._state_at,
                "camera_age": now - self._camera_at,
                "bumps": self._bumps,
                "replans": self._replans,
                "command_drops": self._command_drops,
                "goals_reached": len(self.marked),
                "state_seq": self.state.get("seq"),
                "camera_seq": self.camera.seq if self.camera else None,
                "sim_time": self.state.get("t", 0),
                "epoch": self.epoch,
            },
        )
        with self._lock:
            self._snapshot = value
