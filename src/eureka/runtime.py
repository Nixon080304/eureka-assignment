"""Simulator communication, sensor processing, recording, and GUI snapshots."""

from __future__ import annotations

import hashlib
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .controller import DEFAULT_NAVIGATION_TIMEOUT, Controller, Decision, Mode, Observation
from .domain import (
    CameraPacket,
    Goal,
    JsonDict,
    MapGrid,
    Point,
    Pose,
    RobotInfo,
    Snapshot,
    Velocity,
    wrap_angle,
)
from .planning import GridPlanner, NoPath
from .recording import Recorder
from .transport import Simulator


class PoseEstimate:
    def __init__(self, robot: RobotInfo, initial: Pose | None = None) -> None:
        self.robot = robot
        self.pose = initial
        self.ticks: tuple[int, int] | None = None
        self.t: float | None = None
        self.velocity = (0.0, 0.0)

    def update(self, state: JsonDict) -> Pose:
        measured = Pose.from_dict(state["pose"])
        ticks = state["wheels"]["left_ticks"], state["wheels"]["right_ticks"]
        if self.pose is None:
            self.pose = measured
        if self.ticks is not None:
            scale = 2 * math.pi * self.robot["wheel_radius"] / self.robot["ticks_per_rev"]
            left = (ticks[0] - self.ticks[0]) * scale
            right = (ticks[1] - self.ticks[1]) * scale
            distance = (left + right) / 2
            angle = (right - left) / self.robot["wheel_base"]
            if self.t is not None:
                elapsed = float(state.get("t", 0.0)) - self.t
            else:
                elapsed = 0.0
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


class Runtime:
    def __init__(
        self,
        host: str = "localhost",
        port_base: int = 5555,
        *,
        navigation_timeout: float = DEFAULT_NAVIGATION_TIMEOUT,
    ) -> None:
        self.control = Controller(navigation_timeout=navigation_timeout)
        self.host, self.port_base = host, port_base
        self._commands: queue.Queue[tuple[str, JsonDict]] = queue.Queue(maxsize=32)
        self._shutdown = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._thread: threading.Thread | None = None
        self.sim: Simulator | None = None
        self.recorder: Recorder | None = None
        self.connected = False
        self.scene = ""
        self.seed: int | None = None
        self.epoch = 0
        self.grid: MapGrid | None = None
        self.planner: GridPlanner | None = None
        self.pose: Pose | None = None
        self.camera: CameraPacket | None = None
        self.robot: RobotInfo = {}
        self.calibration: JsonDict = {}
        self.map_info: JsonDict = {}
        self.state: JsonDict = {}
        self.goals: tuple[Goal, ...] = ()
        self.marked: set[str] = set()
        self._mark_attempts: set[str] = set()
        self.trajectory: deque[Point] = deque(maxlen=6000)
        self._samples: deque[tuple[float, Pose]] = deque(maxlen=40)
        self._pose_history: deque[tuple[float, Pose]] = deque(maxlen=250)
        self._state_at, self._camera_at, self._depth_at = 0.0, 0.0, 0.0
        self._sim_clock_offset: float | None = None
        self._control_at, self._publish_at, self._trajectory_at = 0.0, 0.0, 0.0
        self._bumps, self._replans = 0, 0
        self._command_failures, self._command_drops = 0, 0
        self._bumper_before = False
        self._estimator: PoseEstimate | None = None
        self.events: deque[JsonDict] = deque(maxlen=500)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="eureka-runtime", daemon=True)
            self._thread.start()

    def submit(self, action: str, **kwargs: Any) -> None:
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

    def _event(self, event: str, **data: Any) -> None:
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
                    self.control.fault(f"{type(exc).__name__}: {exc}")
                    self._event("fault", error=str(exc))
                    self._publish()
                self._shutdown.wait(0.004)
        finally:
            self._halt()
            self.connected = False
            self.sim.close()
            self.recorder.stop()
            self.control.disconnect("Disconnected. Motors stopped.")
            self._publish()

    def _action(self, action: str, args: JsonDict) -> None:
        if action in ("connect", "load_scene"):
            if action == "load_scene":
                self._bootstrap(args)
            else:
                self._bootstrap(None)
        elif action == "navigate":
            if not self.connected or self.pose is None:
                raise RuntimeError("Wait for a connected simulator and fresh pose")
            goal = None
            for candidate in self.goals:
                if candidate.name == args["name"]:
                    goal = candidate
                    break
            if goal is None:
                raise ValueError("Unknown named goal")
            self._cancel("Preparing navigation.")
            decision = self.control.navigate(goal, self._observation(), time.monotonic())
            self._apply(decision)
        elif action == "stop":
            self._cancel("Stopped by operator.")
        elif action == "start_recording":
            assert self.recorder is not None
            if not self.connected or self.grid is None:
                raise RuntimeError("Connect before recording")
            directory = Path(args.get("directory", "recordings"))
            source_hashes = {}
            for source_file in sorted(Path(__file__).parent.glob("*.py")):
                source_hashes[source_file.name] = hashlib.sha256(
                    source_file.read_bytes()
                ).hexdigest()
            goals = []
            for goal in self.goals:
                goals.append(goal.__dict__)
            metadata = {
                "source_sha256": source_hashes,
                "scene": self.scene,
                "seed": self.seed,
                "epoch": self.epoch,
                "robot": self.robot,
                "camera": self.calibration,
                "map": self.map_info,
                "goals": goals,
            }
            self.recorder.start(directory, metadata)
        elif action == "stop_recording":
            assert self.recorder is not None
            self._cancel("Stopped while finalizing camera recording.")
            self._halt()
            self.recorder.stop()
        elif action == "disconnect":
            assert self.recorder is not None
            self._cancel("Disconnected.")
            self._halt()
            self.recorder.stop()
            self.connected = False
            self.control.mode = Mode.DISCONNECTED
        else:
            raise ValueError(f"Unknown application command: {action}")

    def _cancel(self, detail: str) -> None:
        self._zero()
        self.control.cancel(detail)
        if not self.connected:
            self.control.mode = Mode.DISCONNECTED

    def _bootstrap(self, load: JsonDict | None) -> None:
        assert self.sim is not None
        assert self.recorder is not None
        self.control.mode, self.control.detail = (
            Mode.LOADING,
            "Stopping motors and loading simulator metadata.",
        )
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
        goals = []
        for value in values:
            goals.append(Goal.from_dict(value))
        self.goals = tuple(goals)
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
        self._state_at = self._camera_at = self._depth_at = 0.0
        self._sim_clock_offset = None
        self._bumper_before = False
        self._bumps = self._replans = 0
        self.connected = True
        self.control.reset_episode(self.robot, self.planner)
        self._event("episode", scene=self.scene, epoch=self.epoch, seed=self.seed)

    def _receive_states(self) -> bool:
        assert self.sim is not None
        states = self.sim.states()
        for state in states:
            external_reset = False
            if self.state:
                if state["scene"] != self.state["scene"]:
                    external_reset = True
                elif state["t"] < self.state["t"] - 0.25:
                    if state["seq"] > self.state["seq"] or state["seq"] < self.state["seq"] - 10:
                        external_reset = True
            if external_reset:
                self._halt()
                self.marked.clear()
                self._mark_attempts.clear()
                self._bootstrap(None)
                self.control.detail = "External scene reset detected. Navigation stopped."
                return False
            if self.state and state["seq"] <= self.state["seq"]:
                continue
            old_pose = self.pose
            self.state = state
            self.scene = str(state["scene"])
            assert self._estimator is not None
            self.pose = self._estimator.update(state)
            self._pose_history.append((float(state["t"]), self.pose))
            now = time.monotonic()
            offset = now - float(state["t"])
            if self._sim_clock_offset is None:
                self._sim_clock_offset = offset
            else:
                self._sim_clock_offset = min(self._sim_clock_offset, offset)
            self._state_at = min(now, float(state["t"]) + self._sim_clock_offset)
            self._samples.append((state["t"], Pose.from_dict(state["pose"])))
            assert self.recorder is not None
            self.recorder.offer_state(state)
            self.control.observe_pose(old_pose, self.pose)
            if self._state_at - self._trajectory_at > 0.1:
                self._trajectory_at = self._state_at
                self.trajectory.append((self.pose.x, self.pose.y))
            if state["bumper"] and not self._bumper_before:
                self._bumps += 1
                self._event("bumper", pose=self.pose.as_dict())
                if self.control.mode in (
                    Mode.FOLLOWING,
                    Mode.ALIGNING,
                    Mode.VERIFYING,
                    Mode.RECOVERY,
                ):
                    self._zero()
                    self.control.contact(self._observation(), time.monotonic())
            self._bumper_before = bool(state["bumper"])
        return True

    def _receive(self) -> None:
        assert self.sim is not None
        assert self.recorder is not None
        assert self.planner is not None
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
            if self._sim_clock_offset is None:
                self._sim_clock_offset = offset
            else:
                self._sim_clock_offset = min(self._sim_clock_offset, offset)
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
            if added and self.control.mode == Mode.FOLLOWING and self.control.path:
                assert self.pose is not None
                if not self.planner.segment_clear((self.pose.x, self.pose.y), self.control.path[0]):
                    self._plan()
        self._receive_states()

    def _capture_pose(self, capture_time: float, fallback: Pose) -> Pose:
        history = self._pose_history
        if not history or capture_time < history[0][0] or capture_time > history[-1][0] + 0.04:
            return fallback
        for (t0, p0), (t1, p1) in zip(history, list(history)[1:]):
            if t0 <= capture_time <= t1:
                if t1 > t0:
                    ratio = (capture_time - t0) / (t1 - t0)
                else:
                    ratio = 0.0
                return Pose(
                    p0.x + ratio * (p1.x - p0.x),
                    p0.y + ratio * (p1.y - p0.y),
                    wrap_angle(p0.yaw + ratio * wrap_angle(p1.yaw - p0.yaw)),
                )
        return history[-1][1]

    def _plan(self) -> None:
        if self.pose is None or self.control.active_goal is None:
            return
        assert self.planner is not None
        self.control.begin_planning()
        self._publish()
        self._halt()
        self._receive()
        if self._stop.is_set() or self._shutdown.is_set() or self.control.active_goal is None:
            self._cancel("Stopped before planning.")
            return
        if self.state.get("bumper"):
            self._recover()
            return
        try:
            route = self.planner.plan((self.pose.x, self.pose.y), self.control.active_goal.xy)
        except NoPath as exc:
            self.control.planning_failed(str(exc))
            self._event(
                "blocked",
                reason=str(exc),
                pose=self.pose.as_dict(),
                dynamic_cells=int(self.planner.dynamic.sum()),
            )
            return
        completed_at = time.monotonic()
        self.control.accept_path(route, self.pose, completed_at)
        self._replans += 1
        self._control_at = completed_at
        self._event(
            "path",
            goal=self.control.active_goal.name,
            pose=self.pose.as_dict(),
            points=list(self.control.path),
            dynamic_cells=int(self.planner.dynamic.sum()),
        )

    def _publish(self) -> None:
        now = time.monotonic()
        dynamic = None
        if self.planner is not None:
            dynamic = self.planner.dynamic.copy()
            dynamic.flags.writeable = False
        record: JsonDict = {}
        if self.recorder is not None:
            record = dict(self.recorder.status())
        value = Snapshot(
            connected=self.connected,
            scene=self.scene,
            epoch=self.epoch,
            seed=self.seed,
            mode=self.control.mode.value,
            detail=self.control.detail,
            pose=self.pose,
            state=dict(self.state),
            grid=self.grid,
            dynamic=dynamic,
            path=tuple(self.control.path),
            trajectory=tuple(self.trajectory),
            goals=self.goals,
            active_goal=self.control.active_goal,
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

    def _observation(self) -> Observation:
        assert self.pose is not None
        velocity = (0.0, 0.0)
        if self._estimator is not None:
            velocity = self._estimator.velocity
        return Observation(
            self.pose,
            velocity,
            self._state_at,
            self._camera_at,
            self.state.get("t", 0.0),
            tuple(self._samples),
            tuple(self.trajectory),
        )

    def _control(self, now: float) -> None:
        if not self.connected:
            return
        if self._stop.is_set() or self._shutdown.is_set():
            self._cancel("Stopped by operator.")
            return
        if self.pose is None:
            self._zero()
            return
        self._apply(self.control.step(self._observation(), now))

    def _apply(self, decision: Decision) -> None:
        if decision.stop:
            self._zero()
        elif decision.wheels is not None:
            self._send(decision.wheels)
        if decision.request == "plan":
            self._plan()
        elif decision.request == "mark":
            assert decision.mean is not None
            self._mark(decision.mean)

    def _recover(self) -> None:
        self._zero()
        self.control.recover(self._observation(), time.monotonic())

    def _send(self, speeds: Velocity) -> None:
        assert self.sim is not None
        if self._stop.is_set() or self._shutdown.is_set():
            self._zero()
            return
        recovering = self._command_failures > 0
        if recovering:
            speeds = (0.0, 0.0)
        if self.sim.wheels(*speeds):
            self._command_failures = 0
            return
        self._command_failures += 1
        self._command_drops += 1
        self.control.detail = "Command transport is busy. Sending stop before resuming."
        if self._command_failures >= 4:
            self.control.mode = Mode.FAULT
            self.control.detail = "Command transport unavailable. Motor watchdog will stop motion."

    def _mark(self, mean: Pose) -> None:
        assert self.sim is not None
        assert self.pose is not None
        assert self.control.active_goal is not None
        name = self.control.active_goal.name
        self._halt()
        if name not in self.marked:
            if name in self._mark_attempts:
                self.control.mode = Mode.FAULT
                self.control.detail = (
                    "Previous goal mark outcome is unknown. Reset the scene to retry."
                )
                return
            self._mark_attempts.add(name)
            self.control.detail = "Goal mark requested. A timeout leaves its outcome unknown."
            result, _ = self.sim.call("mark_goal", name=name)
            self.marked.add(name)
            self._event(
                "goal_marked",
                name=name,
                mark_t=result["t"],
                estimated_pose=self.pose.as_dict(),
                mean_pose=mean.as_dict(),
                position_error=self.pose.distance(self.control.active_goal.xy),
            )
        self.control.arrival_confirmed(name)
