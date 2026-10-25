import ast
import inspect
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

import eureka.runtime as runtime_module
from eureka.controller import Mode, wheel_speeds
from eureka.domain import CameraPacket, Goal, MapGrid, Pose
from eureka.planning import GridPlanner
from eureka.runtime import PoseEstimate, Runtime

ROBOT = {"wheel_radius": 0.035, "wheel_base": 0.23, "max_wheel_speed": 10, "ticks_per_rev": 1024}


class Motors:
    def __init__(self):
        self.sent = []

    def wheels(self, left=0, right=0):
        self.sent.append((left, right))
        return True


def camera_calibration():
    return {
        "width": 640,
        "height": 480,
        "fx": 415.69,
        "fy": 415.69,
        "cx": 319.5,
        "cy": 239.5,
        "depth_scale": 0.001,
        "max_depth": 6.0,
        "rate": 15,
        "T_base_camera": [[0, 0, 1, 0.16], [-1, 0, 0, 0], [0, -1, 0, 0.08], [0, 0, 0, 1]],
    }


def navigating_runtime(pose=Pose(1, 1, 0), goal=Goal("g", 2, 1), now=None):
    if now is None:
        now = time.monotonic()
    runtime = Runtime()
    runtime.connected = True
    runtime.sim = Motors()
    runtime.robot = ROBOT
    runtime.pose = pose
    runtime.state = {"t": 1.0}
    runtime.planner = GridPlanner(MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0)))
    runtime.control.reset_episode(ROBOT, runtime.planner)
    runtime._state_at = runtime._camera_at = now
    runtime.control.navigate(goal, runtime._observation(), now)
    return runtime


def test_stop_supersedes_queued_navigation():
    runtime = Runtime()
    runtime.submit("navigate", name="first")
    runtime.submit("navigate", name="second")
    runtime.submit("stop")
    assert runtime._commands.qsize() == 1
    assert runtime._commands.get_nowait()[0] == "stop"
    assert runtime._stop.is_set()


def test_disconnected_runtime_does_not_keep_sending_commands():
    runtime = Runtime()
    runtime.sim = Motors()
    runtime._control(time.monotonic())
    assert runtime.sim.sent == []


def test_encoder_integration_tracks_forward_displacement():
    estimate = PoseEstimate(ROBOT, Pose(0, 0, 0))
    estimate.update(
        {"t": 1.0, "pose": Pose(0, 0, 0).as_dict(), "wheels": {"left_ticks": 0, "right_ticks": 0}}
    )
    expected = 2 * math.pi * 0.035
    actual = estimate.update(
        {
            "t": 1.1,
            "pose": Pose(expected, 0, 0).as_dict(),
            "wheels": {"left_ticks": 1024, "right_ticks": 1024},
        }
    )
    assert actual.x == pytest.approx(expected)
    assert actual.y == 0
    assert estimate.velocity == pytest.approx((expected / 0.1, 0))


def test_encoder_velocity_includes_turning_and_signed_reverse():
    estimate = PoseEstimate(ROBOT, Pose(0, 0, 0))
    estimate.update(
        {"t": 1.0, "pose": Pose(0, 0, 0).as_dict(), "wheels": {"left_ticks": 0, "right_ticks": 0}}
    )
    estimate.update(
        {
            "t": 1.1,
            "pose": Pose(0, 0, 0).as_dict(),
            "wheels": {"left_ticks": -20, "right_ticks": -10},
        }
    )
    assert estimate.velocity[0] < 0
    assert estimate.velocity[1] > 0


def test_encoder_prediction_is_gradually_corrected_toward_reported_pose():
    estimate = PoseEstimate(ROBOT, Pose(1, 1, 0))
    estimate.update(
        {
            "t": 1.0,
            "pose": Pose(1, 1, 0).as_dict(),
            "wheels": {"left_ticks": 1000, "right_ticks": 1000},
        }
    )
    actual = estimate.update(
        {
            "t": 1.1,
            "pose": Pose(1.024, 1, 0).as_dict(),
            "wheels": {"left_ticks": 1100, "right_ticks": 1100},
        }
    )
    assert actual.x == pytest.approx(1.021564080444335)
    assert actual.y == 1
    assert actual.yaw == 0


def test_planning_defers_control_until_stream_can_be_refreshed(monkeypatch):
    runtime = navigating_runtime()
    runtime._halt = runtime._publish = runtime._receive = lambda: None
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    runtime._plan()
    assert 10.0 - runtime._control_at < 0.05
    assert runtime.control.mode is Mode.WAITING
    assert not any(left or right for left, right in runtime.sim.sent)


def test_replanning_uses_pose_after_motors_stop():
    runtime = navigating_runtime(Pose(0.5, 1, 0))
    runtime._halt = runtime._publish = lambda: None
    runtime._receive = lambda: setattr(runtime, "pose", Pose(0.52, 1, 0))
    starts = []

    class Planner(GridPlanner):
        def plan(self, start, goal):
            starts.append(start)
            return [start, goal]

    runtime.planner = Planner(MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0)))
    runtime._plan()
    assert starts == [(0.52, 1)]


def test_unknown_goal_mark_is_never_retried_or_reported_as_confirmed():
    runtime = navigating_runtime(Pose(1, 1, 0), Goal("g", 1, 1))
    runtime._halt = lambda: None

    class LostReply(Motors):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def call(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError("mark outcome unknown")

    runtime.sim = LostReply()
    with pytest.raises(TimeoutError):
        runtime._mark(runtime.pose)
    assert not runtime.marked
    runtime._mark(runtime.pose)
    assert runtime.control.mode is Mode.FAULT
    assert runtime.sim.calls == 1
    assert not runtime.marked
    assert "unknown" in runtime.control.detail.lower()


def test_acknowledged_goal_mark_is_recorded_once():
    runtime = navigating_runtime(Pose(1, 1, 0), Goal("g", 1, 1))
    runtime._halt = lambda: None

    class MarkReply(Motors):
        calls = 0

        def call(self, method, **params):
            assert method == "mark_goal"
            assert params == {"name": "g"}
            self.calls += 1
            return {"t": 1.0}, []

    runtime.sim = MarkReply()
    runtime._mark(runtime.pose)
    assert runtime.control.mode is Mode.ARRIVED
    assert runtime.marked == {"g"}
    runtime._mark(runtime.pose)
    assert runtime.sim.calls == 1
    assert runtime.control.path == []


def test_reset_with_continuing_sequence_counter_clears_old_episode():
    runtime = Runtime()
    runtime.state = {"scene": "01", "t": 50.0, "seq": 2500}
    runtime.marked.add("old")
    runtime._mark_attempts.add("old")
    bootstraps = []

    class ResetStream(Motors):
        def states(self):
            return [{"scene": "01", "t": 0.1, "seq": 2501}]

    runtime.sim = ResetStream()
    runtime.recorder = runtime_module.Recorder(make_video=False)
    runtime.planner = GridPlanner(MapGrid(np.zeros((50, 50), dtype=bool), 0.02, (0, 0)))
    runtime._halt = runtime._zero
    runtime._bootstrap = lambda load: bootstraps.append(load)
    runtime._receive()
    assert bootstraps == [None]
    assert not runtime.marked
    assert not runtime._mark_attempts
    assert runtime.sim.sent[-1] == (0, 0)


def test_newly_received_delayed_state_cannot_authorize_motion(monkeypatch):
    runtime = navigating_runtime(Pose(1, 1, 0), now=100.0)
    runtime.control.accept_path([(1, 1), (2, 1)], runtime.pose, 100.0)
    runtime.control.step(runtime._observation(), 100.0)
    runtime.state = {"scene": "01", "t": 0.0, "seq": 1}
    runtime._sim_clock_offset = 100.0
    runtime._camera_at = 101.0
    runtime._estimator = PoseEstimate(ROBOT, runtime.pose)

    class DelayedState(Motors):
        def states(self):
            return [
                {
                    "scene": "01",
                    "t": 0.02,
                    "seq": 2,
                    "bumper": False,
                    "pose": Pose(1, 1, 0).as_dict(),
                    "wheels": {"left_ticks": 0, "right_ticks": 0},
                }
            ]

        def cameras(self):
            return []

    class NoRecording:
        def offer_state(self, state):
            pass

    runtime.sim, runtime.recorder = DelayedState(), NoRecording()
    monkeypatch.setattr(time, "monotonic", lambda: 101.0)
    runtime._receive()
    runtime._control(101.0)
    assert runtime.control.mode is Mode.FAULT
    assert runtime.sim.sent[-1] == (0, 0)


def test_mapping_pose_interpolates_capture_time_and_wraps_heading():
    runtime = Runtime()
    runtime._pose_history = deque([(1.0, Pose(0, 0, 3.10)), (2.0, Pose(1, 0, -3.10))])
    fallback = Pose(20, 20, 0)
    pose = runtime._capture_pose(1.5, fallback)
    assert pose.x == pytest.approx(0.5)
    assert pose.y == 0
    assert abs(abs(pose.yaw) - math.pi) < 1e-6
    assert runtime._capture_pose(0.0, fallback) is fallback


def test_close_reports_incomplete_finalization_and_default_waits(monkeypatch, tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class NoNetwork(Motors):
        def __init__(self, *args):
            super().__init__()

        def close(self):
            pass

    def encode(self):
        entered.set()
        release.wait(3)

    monkeypatch.setattr(runtime_module, "Simulator", NoNetwork)
    monkeypatch.setattr(runtime_module.Recorder, "_video", encode)
    runtime = Runtime()
    runtime.start()
    deadline = time.monotonic() + 2
    while runtime.recorder is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtime.recorder is not None
    folder = runtime.recorder.start(tmp_path, {"camera": camera_calibration()})
    try:
        with pytest.raises(TimeoutError):
            runtime.close(timeout=0.4)
        assert entered.is_set()
    finally:
        release.set()
        runtime.close()
    assert not runtime._thread.is_alive()
    assert json.loads((folder / "manifest.json").read_text())["clean_close"]


def test_goal_start_does_not_send_motion_in_same_tick_as_stop(monkeypatch):
    runtime = Runtime()
    runtime.sim = Motors()
    runtime.recorder = runtime_module.Recorder(make_video=False)
    runtime.connected = True
    runtime.pose = Pose(0, 0, 0)
    runtime.goals = (Goal("g", 1, 0),)
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    runtime._action("navigate", {"name": "g"})
    assert 10.0 - runtime._control_at < 0.05
    assert runtime.sim.sent == [(0, 0)]


def test_temporary_command_backpressure_stops_before_resuming():
    runtime = navigating_runtime()

    class Backpressure(Motors):
        def wheels(self, left=0, right=0):
            self.sent.append((left, right))
            return len(self.sent) > 1

    runtime.sim = Backpressure()
    speeds = wheel_speeds(0.1, 0, ROBOT)
    runtime._send(speeds)
    assert runtime.control.mode is not Mode.FAULT
    runtime._send(speeds)
    assert runtime.sim.sent[-1] == (0, 0)
    runtime._send(speeds)
    assert runtime.sim.sent[-1][0] > 0


def test_persistent_command_backpressure_faults():
    runtime = navigating_runtime()

    class Unavailable(Motors):
        def wheels(self, left=0, right=0):
            self.sent.append((left, right))
            return False

    runtime.sim = Unavailable()
    for _ in range(4):
        runtime._send(wheel_speeds(0.1, 0, ROBOT))
    assert runtime.control.mode is Mode.FAULT
    assert all(pair == (0, 0) for pair in runtime.sim.sent[1:])


def test_delayed_camera_frames_are_recorded_even_when_too_old_for_navigation():
    runtime = Runtime()
    runtime.state = {"t": 3.0}
    runtime._receive_states = lambda: True
    packet = CameraPacket(
        {"t": 1.0, "seq": 1, "width": 1, "height": 1, "pose": Pose(0, 0, 0).as_dict()},
        b"jpeg",
        b"\x00\x00",
        0,
    )
    saved = []

    class CameraStream:
        def cameras(self):
            return [packet]

    class Writer:
        def offer(self, value):
            saved.append(value)

    runtime.sim, runtime.recorder = CameraStream(), Writer()
    runtime.planner = GridPlanner(MapGrid(np.zeros((50, 50), dtype=bool), 0.02, (0, 0)))
    runtime._receive()
    assert saved == [packet]
    assert runtime.camera is None


def test_connect_without_load_clears_unknown_seed_and_uses_runtime_body_height():
    runtime = Runtime()
    runtime.seed = 18
    runtime._halt = runtime._publish = lambda: None
    grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))

    class ExistingEpisode(Motors):
        def call(self, method):
            values = {
                "get_robot_info": {**ROBOT, "radius": 0.17, "height": 0.11},
                "get_camera_info": camera_calibration(),
                "get_goals": [],
            }
            return values[method], []

        def get_map(self):
            return grid, {}, b""

        def restart_streams(self):
            pass

    class Writer:
        def stop(self):
            pass

    runtime.sim, runtime.recorder = ExistingEpisode(), Writer()
    runtime._bootstrap(None)
    assert runtime.seed is None
    assert runtime.planner.height == 0.11
    assert runtime.control.mode is Mode.IDLE


def test_snapshot_publishes_plain_mode_string():
    runtime = Runtime()
    runtime._publish()
    snapshot = runtime.snapshot()
    assert snapshot.mode == "disconnected"
    assert type(snapshot.mode) is str


def test_runtime_does_not_access_private_controller_state():
    tree = ast.parse(Path(inspect.getfile(Runtime)).read_text())
    private = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("_")
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "control"
    ]
    assert private == []
