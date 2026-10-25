"""Waypoint following, recovery, and motion safety."""

import math
import time
from dataclasses import dataclass

import numpy as np

from .domain import Pose, wrap_angle

DEFAULT_NAVIGATION_TIMEOUT = 300.0


@dataclass(frozen=True)
class Observation:
    pose: Pose | None
    velocity: tuple
    state_at: float
    camera_at: float
    sim_time: float
    samples: tuple
    trajectory: tuple


@dataclass
class Decision:
    wheels: tuple | None = None
    stop: bool = False
    request: str | None = None
    mean: Pose | None = None


def goal_satisfied(pose, goal, distance=0.025, angle=0.04):
    position_ok = pose.distance(goal.xy) <= distance
    if not position_ok:
        return False
    if goal.yaw is None:
        return True
    heading_error = abs(wrap_angle(goal.yaw - pose.yaw))
    return heading_error <= angle


def wheel_speeds(linear, angular, robot):
    half = robot["wheel_base"] / 2
    radius = robot["wheel_radius"]
    left = (linear - angular * half) / radius
    right = (linear + angular * half) / radius
    left_ratio = abs(left) / robot["max_wheel_speed"]
    right_ratio = abs(right) / robot["max_wheel_speed"]
    factor = max(1.0, left_ratio, right_ratio)
    return left / factor, right / factor


@dataclass(frozen=True)
class Retreat:
    origin: Pose
    target: tuple[float, float]
    deadline: float


class Controller:
    def __init__(self, robot=None, planner=None, *, navigation_timeout=DEFAULT_NAVIGATION_TIMEOUT):
        if not math.isfinite(navigation_timeout) or navigation_timeout <= 0:
            raise ValueError("Navigation timeout must be finite and positive")
        self.navigation_timeout = navigation_timeout
        self.robot = robot or {}
        self.planner = planner
        self.mode, self.detail = "disconnected", "Connect to the simulator to begin."
        self.active_goal, self.path = None, []
        self._scan_angle, self._scanned = 0.0, False
        self._verify_at, self._retreat = None, None
        self._progress_at, self._best_distance = 0.0, math.inf
        self._nav_at, self._plan_at, self._recoveries = 0.0, 0.0, 0
        self.observation = None
        self.decision = Decision()

    def cancel(self, detail):
        self.mode, self.detail = "idle", detail
        self.active_goal, self.path, self._verify_at = None, [], None
        self._zero()

    def navigate(self, goal, observation, now):
        self.observation = observation
        self.decision = Decision()
        self.active_goal = goal
        self._recoveries = 0
        self._nav_at = self._progress_at = now
        self._best_distance = observation.pose.distance(goal.xy)
        if goal_satisfied(observation.pose, goal):
            self.mode, self.detail = "verifying", "Holding position to verify goal tolerance."
            self._verify_at = None
        elif not self._scanned:
            self.mode, self.detail = "scanning", "Scanning for obstacles missing from the map."
            self._scan_angle = 0.0
        else:
            self._plan()
        return self.decision

    def step(self, observation, now):
        self.observation = observation
        self.decision = Decision()
        self._control(now)
        return self.decision

    def observe_pose(self, previous, current):
        if self.mode == "scanning" and previous is not None:
            self._scan_angle += abs(wrap_angle(current.yaw - previous.yaw))

    def _zero(self):
        self.decision.wheels = (0.0, 0.0)
        self.decision.stop = True

    def _drive(self, linear, angular):
        self.decision.wheels = wheel_speeds(linear, angular, self.robot)
        self.decision.stop = False

    def _plan(self):
        self.decision.request = "plan"

    def recover(self, observation, now):
        self.observation = observation
        pose = observation.pose
        self._recoveries += 1
        self._retreat = None
        if self._recoveries > 3:
            self.mode, self.detail = "blocked", "Repeated contact. Stopped for operator review."
            return
        target = (
            pose.x - 0.06 * math.cos(pose.yaw),
            pose.y - 0.06 * math.sin(pose.yaw),
        )
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        visited = False
        for x, y in list(self.observation.trajectory)[-150:]:
            distance_behind = -(x - pose.x) * cos_yaw - (y - pose.y) * sin_yaw
            if 0.065 <= distance_behind <= 0.15:
                sideways_distance = abs(-(x - pose.x) * sin_yaw + (y - pose.y) * cos_yaw)
                if sideways_distance <= 0.01:
                    visited = True
                    break
        if not visited or not self.planner.segment_clear((pose.x, pose.y), target):
            self.mode, self.detail = (
                "scanning",
                "Contact bearing unknown. Scanning without translation.",
            )
            self._scan_angle = 0.0
            self._scanned = False
            return
        self._retreat = Retreat(pose, target, now + 2.0)
        self.mode, self.detail = "recovery", "Retreating at most 6 cm on a checked segment."

    def _motion_clear(self, linear, angular):
        velocity = self.observation.velocity
        return self.planner.motion_clear(
            self.observation.pose,
            linear,
            angular,
            velocity,
            (self.robot.get("max_linear_accel", 1.0), self.robot.get("max_angular_accel", 6.0)),
            self.robot.get("watchdog", 0.25)
            + max(0.0, time.monotonic() - self.observation.state_at),
        )

    def _control(self, now):
        pose = self.observation.pose
        moving = self.mode in (
            "scanning",
            "waiting",
            "following",
            "aligning",
            "verifying",
            "recovery",
        )
        if not moving:
            self._zero()
            return
        if now - self.observation.state_at > 0.20:
            self.cancel("State stream is stale. Select the goal again after reconnection.")
            self.mode = "fault"
            return
        if self.active_goal is not None and now - self._nav_at > self.navigation_timeout:
            self.cancel(f"Navigation exceeded {self.navigation_timeout:g} seconds.")
            self.mode = "blocked"
            return
        if self.mode == "waiting":
            self._zero()
            if (
                self.observation.state_at >= self._plan_at
                and self.observation.camera_at >= self._plan_at
                and now - self.observation.state_at <= 0.10
                and now - self.observation.camera_at <= 0.25
            ):
                self.mode = "following"
                self.detail = f"Following path to {self.active_goal.name}."
            elif now - self._plan_at > 3:
                self.cancel("Fresh streams did not arrive after planning. Motors remain stopped.")
                self.mode = "fault"
            return
        if self.mode == "scanning":
            if now - self.observation.camera_at > 0.7:
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
            if retreat is None:
                progress = 0.0
            else:
                progress = (retreat.origin.x - pose.x) * math.cos(retreat.origin.yaw) + (
                    retreat.origin.y - pose.y
                ) * math.sin(retreat.origin.yaw)
            velocity = self.observation.velocity
            reverse_speed = max(0.04, -velocity[0])
            observation_age = max(0.0, now - self.observation.state_at)
            stopping_distance = reverse_speed * (0.05 + observation_age) + reverse_speed**2 / (
                2 * self.robot.get("max_linear_accel", 1.0)
            )
            if retreat is None or progress + stopping_distance >= 0.06 or now >= retreat.deadline:
                self._zero()
                self._retreat = None
                self._plan()
            elif (
                not self.planner.segment_clear((pose.x, pose.y), retreat.target)
                or abs(wrap_angle(pose.yaw - retreat.origin.yaw)) > 0.03
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
        distance = pose.distance(goal.xy)
        if distance < self._best_distance - 0.025:
            self._best_distance, self._progress_at = distance, now
        if now - self._progress_at > 15 and self.mode == "following":
            self._plan()
            return
        if distance <= 0.020:
            if goal.yaw is None:
                error = 0.0
            else:
                error = wrap_angle(goal.yaw - pose.yaw)
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
        if now - self.observation.camera_at > 0.55:
            self.cancel("Camera stream is stale. Stopped before moving into unseen space.")
            self.mode = "fault"
            return
        while len(self.path) > 1 and pose.distance(self.path[0]) < 0.10:
            if not self.planner.segment_clear((pose.x, pose.y), self.path[1]):
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
        if not self.planner.segment_clear((pose.x, pose.y), target):
            self._zero()
            self._plan()
            return
        error = wrap_angle(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw)
        angular = float(np.clip(2.6 * error, -1.3, 1.3))
        forward_speed = min(0.20, 0.9 * pose.distance(target))
        heading_factor = max(0.0, math.cos(error))
        linear = forward_speed * heading_factor
        if abs(error) > 0.60:
            linear = 0.0
        if self.planner._clearance((pose.x, pose.y)) < (
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

    def _verify(self, now):
        pose = self.observation.pose
        if not goal_satisfied(pose, self.active_goal):
            self._verify_at = None
            return
        if self._verify_at is None:
            self._verify_at = now
            return
        if now - self._verify_at < 0.6 or len(self.observation.samples) < 15:
            return
        recent = []
        for sample_time, pose in self.observation.samples:
            if sample_time > self.observation.sim_time - 0.5:
                recent.append(pose)
        if len(recent) < 15:
            return
        x_values = []
        for pose in recent:
            x_values.append(pose.x)
        mean_x = float(np.mean(x_values))

        y_values = []
        for pose in recent:
            y_values.append(pose.y)
        mean_y = float(np.mean(y_values))

        sin_values = []
        for pose in recent:
            sin_values.append(math.sin(pose.yaw))
        sin_sum = sum(sin_values)
        cos_values = []
        for pose in recent:
            cos_values.append(math.cos(pose.yaw))
        cos_sum = sum(cos_values)
        mean_yaw = math.atan2(sin_sum, cos_sum)
        mean = Pose(mean_x, mean_y, mean_yaw)
        if not goal_satisfied(mean, self.active_goal, 0.030, 0.05):
            return
        self.decision.request = "mark"
        self.decision.mean = mean
