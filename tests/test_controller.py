import math
import time
from dataclasses import replace

import numpy as np
import pytest

from eureka.controller import Controller, Mode, Observation, goal_satisfied, wheel_speeds
from eureka.domain import Goal, MapGrid, Pose, wrap_angle
from eureka.planning import GridPlanner

ROBOT = {"wheel_radius": 0.035, "wheel_base": 0.23, "max_wheel_speed": 10, "ticks_per_rev": 1024}
NOW = 100.0


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: NOW)


def observation(pose=Pose(1, 1, 0), now=NOW, **changes):
    value = Observation(pose, (0.0, 0.0), now, now, 1.0, (), ())
    return replace(value, **changes)


def following(pose=Pose(1, 1, 0), goal=Goal("g", 2, 1), grid=None, route=None):
    if grid is None:
        grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))
    planner = GridPlanner(grid)
    control = Controller(ROBOT, planner)
    control.reset_episode(ROBOT, planner)
    data = observation(pose)
    control.navigate(goal, data, NOW)
    if route is None:
        route = planner.plan((pose.x, pose.y), goal.xy)
    control.accept_path(route, pose, NOW)
    waiting = control.step(data, NOW)
    assert waiting.stop
    assert control.mode is Mode.FOLLOWING
    return control, planner, data


def test_goal_tolerances_wrap_yaw_and_optional_heading():
    assert goal_satisfied(Pose(0, 0, -math.pi + 0.01), Goal("g", 0, 0, math.pi - 0.01))
    assert goal_satisfied(Pose(0, 0, 2), Goal("g", 0, 0))
    assert not goal_satisfied(Pose(0.04, 0, 0), Goal("g", 0, 0))
    assert not goal_satisfied(Pose(0, 0, 0.2), Goal("g", 0, 0, 0))
    assert abs(wrap_angle(2 * math.pi + 0.2) - 0.2) < 1e-10


def test_wheel_limits_preserve_curvature():
    left, right = wheel_speeds(0.4, 1.0, ROBOT)
    assert max(abs(left), abs(right)) == 10
    assert right > left > 0
    left, right = wheel_speeds(0, 1, ROBOT)
    assert left == -right


def test_idle_requests_stopped_wheels():
    control = Controller()
    decision = control.step(observation(), NOW)
    assert decision.stop
    assert decision.wheels == (0, 0)
    assert decision.request is None


def test_stale_state_stops_and_cancels_navigation():
    control, _, data = following()
    decision = control.step(replace(data, state_at=NOW - 1), NOW)
    assert decision.stop
    assert decision.wheels == (0, 0)
    assert control.mode is Mode.FAULT
    assert control.active_goal is None


def test_stale_camera_stops_before_entering_unseen_space():
    control, _, data = following()
    decision = control.step(replace(data, camera_at=NOW - 1), NOW)
    assert decision.stop
    assert control.mode is Mode.FAULT
    assert control.active_goal is None


def test_following_commands_forward_movement_toward_waypoint():
    control, _, data = following()
    decision = control.step(data, NOW)
    assert decision.request is None
    assert decision.wheels[0] == pytest.approx(decision.wheels[1])
    assert decision.wheels[0] > 0


def test_cancel_clears_route_and_requests_stopping():
    control, _, data = following()
    control.cancel("Stopped by operator.")
    decision = control.step(data, NOW)
    assert control.mode is Mode.IDLE
    assert control.active_goal is None
    assert control.path == []
    assert decision.stop


def test_scanning_rotates_then_requests_route():
    grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))
    control = Controller(ROBOT, GridPlanner(grid))
    data = observation()
    control.navigate(Goal("g", 2, 1), data, NOW)
    decision = control.step(data, NOW)
    assert control.mode is Mode.SCANNING
    assert decision.wheels[0] == -decision.wheels[1]
    assert decision.wheels[1] > 0
    previous = data.pose
    for heading in np.linspace(0, 6.2, 20)[1:]:
        current = Pose(1, 1, float(heading))
        control.observe_pose(previous, current)
        previous = current
    decision = control.step(replace(data, pose=previous), NOW)
    assert decision.stop
    assert decision.request == "plan"


def test_scanning_waits_for_fresh_camera():
    grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))
    control = Controller(ROBOT, GridPlanner(grid))
    data = observation(camera_at=NOW - 1)
    control.navigate(Goal("g", 2, 1), data, NOW)
    decision = control.step(data, NOW)
    assert control.mode is Mode.SCANNING
    assert decision.stop
    assert decision.wheels == (0, 0)


def test_postplanning_wait_requires_fresh_postplan_observations():
    control, planner, data = following()
    control.accept_path(planner.plan((1, 1), (2, 1)), data.pose, NOW)
    decision = control.step(replace(data, camera_at=NOW - 0.2), NOW)
    assert control.mode is Mode.WAITING
    assert decision.stop
    decision = control.step(observation(now=NOW + 0.1), NOW + 0.1)
    assert control.mode is Mode.FOLLOWING
    assert decision.stop


def test_postplanning_wait_times_out_stopped():
    control, _, data = following()
    control.accept_path([(1, 1), (2, 1)], data.pose, NOW - 4)
    decision = control.step(replace(data, camera_at=NOW - 4), NOW)
    assert control.mode is Mode.FAULT
    assert control.active_goal is None
    assert decision.stop


def test_corner_waypoint_is_followed_before_unsafe_shortcut():
    grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))
    grid.occupied[85:100, 50:65] = True
    current, corner, after = (0.78, 1.43), (0.78, 1.52), (1.52, 1.52)
    control, planner, data = following(
        Pose(*current, math.pi / 2),
        Goal("g", 2, 1.52),
        grid,
        [current, corner, after, (2, 1.52)],
    )
    assert planner.segment_clear(current, corner)
    assert planner.segment_clear(corner, after)
    assert not planner.segment_clear(current, after)
    decision = control.step(data, NOW)
    assert decision.request is None
    assert control.path[0] == corner
    assert all(speed > 0 for speed in decision.wheels)


def test_confirmed_obstacle_on_later_segment_requests_replanning_without_forward_motion():
    control, planner, data = following(
        Pose(0.95, 1, 0), route=[(0.95, 1), (1, 1), (1.5, 1), (2, 1)]
    )
    planner.dynamic[planner.grid.world_to_cell(1.7, 1)] = True
    planner.revision += 1
    decision = control.step(data, NOW)
    assert decision.request == "plan"
    assert decision.wheels is None or decision.wheels == (0, 0)


def test_following_without_progress_requests_another_route():
    control, _, data = following()
    decision = control.step(replace(data, state_at=NOW + 16, camera_at=NOW + 16), NOW + 16)
    assert decision.request == "plan"
    assert decision.wheels is None or decision.wheels == (0, 0)


def test_contact_does_not_invent_direction_or_reverse_into_known_obstacle():
    grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))
    grid.occupied[grid.world_to_cell(0.75, 1.01)] = True
    control, planner, data = following(Pose(1.01, 1, 0), grid=grid)
    data = replace(data, trajectory=((0.961, 1.0),))
    control.recover(data, NOW)
    decision = control.step(data, NOW)
    assert not planner.dynamic.any()
    assert control.mode is Mode.SCANNING
    assert sum(decision.wheels) >= 0


@pytest.mark.parametrize("visited", [(0.98, 1), (0.80, 1), (0.92, 1.02)])
def test_retreat_needs_recent_space_behind_and_aligned(visited):
    control, _, data = following()
    data = replace(data, trajectory=(visited,))
    control.recover(data, NOW)
    assert control.mode is Mode.SCANNING
    decision = control.step(data, NOW)
    assert decision.wheels[0] == -decision.wheels[1]


def test_recovery_stops_at_validated_distance_and_requests_replanning():
    control, _, data = following()
    data = replace(data, trajectory=((0.92, 1),))
    control.recover(data, NOW)
    decision = control.step(data, NOW)
    assert all(speed < 0 for speed in decision.wheels)
    decision = control.step(observation(Pose(0.94, 1, 0), NOW + 0.1), NOW + 0.1)
    assert decision.stop
    assert decision.request == "plan"


def test_recovery_brakes_before_limit_with_incoming_reverse_velocity():
    control, _, data = following()
    control.recover(replace(data, trajectory=((0.92, 1),)), NOW)
    decision = control.step(replace(data, pose=Pose(0.95, 1, 0), velocity=(-0.15, 0)), NOW)
    assert decision.stop
    assert decision.request == "plan"


def test_recovery_rechecks_obstacle_added_after_route_validation():
    control, planner, data = following()
    data = replace(data, trajectory=((0.92, 1),))
    control.recover(data, NOW)
    planner.dynamic[planner.grid.world_to_cell(0.78, 1)] = True
    planner.revision += 1
    decision = control.step(data, NOW)
    assert decision.stop
    assert control.mode is Mode.SCANNING


def test_new_contact_during_retreat_switches_to_scanning():
    control, _, data = following()
    data = replace(data, trajectory=((0.92, 1),))
    control.recover(data, NOW)
    assert control.mode is Mode.RECOVERY
    control.contact(data, NOW + 0.1)
    assert control.mode is Mode.SCANNING
    decision = control.step(observation(now=NOW + 0.1), NOW + 0.1)
    assert decision.wheels[0] == -decision.wheels[1]


def test_fourth_recovery_is_blocked_and_reset_starts_new_episode():
    control, planner, data = following()
    data = replace(data, trajectory=((0.92, 1),))
    for _ in range(3):
        control.recover(data, NOW)
        assert control.mode is Mode.RECOVERY
    control.recover(data, NOW)
    assert control.mode is Mode.BLOCKED
    assert control.step(data, NOW).stop
    control.reset_episode(ROBOT, planner)
    control.navigate(Goal("g", 2, 1), data, NOW)
    assert control.mode is Mode.SCANNING
    control.recover(data, NOW)
    assert control.mode is Mode.RECOVERY


def test_retreat_deadline_requests_stopping_and_new_route():
    control, _, data = following()
    control.recover(replace(data, trajectory=((0.92, 1),)), NOW)
    decision = control.step(observation(now=NOW + 2.1), NOW + 2.1)
    assert decision.stop
    assert decision.request == "plan"


def test_narrow_escape_aligns_without_requesting_unsafe_arc():
    grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))
    grid.occupied[-1, :] = True
    control, _, data = following(Pose(0.5, 0.1921, -0.2913113497618344), Goal("g", 1.5, 0.5), grid)
    decision = control.step(data, NOW)
    left, right = decision.wheels
    assert left == -right
    assert right > 0


def test_goal_alignment_does_not_request_mark_outside_required_heading():
    control, _, data = following(Pose(1, 1, 0.2), Goal("g", 1, 1, 0))
    decision = control.step(data, NOW)
    assert control.mode is Mode.ALIGNING
    assert decision.request is None
    assert decision.wheels[0] == -decision.wheels[1]


def test_arrival_requires_hold_time_and_recent_samples():
    control = Controller(ROBOT)
    goal = Goal("g", 1, 1, 0)
    data = observation(samples=((1.0, Pose(1, 1, 0)),) * 20)
    control.navigate(goal, data, NOW)
    assert control.step(data, NOW).request is None
    assert control.step(observation(now=NOW + 0.3, samples=data.samples), NOW + 0.3).request is None
    old = observation(now=NOW + 0.7, sim_time=2.0, samples=data.samples)
    assert control.step(old, NOW + 0.7).request is None
    fresh = replace(old, samples=((1.8, Pose(1.01, 1, 0)),) * 20)
    decision = control.step(fresh, NOW + 0.7)
    assert decision.stop
    assert decision.request == "mark"
    assert decision.mean.x == pytest.approx(1.01)


def test_arrival_rejects_average_pose_outside_tolerance():
    control = Controller(ROBOT)
    data = observation(samples=((1.0, Pose(1.04, 1, 0)),) * 20)
    control.navigate(Goal("g", 1, 1, 0), data, NOW)
    control.step(data, NOW)
    decision = control.step(replace(data, state_at=NOW + 0.7, camera_at=NOW + 0.7), NOW + 0.7)
    assert decision.request is None


def test_arrival_heading_average_handles_angle_wrap():
    control = Controller(ROBOT)
    poses = ((1.0, Pose(1, 1, math.pi - 0.01)), (1.0, Pose(1, 1, -math.pi + 0.01))) * 10
    data = observation(Pose(1, 1, math.pi), samples=poses)
    control.navigate(Goal("g", 1, 1, math.pi), data, NOW)
    control.step(data, NOW)
    decision = control.step(replace(data, state_at=NOW + 0.7, camera_at=NOW + 0.7), NOW + 0.7)
    assert decision.request == "mark"
    assert abs(abs(decision.mean.yaw) - math.pi) < 1e-8


@pytest.mark.parametrize(
    "budget,elapsed,expected", [(300, 181, Mode.FOLLOWING), (2, 3, Mode.BLOCKED)]
)
def test_navigation_uses_configured_timeout(budget, elapsed, expected):
    control = Controller(ROBOT, navigation_timeout=budget)
    data = observation()
    control.navigate(Goal("g", 2, 1), data, NOW - elapsed)
    control.accept_path([(1, 1), (2, 1)], data.pose, NOW)
    decision = control.step(data, NOW)
    assert control.mode is expected
    assert decision.stop
    if expected is Mode.BLOCKED:
        assert control.active_goal is None
    else:
        assert control.active_goal is not None


@pytest.mark.parametrize("timeout", [0, -1, math.inf, -math.inf, math.nan])
def test_navigation_timeout_rejects_invalid_configuration(timeout):
    with pytest.raises(ValueError, match="Navigation timeout must be finite and positive"):
        Controller(navigation_timeout=timeout)


def test_default_deadline_allows_navigation_after_180_seconds():
    control = Controller(ROBOT)
    data = observation()
    control.navigate(Goal("g", 2, 1), data, NOW - 181)
    control.accept_path([(1, 1), (2, 1)], data.pose, NOW)
    decision = control.step(data, NOW)
    assert control.navigation_timeout == 300
    assert control.mode is Mode.FOLLOWING
    assert control.active_goal is not None
    assert decision.stop
