import math

import numpy as np
import pytest

from eureka.domain import CameraPacket, MapGrid, Pose
from eureka.planning import GridPlanner, NoPath, project_obstacles


def room(size=100, resolution=0.02):
    occupied = np.zeros((size, size), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    return MapGrid(occupied, resolution, (0.0, 0.0))


def camera(depth, pose=Pose(1.0, 2.0, 0.0), seq=1):
    depth = np.asarray(depth, dtype="<u2")
    return CameraPacket(
        {
            "width": depth.shape[1],
            "height": depth.shape[0],
            "pose": pose.as_dict(),
            "seq": seq,
            "t": seq / 15,
        },
        b"",
        depth.tobytes(),
        0,
    )


def calibration():
    return {
        "fx": 100.0,
        "fy": 100.0,
        "cx": 0.0,
        "cy": 0.0,
        "depth_scale": 0.001,
        "max_depth": 6.0,
        "T_base_camera": [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.08], [0, 0, 0, 1]],
    }


def test_map_coordinates_use_bottom_left_origin_and_top_image_row():
    grid = MapGrid(np.zeros((4, 5), dtype=bool), 0.2, (-1.0, -2.0))
    assert grid.cell_to_world(0, 0) == pytest.approx((-0.9, -1.3))
    assert grid.cell_to_world(3, 4) == pytest.approx((-0.1, -1.9))
    assert grid.world_to_cell(-0.9, -1.3) == (0, 0)
    assert grid.world_to_cell(-0.1, -1.9) == (3, 4)
    assert grid.world_to_cell(-1.01, -1.9)[1] == -1


def test_planner_detours_around_wall_with_swept_clearance():
    grid = room(150)
    grid.occupied[45:135, 75] = True
    planner = GridPlanner(grid)
    start, goal = (0.7, 0.8), (2.25, 0.8)
    path = planner.plan(start, goal)
    assert path[0] == start
    assert path[-1] == goal
    assert max(y for _, y in path) > 2.3
    assert all(planner.segment_clear(a, b) for a, b in zip(path, path[1:]))
    assert all(math.dist(a, b) <= 0.5 + 1e-10 for a, b in zip(path, path[1:]))
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0, 1, 101):
            point = np.array(a) * (1 - t) + np.array(b) * t
            nearest_wall = np.array([1.51, np.clip(point[1], 0.3, 2.1)])
            assert np.linalg.norm(point - nearest_wall) > 0.2


def test_planner_reports_unreachable_goal():
    grid = room()
    grid.occupied[:, 50] = True
    with pytest.raises(NoPath):
        GridPlanner(grid).plan((0.5, 1.0), (1.5, 1.0))


def test_swept_disc_cannot_clip_obstacle_or_cut_narrow_gap():
    grid = room()
    grid.occupied[50, 50] = True
    planner = GridPlanner(grid)
    assert not planner.segment_clear((0.4, 1.13), (1.6, 1.13))
    assert planner.segment_clear((0.4, 1.4), (1.6, 1.4))
    assert not planner.segment_clear((0.1, 0.1), (0.4, 0.4))
    with pytest.raises(NoPath):
        planner.plan((0.5, 0.5), (1.03, 1.0))


def test_segment_cannot_touch_corner_between_diagonally_adjacent_free_cells():
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[7, 1] = True
    grid = MapGrid(occupied, 0.1, (0.0, 0.0))
    planner = GridPlanner(grid, radius=0.01, margin=0.0)
    assert not planner.segment_clear((0.15, 0.15), (0.25, 0.25))


def test_start_can_escape_conservative_margin_without_approaching_wall():
    planner = GridPlanner(room())
    start = (0.205, 1.0)
    assert planner.blocked[planner.grid.world_to_cell(*start)]
    path = planner.plan(start, (1.0, 1.0))
    assert path[-1] == (1.0, 1.0)
    assert planner.segment_clear(path[0], path[1])
    assert not planner.segment_clear(start, (0.19, 1.0))


def test_subcell_start_can_escape_margin_around_obstacle_corner():
    grid = MapGrid(np.zeros((100, 100), dtype=bool), 0.02, (-1.0, -2.0))
    grid.occupied[grid.world_to_cell(0.29, -1.41)] = True
    planner = GridPlanner(grid)
    start = (0.17049664680746804, -1.5862020091080227)
    goal = (-0.3, -0.9)

    path = planner.plan(start, goal)

    assert path[0] == start
    assert path[-1] == goal
    assert all(planner.segment_clear(a, b) for a, b in zip(path, path[1:]))
    assert planner.segment_clear(start, (0.15, -1.59))
    assert not planner.segment_clear(start, (0.18, -1.57))
    assert not planner.segment_clear(start, (0.29, -1.41))


def test_depth_projection_uses_capture_pose_optical_depth_and_height_band():
    packet = camera([[1000, 0], [1000, 1000]], Pose(1, 2, math.pi / 2))
    points = project_obstacles(packet, calibration())
    assert len(points) == 3
    assert points[0] == pytest.approx((1.0, 3.0, 0.08))
    assert points[-1] == pytest.approx((1.01, 3.0, 0.07))
    invalid = camera([[0, 49, 6001]])
    assert project_obstacles(invalid, calibration()).shape == (0, 3)
    low = calibration()
    low["T_base_camera"][2][3] = 0.01
    assert project_obstacles(camera([[1000]]), low).shape == (0, 3)
    high = calibration()
    high["T_base_camera"][2][3] = 0.3
    assert project_obstacles(camera([[1000]]), high).shape == (0, 3)


def test_dynamic_obstacle_requires_two_distinct_supported_captures():
    grid = room(200)
    planner = GridPlanner(grid)
    packet = camera(np.full((4, 4), 600), Pose(1.0, 1.0, 0.0))
    assert planner.update_depth(packet, calibration()) == 0
    assert planner.update_depth(packet, calibration()) == 0
    assert not planner.dynamic.any()
    newer = camera(np.full((4, 4), 600), Pose(1.0, 1.0, 0.0), 2)
    assert planner.update_depth(newer, calibration()) > 0
    assert planner.revision == 1
    assert not planner.segment_clear((1.0, 1.0), (2.0, 1.0))
    assert planner.update_depth(newer, calibration()) == 0


def test_known_wall_returns_do_not_grow_dynamic_obstacles():
    grid = room(200)
    grid.occupied[:, 80] = True
    planner = GridPlanner(grid)
    for seq in (1, 2, 3):
        planner.update_depth(camera(np.full((4, 4), 600), Pose(1, 1, 0), seq), calibration())
    assert not planner.dynamic.any()


def test_projection_keeps_top_of_runtime_collision_height():
    info = calibration()
    info["T_base_camera"][2][3] = 0.088
    assert len(project_obstacles(camera([[1000]]), info)) == 1
    assert len(project_obstacles(camera([[1000]]), info, height=0.087)) == 0


def test_motion_checks_requested_arc_and_incoming_braking_distance():
    grid = MapGrid(np.zeros((150, 150), dtype=bool), 0.02, (0, 0))
    grid.occupied[-1, :] = True
    planner = GridPlanner(grid)
    pose = Pose(0.5, 0.1921, -0.2913113497618344)
    assert planner.segment_clear((pose.x, pose.y), (1.5, 0.5))
    assert not planner.motion_clear(pose, 0.166188, 1.3, velocity=(0.166188, 1.3))
    assert planner.motion_clear(pose, 0, 1.0)
    assert not planner.motion_clear(pose, 0, 0, velocity=(0.20, 0))
