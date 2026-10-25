# Eureka robot vacuum control: design note

## Architecture

The application uses Python, NumPy, SciPy, and Qt. It communicates only through the simulator's public ZeroMQ interface. A 2D representation matches the disc-shaped robot and supplied occupancy map.

The source separates the main responsibilities:

- `app.py` starts the application and closes it on exit.
- `transport.py` owns the ZeroMQ protocol, message decoding, and RPC timeouts.
- `domain.py` defines poses, goals, map coordinates, camera packets, and UI snapshots.
- `planning.py` handles obstacle mapping, A* search, and collision checks.
- `controller.py` estimates pose, follows waypoints, handles navigation states, and marks arrivals.
- `recording.py` writes camera and state data.
- `ui.py` displays snapshots and submits user commands.

The Qt thread handles the interface. One controller thread owns the simulator sockets and navigation state. The UI submits queued commands and reads snapshots. A separate recorder thread handles disk writes.

## Planning and obstacle detection

The planner combines the supplied map with depth observations. It expands obstacles by the robot radius, a 3 cm safety margin, and an allowance for cell size. A* then searches positions where the robot centre can fit. Using eight-direction movement gives shorter routes than four-direction movement, while diagonal checks prevent cutting across blocked corners.

A clear straight route avoids grid search. Otherwise, A* uses travelled distance and a Euclidean distance estimate. Smoothing removes intermediate points only when the connecting segment passes a swept-disc collision check. Subdivision limits waypoint spacing to 0.5 m and preserves the requested endpoint.

Depth pixels are projected through camera calibration into robot and world coordinates using the capture pose. The controller interpolates pose history when available. Only points in the robot's collision height band contribute evidence. Confirmation requires multiple returns in a cell and nearby support from an earlier capture. This reduces noise but delays detection. Confirmed obstacles remain until the scene reloads.

An initial rotation gathers observations before the first route. Obstacles obstructing the next segment can trigger replanning. Distance transforms and a spatial tree are cached until the map changes. Conservative clearance can reject passages that the physical robot could traverse.

## Motion control and robustness

The controller combines wheel-encoder odometry with gradual correction toward the reported pose to reduce jitter. It follows waypoints using heading error and distance, reducing speed during turns. Commands become differential wheel speeds using published robot dimensions and limits.

Wheel commands are requested at 20 Hz. The simulator applies its own 100 Hz kinematics and acceleration limits. Normal forward speed is capped at 0.20 m/s, below the published maximum. Before forward or retreat motion, the planner checks a predicted arc and braking trajectory using estimated velocity, acceleration limits, and the watchdog interval.

Navigation stops on stale streams or transport faults. Bumper recovery allows only a short, checked retreat along previously visited space; otherwise it scans without translation. Repeated contact eventually stops navigation. Arrival requires stable position and, where specified, heading before `mark_goal`. A mark request with an uncertain outcome is not automatically repeated.

## Recording and visualisation

The recorder uses a bounded queue so normal disk writes do not block control. Overflow increments drop counts and reports an error. Sessions keep original RGB JPEGs, lossless depth arrays, timestamps and poses, state messages, and calibration metadata for offline processing.

FFmpeg produces an RGB MP4 at a fixed 15 frames per second after recording stops. Actual timing remains in `frames.jsonl`. Depth arrays preserve measurement precision, but require an offline reader.

The UI shows the map, detected obstacles, trajectory, goals, planned path, and RGBD previews. Text states distinguish scanning, planning, following, arrival, and faults.

## Assumptions and known limitations

The design assumes a static environment within each scene, valid calibration, and the simulator's published frame and depth conventions. It uses the supplied pose stream as a correction source rather than performing SLAM. The forward camera cannot observe all surroundings at once, and false obstacle confirmations cannot currently be cleared.

Planning runs synchronously in the controller thread with the robot stopped. Large searches and repeated smoothing checks delay control processing. Recording finalisation also waits for video export. Initial scanning adds startup delay, and the waypoint follower can pause or turn sharply near corners. Conservative margins and limited recovery do not guarantee arrival at every reachable goal.

A verified Eureka demo reached `kitchen` on scene `01`, seed `18`, with no observed bumper events or recording queue drops. Stop, video export, and shutdown also completed. This single run does not establish coverage across all scenes or starts. The repository currently provides manual test instructions rather than an automated test suite.

## With more time

With more time, I would adapt the earlier prototype’s headless validation runner into a separate test script. It would run repeatable scene, goal, and seed combinations, then save navigation and recording results without requiring someone to watch the GUI. I would also improve corner following to be smoother. Automatic zip file creation would also be useful.