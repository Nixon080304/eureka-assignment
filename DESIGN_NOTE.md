# Eureka robot vacuum control: design note

## Architecture

The application uses Python, NumPy, SciPy, and Qt. It communicates only through the simulator's public ZeroMQ interface. A 2D representation matches the disc-shaped robot and supplied occupancy map.

Python makes the protocol, numerical routines, recording, and Qt interface practical within the assignment's time limit. NumPy and SciPy provide compiled numerical operations. Python scheduling, allocation, and blocking work give no hard real-time guarantee. The application requests commands at 20 Hz; the simulator owns its 100 Hz dynamics loop.

The source separates the main responsibilities:

- `app.py` starts the application and closes it on exit.
- `transport.py` owns the ZeroMQ protocol, message decoding, and RPC timeouts.
- `domain.py` defines poses, goals, map coordinates, camera packets, and UI snapshots.
- `planning.py` handles obstacle mapping, A* search, and collision checks.
- `controller.py` follows waypoints, checks motion safety, handles recovery, and verifies arrival tolerances.
- `runtime.py` receives sensor data, estimates pose, runs planning, sends wheel commands, and marks arrivals.
- `recording.py` writes camera and state data.
- `ui.py` displays snapshots and submits user commands.

The Qt thread handles the interface. One runtime thread owns the simulator sockets and calls the motion controller. The controller receives observations and returns wheel commands or planning and marking requests. A `Mode` Enum names navigation states. Public episode, contact, path-result, and arrival methods keep navigation timers and recovery state inside the controller. The UI reads snapshots. A recorder thread handles disk writes.

## Planning and obstacle detection

The planner combines the supplied map with confirmed depth observations. Obstacles expand by the robot radius, a 3 cm safety margin, and a cell-size allowance. Eight-direction A* searches positions where the robot centre fits. Diagonal checks prevent cutting blocked corners.

A clear straight route avoids grid search. A* uses travelled distance and a Euclidean distance estimate. Smoothing removes intermediate points only after a swept-disc check. Subdivision limits waypoint spacing to 0.5 m and preserves the exact endpoint.

Calibrated depth pixels become world points using the capture pose, interpolated from pose history when available. Only points in the collision height band contribute evidence. Confirmation requires multiple cell returns and nearby support from an earlier capture. This reduces noise but delays detection. Confirmed obstacles remain until scene reload.

An initial rotation gathers observations. Newly detected obstacles can trigger replanning. Distance transforms and a spatial tree are cached until the map changes. Conservative clearance can reject physically traversable passages.

## Motion control and robustness

The runtime combines wheel-encoder odometry with gradual correction toward the reported pose to reduce jitter. The controller follows waypoints using heading error and distance, reducing speed during turns. Commands become differential wheel speeds using published robot dimensions and limits.

Wheel commands are requested at 20 Hz. The simulator applies its own 100 Hz kinematics and acceleration limits. Normal forward speed is capped at 0.20 m/s, below the published maximum. Before forward or retreat motion, the planner checks a predicted arc and braking trajectory using estimated velocity, acceleration limits, and the watchdog interval.

Navigation stops on stale streams or transport faults. Bumper recovery allows only a short, checked retreat along previously visited space; otherwise it scans without translation. Repeated contact eventually stops navigation. Arrival requires stable position and, where specified, heading before `mark_goal`. A mark request with an uncertain outcome is not automatically repeated.

## Recording and visualisation

The recorder uses a bounded queue to keep normal disk writes out of control updates. Overflow reports an error and drop counts. Sessions retain RGB JPEGs, lossless depth, timestamps, poses, state messages, and calibration for offline processing.

FFmpeg exports a fixed 15 fps RGB MP4 after recording. Actual timing remains in `frames.jsonl`; lossless depth arrays require an offline reader.

The UI shows the map, detected obstacles, trajectory, goals, planned path, and RGBD previews. Text states distinguish scanning, planning, following, arrival, and faults.

## Assumptions and known limitations

The design assumes a static environment within each scene, valid calibration, and the simulator's published frame and depth conventions. It uses the supplied pose stream as a correction source rather than performing SLAM. The forward camera cannot observe all surroundings at once, and false obstacle confirmations cannot currently be cleared.

Planning runs synchronously in the runtime thread with the robot stopped. Large searches and repeated smoothing checks delay control processing. Recording finalisation also waits for video export. Initial scanning adds startup delay, and the waypoint follower can pause or turn sharply near corners. Conservative margins and limited recovery do not guarantee arrival at every reachable goal.

Pytest covers planning geometry and mapping, controller safety decisions, and runtime stop, reset, delivery, and marking failures. Mypy checks all source modules. Manual simulator checks remain necessary for timing, camera streams, and GUI behavior. A verified scene `01`, seed `18` run reached `kitchen` without observed bumper events or recording queue drops, and completed stop, export, and shutdown. That run does not establish coverage across all scenes or starts.

## With more time

With more time, I would adapt the earlier prototype's headless validation runner to save repeatable scene, goal, and seed results. I would also improve corner following and add automatic recording archives.
