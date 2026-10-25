# Eureka robot vacuum control

A Python application for the Eureka Robotics robot vacuum simulator. It plans paths with A*, drives the robot using wheel commands, and uses depth images to detect obstacles missing from the supplied map. The Qt interface shows the map, robot trajectory, goals, planned path, and RGBD camera previews.

## Demo

The 4-minute demo shows navigation to one goal in each scene, `01`–`06`, using seed `18`.

https://github.com/user-attachments/assets/f979e5b6-ffc9-45c1-a68a-06e0cd5773a1

[View the MP4 file](deliverables/demo.mp4).

## Requirements

- Python 3.10 or newer.
- uv for Python dependency management.
- Docker with at least 2 CPUs available to the simulator.
- A desktop session for the Qt interface.
- FFmpeg with the `libx264` encoder available on your `PATH` for MP4 export.

The simulator supports Linux and macOS. These instructions use a Unix shell. Running the GUI over SSH requires display forwarding or a remote desktop.

## Install

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/Nixon080304/eureka-assignment.git
cd eureka-assignment
uv sync --locked
```

This creates `.venv` and installs the application using the versions in `uv.lock`. There is no separate compilation step to run.

Check that FFmpeg is available:

```bash
ffmpeg -version
```

## Run

Start the simulator in one terminal:

```bash
docker pull ghcr.io/eurekarobotics/vacuum-sim:1.0.0
docker run --rm --name eureka-sim \
  -p 127.0.0.1:5555-5558:5555-5558 \
  ghcr.io/eurekarobotics/vacuum-sim:1.0.0 --scene 01 --seed 18
```

In another terminal, open the project directory and start the application:

```bash
uv run eureka
```

The application connects to `localhost` on ports `5555` through `5558`. At startup, it loads scene `01` with seed `18`, even if the simulator started with a different scene or seed. Use one controlling application per simulator.

If those ports are already in use by the simulator, use that instance instead of starting a second one. If the application starts before the simulator is ready, click **Load scene** after the simulator starts.

## Use the application

1. Select a scene and seed, then click **Load scene**. Loading resets the robot and clears the previous trajectory and reached goals.
2. Select a named goal and click **Navigate**.
3. Wait for the initial obstacle scan, path planning, and movement. Before its first route after a scene load, the robot turns almost a full circle to scan its surroundings.
4. Check the status text. A successful arrival shows `arrived`, and the goal receives a check mark after the application calls `mark_goal`.
5. Click **STOP** to cancel navigation and stop the robot. Navigation can be started again by selecting a goal and clicking **Navigate**.

The map uses grey for supplied obstacles, orange for detected obstacles, blue for the robot trajectory, and teal for the planned path. Goal markers change colour when marked reached.

## Record the camera

Enter an output directory in the recording field, then click **Start recording**. The default directory is `recordings`, relative to where you launched the application. You can record while the robot moves.

Click **Stop recording** to finish the session. This also stops navigation. Allow FFmpeg to finish exporting before opening the MP4 or starting another recording. Export can take some time. Check the recording status for errors and the output path.

Each session contains:

| File or directory | Contents |
| --- | --- |
| `rgb.mp4` | RGB preview video encoded at a fixed 15 frames per second. |
| `rgb/` | Original RGB JPEG frames. |
| `depth/` | Lossless NumPy `.npz` files containing a `depth` array of `uint16` values. |
| `frames.jsonl` | Frame filenames, capture timestamps, sequence numbers, capture poses, original headers, and host receive timestamps. |
| `state.jsonl` | Received robot state messages, including pose, encoder ticks, and bumper state. |
| `manifest.json` | Scene, seed, camera calibration, robot and map metadata, source hashes, recording counts, and close status. |
| `video.log` | FFmpeg errors, if any. |

For offline processing, use the original frames and `frames.jsonl` for timing. The fixed-rate MP4 does not preserve gaps between captures. Capture time `t` is simulation time in seconds since the last scene load or reset. `received_ns` uses the host's monotonic clock in nanoseconds and is not a wall-clock timestamp.

Read the camera calibration from `manifest.json` under `metadata.camera`. Multiply depth values by `depth_scale` to obtain metres. Zero depth means invalid or out of range. Depth measures distance along the camera's optical axis. There is no depth video; the depth arrays retain the original measurements.

## Test manually

Run the following checks with the simulator and application open:

1. Load scene `01` with seed `18`. Confirm that the map, robot pose, named goals, RGB preview, and depth preview appear.
2. Select `kitchen` and click **Navigate**. Confirm that a path appears, the robot moves, and the status eventually shows `arrived` with a check mark beside the goal. Check the simulator terminal for goal marks and collisions as well.
3. Load the scene again, start navigation, and click **STOP** while the robot moves. Confirm that the robot stops and the status returns to `idle`.
4. Start recording, navigate to a goal, and stop recording. Play `rgb.mp4` and check that the session contains RGB frames, depth arrays, timestamps, state messages, and calibration metadata. Check `manifest.json` for `clean_close: true` and review its counts and errors.
5. Repeat navigation for the other goals on scenes `01-06`. Reload with the same seed to repeat a start pose. Try different seeds to check other starts. Record the scene, seed, goal, arrival result, collisions, and any failure message.

The assignment's arrival limits are 5 cm in position and 5 degrees for goals with a required yaw. A goal check mark confirms that the application sent `mark_goal`; use the simulator's records to assess the arrival and collisions independently.

## Libraries and tools

- NumPy handles arrays, depth images, and coordinate calculations.
- SciPy provides distance transforms, spatial queries, and obstacle evidence filtering.
- Pillow decodes map images.
- PyZMQ communicates with the simulator over ZeroMQ.
- PySide6 provides the Qt interface.
- FFmpeg exports RGB frames as H.264 video.
- uv manages Python dependencies and the virtual environment. Hatchling builds the Python package during installation.
- Docker runs the supplied simulator. Git tracks the source code, and GitHub hosts the repository.

## AI assistance

I used OpenAI Codex to brainstorm motion planning and control algorithms, review and validate the code whether it matches the requirements, and prepare documentation.
  