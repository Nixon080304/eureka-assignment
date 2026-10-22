import copy
import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, queue_size=128, *, make_video=True):
        if queue_size < 1:
            raise ValueError("Recording queue size must be positive")
        self.queue_size = queue_size
        self.make_video = make_video
        self.thread = None
        self.lock = threading.Lock()
        self._status = {"frames": 0, "states": 0, "dropped": 0, "dropped_states": 0}
        self._status.update(active=False, path=None, error=None, video="not_started")

    def start(self, directory, metadata):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Recording is already active or finalizing")
        name = time.strftime("session_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.folder = Path(directory).expanduser() / name
        (self.folder / "rgb").mkdir(parents=True)
        (self.folder / "depth").mkdir()
        self.metadata = copy.deepcopy(metadata)
        self.queue = queue.Queue(self.queue_size)
        self.finished = threading.Event()
        self._status = {"frames": 0, "states": 0, "dropped": 0, "dropped_states": 0}
        self._status.update(
            active=True,
            path=str(self.folder),
            error=None,
            video="pending" if self.make_video else "disabled",
        )
        self.thread = threading.Thread(target=self._write, name="rgbd-writer", daemon=True)
        self.thread.start()
        return self.folder

    def offer(self, packet):
        return self._offer("frame", packet)

    def offer_state(self, state):
        self._offer("state", state)

    def _offer(self, kind, value):
        with self.lock:
            if not self._status["active"]:
                return False
            try:
                self.queue.put_nowait((kind, copy.deepcopy(value)))
                return True
            except queue.Full:
                key = "dropped" if kind == "frame" else "dropped_states"
                self._status[key] += 1
                self._status["error"] = "Recording queue full. Dropped " + kind
                return False

    def stop(self):
        if self.thread is not None:
            with self.lock:
                self._status["active"] = False
                self.finished.set()
            self.thread.join()

    def status(self):
        with self.lock:
            return dict(self._status)

    def _write(self):
        clean_close = False
        try:
            self._manifest(False)
            with (
                (self.folder / "frames.jsonl").open("w") as frames,
                (self.folder / "state.jsonl").open("w") as states,
            ):
                while not self.finished.is_set() or not self.queue.empty():
                    try:
                        kind, value = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if kind == "state":
                        states.write(json.dumps(value, allow_nan=False) + "\n")
                        states.flush()
                        self._status["states"] += 1
                    else:
                        if (value.header["width"], value.header["height"]) != (
                            self.metadata["camera"]["width"],
                            self.metadata["camera"]["height"],
                        ):
                            raise ValueError("Frame dimensions differ from camera calibration")
                        depth = value.depth_array()
                        index = self._status["frames"]
                        rgb_name = f"rgb/{index:08d}.jpg"
                        depth_name = f"depth/{index:08d}.npz"
                        (self.folder / rgb_name).write_bytes(value.jpeg)
                        np.savez_compressed(self.folder / depth_name, depth=depth)
                        row = dict(index=index, header=value.header, rgb=rgb_name, depth=depth_name)
                        row.update(t=value.t, seq=value.seq, pose=value.header["pose"])
                        row["received_ns"] = value.received_ns
                        frames.write(json.dumps(row, allow_nan=False) + "\n")
                        frames.flush()
                        self._status["frames"] += 1
            clean_close = True
            if self.make_video:
                self._video()
        except Exception as error:
            self._status["error"] = str(error)
            if clean_close and self.make_video:
                self._status["video"] = "failed"
        finally:
            with self.lock:
                self._status["active"] = False
            try:
                self._manifest(clean_close)
            except Exception as error:
                self._status["error"] = "Cannot save recording manifest: " + str(error)

    def _manifest(self, clean_close):
        manifest = {"schema": "eureka.rgbd.simple", "metadata": self.metadata}
        manifest.update(clean_close=clean_close, counts=self.status())
        temporary = self.folder / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        temporary.replace(self.folder / "manifest.json")

    def _video(self):
        if not self._status["frames"]:
            self._status["video"] = "empty"
            return
        self._status["video"] = "encoding"
        command = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", "15"]
        command += ["-i", str(self.folder / "rgb/%08d.jpg")]
        command += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264"]
        command += ["-pix_fmt", "yuv420p", str(self.folder / "rgb.mp4")]
        with (self.folder / "video.log").open("wb") as log:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=log,
                timeout=120,
                check=True,
            )
        self._status["video"] = "rgb.mp4"
