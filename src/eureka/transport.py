"""A step-by-step version of transport.py with the same simulator interface."""

import io
import json
import math
import time

import numpy as np
import zmq
from PIL import Image

from .domain import CameraPacket, MapGrid, Pose


class RpcError(RuntimeError):
    pass


class Simulator:
    def __init__(self, host="localhost", port_base=5555):
        self.host = host
        self.port_base = port_base
        self.context = zmq.Context()

        self.rpc_socket = None
        self.state_socket = None
        self.camera_socket = None

        self.command = self.context.socket(zmq.PUSH)
        self.command.setsockopt(zmq.LINGER, 0)
        self.command.setsockopt(zmq.SNDHWM, 1)
        self.command.setsockopt(zmq.IMMEDIATE, 1)
        self.command.connect(self.url(1))

        self._new_rpc()
        self.restart_streams()

    def url(self, offset):
        port = self.port_base + offset
        return f"tcp://{self.host}:{port}"

    def _new_rpc(self):
        if self.rpc_socket is not None:
            self.rpc_socket.close(0)

        self.rpc_socket = self.context.socket(zmq.REQ)
        self.rpc_socket.setsockopt(zmq.LINGER, 0)
        self.rpc_socket.setsockopt(zmq.SNDTIMEO, 1500)
        self.rpc_socket.connect(self.url(0))

    def call(self, method, timeout=1500, **params):
        request = {"method": method, "params": params}

        try:
            self.rpc_socket.send_json(request)

            if not self.rpc_socket.poll(timeout):
                raise zmq.Again()

            frames = self.rpc_socket.recv_multipart()
            header = json.loads(frames[0])

            if not header.get("ok"):
                message = str(header.get("error", "RPC failed"))
                raise RpcError(message)

            result = header["result"]
            payloads = frames[1:]
            return result, payloads

        except (zmq.ZMQError, ValueError, KeyError, IndexError) as error:
            self._new_rpc()
            message = f"{method} failed or timed out; outcome may be unknown"
            raise RpcError(message) from error

    def restart_streams(self):
        if self.state_socket is not None:
            self.state_socket.close(0)
        if self.camera_socket is not None:
            self.camera_socket.close(0)

        self.state_socket = self.context.socket(zmq.SUB)
        self.camera_socket = self.context.socket(zmq.SUB)

        self.state_socket.setsockopt(zmq.LINGER, 0)
        self.state_socket.setsockopt(zmq.RCVHWM, 200)
        self.state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.state_socket.connect(self.url(2))

        self.camera_socket.setsockopt(zmq.LINGER, 0)
        self.camera_socket.setsockopt(zmq.RCVHWM, 48)
        self.camera_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.camera_socket.connect(self.url(3))

    def wheels(self, left=0.0, right=0.0):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Wheel command contains a non-finite value")

        message = {"type": "wheels", "left": left, "right": right}

        try:
            self.command.send_json(message, flags=zmq.DONTWAIT)
            return True
        except zmq.Again:
            return False

    def states(self):
        messages = []

        for _ in range(250):
            try:
                frames = self.state_socket.recv_multipart(zmq.DONTWAIT)
            except zmq.Again:
                break

            state = json.loads(frames[0])
            Pose.from_dict(state["pose"])

            state_time = float(state["t"])
            if not math.isfinite(state_time):
                raise ValueError("State time is not finite")

            # Check numeric fields while preserving the original state message.
            int(state["seq"])
            int(state["wheels"]["left_ticks"])
            int(state["wheels"]["right_ticks"])

            messages.append(state)

        return messages

    def cameras(self):
        packets = []

        for _ in range(24):
            try:
                frames = self.camera_socket.recv_multipart(zmq.DONTWAIT)
            except zmq.Again:
                break

            if len(frames) != 3:
                raise ValueError("Camera message must have three frames")

            header = json.loads(frames[0])
            width = int(header["width"])
            height = int(header["height"])

            if not (1 <= width <= 4096 and 1 <= height <= 4096):
                raise ValueError("Camera dimensions are invalid")

            rgb = frames[1]
            depth = frames[2]
            expected_bytes = width * height * 2
            if len(depth) != expected_bytes:
                raise ValueError("Camera depth payload length does not match dimensions")

            Pose.from_dict(header["pose"])
            capture_time = float(header["t"])
            if not math.isfinite(capture_time):
                raise ValueError("Camera time is not finite")

            received_ns = time.monotonic_ns()
            packet = CameraPacket(header, rgb, depth, received_ns)
            packets.append(packet)

        return packets

    def get_map(self):
        metadata, frames = self.call("get_map")
        if len(frames) != 1:
            raise ValueError("Map response must contain one PNG")

        image_bytes = io.BytesIO(frames[0])
        image = Image.open(image_bytes)
        grayscale = image.convert("L")
        pixels = np.array(grayscale)

        resolution = float(metadata["resolution"])
        expected_shape = (metadata["height"], metadata["width"])
        if pixels.shape != expected_shape or resolution <= 0:
            raise ValueError("Map dimensions or resolution are invalid")

        occupied = pixels > 127
        occupied.flags.writeable = False

        origin = []
        for coordinate in metadata["origin"]:
            origin.append(float(coordinate))

        grid = MapGrid(occupied, resolution, tuple(origin))
        return grid, metadata, frames[0]

    def close(self):
        self.wheels()

        sockets = (self.rpc_socket, self.command, self.state_socket, self.camera_socket)
        for socket in sockets:
            if socket is not None:
                socket.close(0)

        self.context.term()
