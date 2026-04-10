from __future__ import annotations

import struct
from dataclasses import dataclass
from multiprocessing import shared_memory
import numpy as np


_HEADER_FMT = "QQIII"  # frame_id, timestamp_ns, height, width, channels
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


@dataclass
class FrameSpec:
    name: str
    height: int
    width: int
    channels: int = 3
    dtype: np.dtype = np.uint8

    @property
    def nbytes(self) -> int:
        return _HEADER_SIZE + self.height * self.width * self.channels * np.dtype(self.dtype).itemsize


class SharedFrameWriter:
    def __init__(self, spec: FrameSpec, create: bool = True):
        self.spec = spec
        self.shm = shared_memory.SharedMemory(name=spec.name, create=create, size=spec.nbytes)
        self.buf = self.shm.buf

    def write(self, frame: np.ndarray, frame_id: int, timestamp_ns: int):
        assert frame.shape == (self.spec.height, self.spec.width, self.spec.channels), (
            frame.shape, self.spec
        )
        header = struct.pack(
            _HEADER_FMT,
            int(frame_id),
            int(timestamp_ns),
            int(self.spec.height),
            int(self.spec.width),
            int(self.spec.channels),
        )
        self.buf[:_HEADER_SIZE] = header
        np.frombuffer(self.buf[_HEADER_SIZE:], dtype=self.spec.dtype).reshape(
            self.spec.height, self.spec.width, self.spec.channels
        )[:] = frame

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


class SharedFrameReader:
    def __init__(self, spec: FrameSpec):
        self.spec = spec
        self.shm = shared_memory.SharedMemory(name=spec.name, create=False)
        self.buf = self.shm.buf

    def read(self):
        frame_id, timestamp_ns, h, w, c = struct.unpack(_HEADER_FMT, self.buf[:_HEADER_SIZE])
        frame = np.frombuffer(self.buf[_HEADER_SIZE:], dtype=self.spec.dtype).reshape(h, w, c).copy()
        return frame_id, timestamp_ns, frame

    def close(self):
        self.shm.close()