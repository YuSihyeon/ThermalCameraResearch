"""Depth replay inputs for deterministic TEED bench and replay runs.

Depth values use metres and timestamps use seconds.  ``.npy`` inputs are dense
``[N, H, W]`` float32/float64 depth maps, one map per video frame.  External
depth maps must already be spatially registered to the video frames; this
module deliberately does not resize or align them.

The ``teed_depth_scalar_v1`` JSON protocol carries one global scalar distance
for bench/replay use.  It does not represent per-pixel or multi-object depth.
"""

from bisect import bisect_right
import json
import math
import numbers
import os

import numpy as np


_SCALAR_PROTOCOL = "teed_depth_scalar_v1"


def _is_real_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _validate_frame_shape(frame_shape):
    if not isinstance(frame_shape, (tuple, list)) or len(frame_shape) != 2:
        raise ValueError("frame_shape must be a (height, width) pair")
    dimensions = []
    for dimension in frame_shape:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, numbers.Integral):
            raise TypeError("frame_shape dimensions must be integers")
        if dimension <= 0:
            raise ValueError("frame_shape dimensions must be positive")
        dimensions.append(int(dimension))
    return tuple(dimensions)


def _validate_fps(fps):
    if not _is_real_number(fps):
        raise TypeError("fps must be a number")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    return fps


class DepthReplay(object):
    """Read registered depth maps or timestamped scalar depth in metres.

    Args:
        path: ``.npy`` depth maps or a ``teed_depth_scalar_v1`` JSON file.
        frame_shape: Video frame ``(height, width)`` in pixels.  Numpy depth
            maps must match it exactly and must already be registered.
        fps: Video frames per second, used to convert a frame index to seconds.

    ``sample(frame_index)`` returns ``(depth, depth_timestamp_s)``.  Dense NPY
    frames use ``frame_index / fps`` as their timestamp.  JSON lookup returns
    the latest causal sample and keeps that sample's original timestamp so the
    renderer can apply its own maximum-age policy.
    """

    def __init__(self, path, frame_shape, fps):
        self.frame_shape = _validate_frame_shape(frame_shape)
        self.fps = _validate_fps(fps)
        self.path = os.fspath(path)

        extension = os.path.splitext(self.path)[1].lower()
        if extension == ".npy":
            self._load_npy()
        elif extension == ".json":
            self._load_json()
        else:
            raise ValueError("unsupported depth replay format: {0}".format(extension or "<none>"))

    def _load_npy(self):
        frames = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if frames.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("NPY depth maps must use float32 or float64 metres")
        if frames.ndim != 3:
            raise ValueError("NPY depth maps must have shape [N, H, W]")
        if tuple(frames.shape[1:]) != self.frame_shape:
            raise ValueError(
                "NPY depth map size {0} does not match frame_shape {1}; maps must already be registered".format(
                    tuple(frames.shape[1:]), self.frame_shape
                )
            )
        self._kind = "npy"
        self._frames = frames
        self._times = None
        self._depths = None

    def _load_json(self):
        with open(self.path, "r", encoding="utf-8") as input_file:
            document = json.load(input_file)
        if not isinstance(document, dict):
            raise ValueError("depth replay JSON must contain an object")
        if document.get("protocol") != _SCALAR_PROTOCOL:
            raise ValueError("unknown depth replay protocol")
        samples = document.get("samples")
        if not isinstance(samples, list):
            raise ValueError("depth replay JSON samples must be a list")

        times = []
        depths = []
        previous_time = None
        for sample in samples:
            if not isinstance(sample, dict) or "time_s" not in sample or "depth_m" not in sample:
                raise ValueError("each depth sample must contain time_s and depth_m")

            time_s = sample["time_s"]
            if not _is_real_number(time_s):
                raise ValueError("sample time_s must be a number, not bool")
            time_s = float(time_s)
            if not math.isfinite(time_s) or time_s < 0.0:
                raise ValueError("sample time_s must be nonnegative and finite")
            if previous_time is not None and time_s <= previous_time:
                raise ValueError("sample timestamps must be strictly increasing")

            depth_m = sample["depth_m"]
            if depth_m is not None:
                if not _is_real_number(depth_m):
                    raise ValueError("sample depth_m must be numeric or null, not bool")
                depth_m = float(depth_m)
                if not math.isfinite(depth_m) or depth_m <= 0.0:
                    raise ValueError("sample depth_m must be positive and finite")

            times.append(time_s)
            depths.append(depth_m)
            previous_time = time_s

        self._kind = "json"
        self._frames = None
        self._times = times
        self._depths = depths

    def sample(self, frame_index):
        """Return depth in metres and its timestamp in seconds for a frame."""
        if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, numbers.Integral):
            raise TypeError("frame_index must be an integer")
        if frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        frame_index = int(frame_index)

        if self._kind == "npy":
            if frame_index >= self._frames.shape[0]:
                return None, None
            return self._frames[frame_index], frame_index / self.fps

        current_time_s = frame_index / self.fps
        sample_index = bisect_right(self._times, current_time_s) - 1
        if sample_index < 0:
            return None, None
        return self._depths[sample_index], self._times[sample_index]
