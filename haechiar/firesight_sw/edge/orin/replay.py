"""Replay input layer.

Turns any supported source into a stream of Frame objects plus honest metadata
about where those frames came from. Nothing downstream needs to know whether
the pixels arrived from an mp4, a still image, a USB camera or a Boson Y16 node.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")

# Burned-in overlay boxes in the Fire360 material, as normalized
# (left, top, right, bottom). Left: agency logo. Right: temperature colorbar.
FIRE360_OSD_REGIONS = (
    (0.13, 0.72, 0.27, 1.0),
    (0.75, 0.0, 0.90, 0.90),
)
FIRE360_PROFILE_PROTOCOL = "fire360_video_profile_v1"


class ReplayError(RuntimeError):
    """Raised when a replay source cannot be opened or decoded."""


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # HxWx3 uint8 BGR
    index: int
    source_seconds: float  # position within the source timeline
    pixel_origin: str  # bgr8 | gray8 | y16
    discontinuity: bool = False


@dataclass
class SourceInfo:
    """What the source actually is. Never guessed, never hardcoded."""

    kind: str  # video | image | camera | v4l2
    uri: str
    width: int
    height: int
    fps: float  # 0.0 when the source does not report one
    frame_count: int  # 0 when unknown or live
    pixel_origin: str = "unknown"
    osd_regions: tuple = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "uri": self.uri,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "pixel_origin": self.pixel_origin,
            "osd_regions": len(self.osd_regions),
        }


def load_fire360_osd_regions(profile_path: Optional[str], video_key: str) -> tuple:
    """Return OSD regions when the Fire360 profile marks this clip as needing them."""
    if not profile_path:
        return ()
    path = Path(profile_path)
    if not path.is_file():
        raise ReplayError(f"Fire360 profile not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != FIRE360_PROFILE_PROTOCOL:
        raise ReplayError(f"Unsupported Fire360 profile: {path}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ReplayError("Fire360 profile has no profiles list")
    wanted = os.path.basename(video_key).lower()
    for entry in profiles:
        if not isinstance(entry, dict):
            raise ReplayError("Fire360 profile contains an invalid entry")
        input_file = entry.get("input_file")
        ignores = entry.get("ignore_fire360_osd")
        if not isinstance(input_file, str) or not isinstance(ignores, bool):
            raise ReplayError("Fire360 profile entry is missing required fields")
        if os.path.basename(input_file).lower() == wanted:
            return FIRE360_OSD_REGIONS if ignores else ()
    return ()


def unpack_y16(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return a 2D uint16 array when the buffer really is 16-bit, else None.

    A uint8 two-channel buffer is deliberately NOT reinterpreted here: YUYV has
    the same shape and would be silently misread as thermal data.
    """
    array = np.asarray(frame)
    if array.dtype == np.uint16:
        if array.ndim == 3 and array.shape[2] == 1:
            return array[:, :, 0]
        if array.ndim == 2:
            return array
    return None


def to_bgr(
    frame: np.ndarray,
    expected: str = "auto",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> tuple[np.ndarray, str]:
    """Normalize any decoded buffer into HxWx3 uint8 BGR."""
    if expected in ("auto", "y16"):
        mono16 = unpack_y16(frame)
        if mono16 is not None:
            low, high = np.percentile(mono16, [low_percentile, high_percentile])
            high = max(float(high), float(low) + 1.0)
            mono8 = np.clip(
                (mono16.astype(np.float32) - low) * 255.0 / (high - low), 0, 255
            ).astype(np.uint8)
            return cv2.cvtColor(mono8, cv2.COLOR_GRAY2BGR), "y16"
        if expected == "y16":
            raise ReplayError(
                "y16 was requested but the source did not deliver a 16-bit buffer"
            )
    array = np.asarray(frame)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR), "gray8"
    if array.ndim == 3 and array.shape[2] == 1:
        return cv2.cvtColor(array[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2BGR), "gray8"
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3].astype(np.uint8), "bgr8"
    raise ReplayError(f"Unsupported frame shape: {array.shape}")


class ReplaySource:
    """Iterable frame source with optional real-time pacing and looping."""

    def __init__(
        self,
        uri: str,
        input_format: str = "auto",
        capture_width: int = 0,
        capture_height: int = 0,
        capture_fps: float = 0.0,
        fourcc: str = "",
        loop: bool = False,
        pace: bool = False,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        profile_path: Optional[str] = None,
        video_key: Optional[str] = None,
        stereo_half: Optional[str] = None,
        warmup_seconds: float = 5.0,
    ) -> None:
        self.uri = uri
        self.input_format = input_format
        # OV580-class modules deliver both eyes side by side in one frame.
        if stereo_half not in (None, "left", "right"):
            raise ReplayError("stereo_half must be left or right")
        self.stereo_half = stereo_half
        self.warmup_seconds = float(warmup_seconds)
        self.fourcc = fourcc
        self.loop = bool(loop)
        self.pace = bool(pace)
        self.start_frame = max(0, int(start_frame))
        self.max_frames = max_frames
        self._capture: Optional[cv2.VideoCapture] = None
        self._still: Optional[np.ndarray] = None
        self._still_paths: list[str] = []

        suffix = os.path.splitext(uri)[1].lower()
        is_device = uri.startswith("/dev/video") or uri.isdigit()
        if not is_device and os.path.isdir(uri):
            self._still_paths = sorted(
                str(path)
                for path in Path(uri).iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self._still_paths:
                raise ReplayError(f"No images found in directory: {uri}")
            probe = cv2.imread(self._still_paths[0], cv2.IMREAD_UNCHANGED)
            if probe is None:
                raise ReplayError(f"Unable to read image: {self._still_paths[0]}")
            height, width = probe.shape[:2]
            self.info = SourceInfo(
                "image_sequence", uri, width, height,
                float(capture_fps or 0.0), len(self._still_paths),
            )
        elif suffix in IMAGE_SUFFIXES and not is_device:
            still = cv2.imread(uri, cv2.IMREAD_UNCHANGED)
            if still is None:
                raise ReplayError(f"Unable to read image: {uri}")
            self._still = still
            height, width = still.shape[:2]
            self.info = SourceInfo("image", uri, width, height, 0.0, 1)
        else:
            self._capture, kind = self._open_capture(
                uri, is_device, capture_width, capture_height, capture_fps,
                self.fourcc,
            )
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
            count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.info = SourceInfo(kind, uri, width, height, fps, max(count, 0))
            if self.start_frame and kind == "video":
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.info.osd_regions = load_fire360_osd_regions(
            profile_path, video_key or os.path.basename(uri)
        )

    def _open_capture(
        self,
        uri: str,
        is_device: bool,
        capture_width: int,
        capture_height: int,
        capture_fps: float,
        fourcc: str = "",
    ) -> tuple[cv2.VideoCapture, str]:
        if uri.isdigit():
            capture = cv2.VideoCapture(int(uri))
            kind = "camera"
        elif uri.startswith("/dev/video"):
            capture = cv2.VideoCapture(uri, cv2.CAP_V4L2)
            kind = "v4l2"
        else:
            capture = cv2.VideoCapture(uri)
            kind = "video"
        if not capture.isOpened():
            raise ReplayError(f"Unable to open source: {uri}")
        if is_device:
            if fourcc:
                # Request the codec before geometry. An uncompressed YUYV stream
                # saturates USB bandwidth and throttles the frame rate; MJPG lets
                # the same camera deliver 1080p30 over the same link.
                if len(fourcc) != 4:
                    raise ReplayError("fourcc must be exactly 4 characters, e.g. MJPG")
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
            if capture_width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
            if capture_height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
            if capture_fps:
                capture.set(cv2.CAP_PROP_FPS, capture_fps)
            if self.input_format == "y16":
                capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"Y16 "))
        return capture, kind

    @property
    def playback_fps(self) -> float:
        """FPS to use when writing a recording, so replays keep their real speed."""
        if self.info.fps and self.info.fps > 0:
            return self.info.fps
        return 30.0

    def frames(self) -> Iterator[Frame]:
        index = 0
        interval = 1.0 / self.info.fps if (self.pace and self.info.fps > 0) else 0.0
        next_deadline = time.perf_counter()
        while True:
            discontinuity = False
            if self._still_paths:
                if index >= len(self._still_paths):
                    if not self.loop:
                        return
                    index = 0
                    discontinuity = True
                path = self._still_paths[index]
                raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise ReplayError(f"Unable to read image: {path}")
            elif self._still is not None:
                raw = self._still
            else:
                assert self._capture is not None
                success, raw = self._capture.read()
                if not success and index == 0 and self.info.kind in ("camera", "v4l2"):
                    # A camera's first read routinely fails while the sensor is
                    # still starting up; give it a few seconds before deciding
                    # the source is dead.
                    deadline = time.perf_counter() + self.warmup_seconds
                    while not success and time.perf_counter() < deadline:
                        time.sleep(0.1)
                        success, raw = self._capture.read()
                    if success:
                        print("[source] camera ready after warm-up")
                if not success:
                    if self.loop and self.info.kind == "video":
                        discontinuity = True
                        self._capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                        success, raw = self._capture.read()
                        if not success:
                            return
                    else:
                        return
            image, pixel_origin = to_bgr(raw, self.input_format)
            if self.stereo_half:
                middle = image.shape[1] // 2
                image = (
                    image[:, :middle] if self.stereo_half == "left" else image[:, middle:]
                )
            if self.info.pixel_origin == "unknown":
                self.info.pixel_origin = pixel_origin
            source_seconds = (
                index / self.info.fps if self.info.fps > 0 else index * 0.0
            )
            yield Frame(image, index, source_seconds, pixel_origin, discontinuity)
            index += 1
            if self._still is not None and not self._still_paths:
                return
            if self.max_frames is not None and index >= self.max_frames:
                return
            if interval:
                next_deadline += interval
                remaining = next_deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_deadline = time.perf_counter()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "ReplaySource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
