"""Display and recording sinks.

The window sink is what drives the nreal Light: the glasses appear to the OS as
an ordinary external monitor, so putting a fullscreen window on that screen is
the whole integration. No AR SDK is involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _list_displays_xrandr() -> list[dict]:
    """Read the X screen layout on Linux, so --screen works on the Jetson too.

    `xrandr --listmonitors` prints one line per active monitor with the
    geometry as WIDTH/mmxHEIGHT/mm+X+Y, which is all the placement we need.
    """
    import re
    import shutil
    import subprocess

    if not shutil.which("xrandr"):
        return []
    try:
        output = subprocess.run(
            ["xrandr", "--listmonitors"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    displays = []
    pattern = re.compile(
        r"^\s*(\d+):\s+(\+\*?)?(\S+)\s+(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        index, primary, name, width, height, x, y = match.groups()
        displays.append(
            {
                "index": int(index),
                "id": name,
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                # xrandr marks the primary output with an asterisk.
                "is_main": bool(primary and "*" in primary),
            }
        )
    return displays


def list_displays() -> list[dict]:
    """Enumerate attached displays with their arrangement coordinates.

    macOS window coordinates are in points, not pixels, so a Retina built-in
    panel reports a smaller origin than its native resolution suggests. Reading
    the real bounds avoids guessing where a second screen starts. Returns an
    empty list on platforms without CoreGraphics; pass --window-x there.
    """
    import ctypes
    import ctypes.util

    library = ctypes.util.find_library("CoreGraphics")
    if not library:
        return _list_displays_xrandr()
    try:
        core = ctypes.CDLL(library)

        class _Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

        class _Size(ctypes.Structure):
            _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

        class _Rect(ctypes.Structure):
            _fields_ = [("origin", _Point), ("size", _Size)]

        core.CGDisplayBounds.restype = _Rect
        core.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        core.CGMainDisplayID.restype = ctypes.c_uint32

        count = ctypes.c_uint32()
        identifiers = (ctypes.c_uint32 * 16)()
        core.CGGetActiveDisplayList(16, identifiers, ctypes.byref(count))
        main = core.CGMainDisplayID()
    except (OSError, AttributeError):
        return _list_displays_xrandr()

    displays = []
    for index in range(count.value):
        identifier = identifiers[index]
        bounds = core.CGDisplayBounds(identifier)
        displays.append(
            {
                "index": index,
                "id": int(identifier),
                "x": int(bounds.origin.x),
                "y": int(bounds.origin.y),
                "width": int(bounds.size.width),
                "height": int(bounds.size.height),
                "is_main": identifier == main,
            }
        )
    return displays


class DisplayError(RuntimeError):
    """Raised when a display or recording sink cannot be created."""


class Display:
    """Optional preview window plus optional recording, both fed the same image."""

    def __init__(
        self,
        title: str = "FireSight Edge",
        headless: bool = False,
        fullscreen: bool = False,
        window_position: Optional[tuple[int, int]] = None,
        window_size: Optional[tuple[int, int]] = None,
        record_path: Optional[str] = None,
        record_fps: float = 30.0,
        recording: bool = False,
    ) -> None:
        self.title = title
        self.headless = bool(headless)
        self.fullscreen = bool(fullscreen)
        self.record_path = record_path
        self.record_fps = max(float(record_fps), 1.0)
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_size: Optional[tuple[int, int]] = None
        self._window_ready = False
        self._window_position = window_position
        self._window_size = window_size
        self.recording = False
        if recording:
            if not record_path:
                raise DisplayError("recording was requested without a record path")
            self.recording = True

    # -- window ---------------------------------------------------------------

    def _ensure_window(self, image: np.ndarray) -> None:
        if self.headless or self._window_ready:
            return
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if self._window_position is not None:
            # Moving the window onto the glasses' screen before going fullscreen
            # is what makes fullscreen land on the right display.
            cv2.moveWindow(self.title, *self._window_position)
        # Sizing before positioning is what makes the move actually stick on
        # macOS; a bare moveWindow is frequently ignored.
        width, height = self._window_size or (image.shape[1], image.shape[0])
        cv2.resizeWindow(self.title, width, height)
        if self._window_position is not None:
            cv2.moveWindow(self.title, *self._window_position)
        if self.fullscreen:
            cv2.setWindowProperty(
                self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        self._window_ready = True

    # -- recording ------------------------------------------------------------

    def _ensure_writer(self, image: np.ndarray) -> None:
        if self._writer is not None:
            return
        if not self.record_path:
            raise DisplayError("recording was requested without a record path")
        size = (image.shape[1], image.shape[0])
        path = Path(self.record_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.record_fps, size
        )
        if not writer.isOpened():
            raise DisplayError(f"Unable to create recording: {path}")
        self._writer = writer
        self._writer_size = size

    def toggle_fullscreen(self) -> bool:
        """Flip fullscreen at runtime, so the window can be placed by hand first."""
        if self.headless:
            return False
        self.fullscreen = not self.fullscreen
        if self._window_ready:
            cv2.setWindowProperty(
                self.title,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
            )
        return self.fullscreen

    def toggle_recording(self) -> bool:
        if self.recording:
            self.stop_recording()
        else:
            if not self.record_path:
                raise DisplayError("no record path configured")
            self.recording = True
        return self.recording

    def stop_recording(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._writer_size = None
        self.recording = False

    # -- main entry -----------------------------------------------------------

    def show(self, image: np.ndarray) -> int:
        """Present one frame. Returns the pressed key, or -1."""
        if self.recording:
            self._ensure_writer(image)
            size = (image.shape[1], image.shape[0])
            if size != self._writer_size:
                raise DisplayError(
                    f"recording size changed mid-stream: {self._writer_size} -> {size}"
                )
            self._writer.write(image)
        if self.headless:
            return -1
        self._ensure_window(image)
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        self.stop_recording()
        if self._window_ready:
            cv2.destroyWindow(self.title)
            self._window_ready = False

    def __enter__(self) -> "Display":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
