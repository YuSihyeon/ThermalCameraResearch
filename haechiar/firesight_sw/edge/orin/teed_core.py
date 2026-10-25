"""TEED inference core.

This module performs edge inference and nothing else. It never draws, never
opens a window, and never parses arguments. Everything it returns is data, so
the HUD renderer is free to decide how any of it is presented.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from teed_model import TED


# BIPED training mean, kept identical to the upstream TEED preprocessing.
MEAN_BGR = np.asarray((103.939, 116.779, 123.68), dtype=np.float32)

# SHA-256 of the BIPED epoch-5 checkpoint shared as models/teed/5_model.pth.
KNOWN_CHECKPOINT_SHA256 = (
    "0322caf70f588355aaaf59c2bf5872b21a4b7e9f679971a7a3bb1f69b56a01ba"
)


class TEEDError(RuntimeError):
    """Raised when valid TEED inference cannot be performed."""


@dataclass(frozen=True)
class EdgeResult:
    """Everything TEED knows about one frame. No pixels are styled here."""

    probability: np.ndarray  # float32, [0, 1], model resolution
    mask: np.ndarray  # uint8, 0 or 255, model resolution
    source_view: np.ndarray  # the BGR frame resized to model input, reused by the HUD
    size: tuple[int, int]  # (width, height) the model actually ran at
    timings_ms: dict[str, float]  # preprocess / encoder / edge_head / model / postprocess / total

    @property
    def edge_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TEEDError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in {"cuda", "cpu"}:
        raise TEEDError(f"Unsupported device: {requested}")
    return torch.device(requested)


def clear_regions(
    mask: np.ndarray,
    regions: Sequence[tuple[float, float, float, float]],
) -> np.ndarray:
    """Zero out normalized (left, top, right, bottom) boxes, e.g. burned-in OSD."""
    height, width = mask.shape[:2]
    for left, top, right, bottom in regions:
        mask[
            round(top * height) : round(bottom * height) + 1,
            round(left * width) : round(right * width) + 1,
        ] = 0
    return mask


class TEEDCore:
    """Loads the checkpoint once, then turns BGR frames into EdgeResult."""

    def __init__(
        self,
        checkpoint: Path,
        width: int = 320,
        height: int = 256,
        threshold: float = 0.75,
        edge_width: int = 3,
        contrast_clip_limit: float = 2.0,
        device: str = "auto",
        half: bool = False,
        verify_checkpoint: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise TEEDError(f"TEED checkpoint not found: {checkpoint}")
        if width < 64 or height < 64:
            raise ValueError("TEED input must be at least 64x64")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if not 1 <= edge_width <= 15:
            raise ValueError("edge_width must be within [1, 15]")
        if not 0.0 <= contrast_clip_limit <= 10.0:
            raise ValueError("contrast_clip_limit must be within [0, 10]")

        self.checkpoint = checkpoint.resolve()
        self.width = int(width)
        self.height = int(height)
        self.threshold = float(threshold)
        self.edge_width = int(edge_width)
        self.contrast_clip_limit = float(contrast_clip_limit)
        self.device = resolve_device(device)
        self.half = bool(half)
        if self.half and self.device.type != "cuda":
            raise TEEDError("half precision requires CUDA")

        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.checkpoint_is_known = (
            self.checkpoint_sha256 == KNOWN_CHECKPOINT_SHA256
        )
        if verify_checkpoint and not self.checkpoint_is_known:
            raise TEEDError(
                "TEED checkpoint SHA-256 does not match the shared 5_model.pth.\n"
                f"Expected: {KNOWN_CHECKPOINT_SHA256}\n"
                f"Actual  : {self.checkpoint_sha256}\n"
                "Pass verify_checkpoint=False to run a different checkpoint."
            )

        self.model = TED().to(self.device)
        self.loaded_tensor_count = self._load_state_dict()
        self.model.eval()
        if self.half:
            self.model.half()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True

        self._pad_width = (8 - self.width % 8) % 8
        self._pad_height = (8 - self.height % 8) % 8

    def _load_state_dict(self) -> int:
        try:
            try:
                payload = torch.load(
                    self.checkpoint, map_location=self.device, weights_only=True
                )
            except TypeError:
                payload = torch.load(self.checkpoint, map_location=self.device)
        except (OSError, RuntimeError, ValueError) as error:
            raise TEEDError(f"Unable to read {self.checkpoint}: {error}") from error
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TEEDError(f"Unsupported checkpoint payload: {self.checkpoint}")
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in payload.items()
        }
        self.model.load_state_dict(state, strict=True)
        return len(state)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prepare(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise TEEDError(
                f"TEED expects an HxWx3 BGR frame, got shape {frame_bgr.shape}"
            )
        resized = cv2.resize(
            frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        model_input = resized
        if self.contrast_clip_limit > 0.0:
            # Lift local contrast on the lightness channel only, so smoke-washed
            # and low-light frames still present edges to the model. The frame
            # handed back for display stays untouched: this sharpens detection,
            # it does not restyle what the wearer sees.
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.contrast_clip_limit, tileGridSize=(8, 8)
            )
            model_input = cv2.cvtColor(
                cv2.merge((clahe.apply(lightness), channel_a, channel_b)),
                cv2.COLOR_LAB2BGR,
            )
        array = model_input.astype(np.float32) - MEAN_BGR
        tensor = torch.from_numpy(
            np.ascontiguousarray(array.transpose(2, 0, 1))
        ).unsqueeze(0)
        if self._pad_width or self._pad_height:
            tensor = functional.pad(
                tensor, (0, self._pad_width, 0, self._pad_height), mode="replicate"
            )
        tensor = tensor.to(self.device, non_blocking=self.device.type == "cuda")
        return (tensor.half() if self.half else tensor.float()), resized

    def _run_model(self, tensor: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
        """Return the fused logits plus encoder / head / total milliseconds."""
        if self.device.type == "cuda":
            start, middle, end = (
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            )
            start.record()
            features = self.model.encode(tensor)
            middle.record()
            outputs = self.model.head(features)
            end.record()
            torch.cuda.synchronize(self.device)
            return (
                outputs[-1],
                float(start.elapsed_time(middle)),
                float(middle.elapsed_time(end)),
                float(start.elapsed_time(end)),
            )
        started = time.perf_counter()
        features = self.model.encode(tensor)
        middle = time.perf_counter()
        outputs = self.model.head(features)
        ended = time.perf_counter()
        return (
            outputs[-1],
            (middle - started) * 1000.0,
            (ended - middle) * 1000.0,
            (ended - started) * 1000.0,
        )

    def infer(
        self,
        frame_bgr: np.ndarray,
        ignore_regions: Optional[Sequence[tuple[float, float, float, float]]] = None,
    ) -> EdgeResult:
        total_started = time.perf_counter()

        self._synchronize()
        prepare_started = time.perf_counter()
        tensor, source_view = self._prepare(frame_bgr)
        self._synchronize()
        preprocess_ms = (time.perf_counter() - prepare_started) * 1000.0

        with torch.inference_mode():
            logits, encoder_ms, head_ms, model_ms = self._run_model(tensor)
            probability_tensor = torch.sigmoid(logits)[
                0, 0, : self.height, : self.width
            ]

        post_started = time.perf_counter()
        probability = probability_tensor.detach().float().cpu().numpy()
        mask = (probability >= self.threshold).astype(np.uint8) * 255
        if self.edge_width > 1:
            kernel_size = self.edge_width + (1 - self.edge_width % 2)
            mask = cv2.dilate(
                mask, np.ones((kernel_size, kernel_size), dtype=np.uint8)
            )
        if ignore_regions:
            mask = clear_regions(mask, ignore_regions)
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0

        return EdgeResult(
            probability=probability.astype(np.float32, copy=False),
            mask=mask,
            source_view=source_view,
            size=(self.width, self.height),
            timings_ms={
                "preprocess": preprocess_ms,
                "encoder": encoder_ms,
                "edge_head": head_ms,
                "model": model_ms,
                "postprocess": postprocess_ms,
                "total": (time.perf_counter() - total_started) * 1000.0,
            },
        )

    def describe(self) -> dict[str, object]:
        """Facts about this core, for the metrics record."""
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_is_known_5_model": self.checkpoint_is_known,
            "loaded_tensors": self.loaded_tensor_count,
            "device": str(self.device),
            "half": self.half,
            "input_size": [self.width, self.height],
            "threshold": self.threshold,
            "edge_width": self.edge_width,
            "contrast_clip_limit": self.contrast_clip_limit,
            "torch_version": torch.__version__,
        }
