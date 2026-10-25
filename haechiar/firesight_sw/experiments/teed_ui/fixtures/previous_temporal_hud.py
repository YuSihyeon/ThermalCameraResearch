"""HUD renderer.

Everything visual lives here. The renderer consumes an EdgeResult and a stats
dictionary and produces one composited image at the display resolution. Change
the look of the HUD by changing this file only; teed_core stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import cv2
import numpy as np
from edge_filter import EdgeFilterConfig, select_edges
from teed_core import EdgeResult
from temporal_edges import TemporalEdgeConfig, TemporalEdgeStabilizer
from thermal_overlay import ThermalConfig, draw_temperature
from visible_flame import VisibleFlameDetector


@dataclass
class HUDConfig:
    output_size: Optional[tuple[int, int]] = (
        None  # (width, height); None keeps model size
    )
    background_scale: float = 0.42
    edge_color: tuple[int, int, int] = (0, 255, 0)
    edge_filter: EdgeFilterConfig = field(default_factory=EdgeFilterConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    visible_flame_candidates: bool = False
    filter_edges: bool = True
    show_panel: bool = False
    # Passthrough draws the background at display resolution instead of the
    # model's 320x240, so the camera view stays sharp. Costs a little more per
    # frame and is pointless when the background is black.
    full_res_background: bool = True
    letterbox_color: tuple[int, int, int] = (0, 0, 0)
    temporal_edges: bool = False
    temporal_config: TemporalEdgeConfig = field(default_factory=TemporalEdgeConfig)


@dataclass
class HUDFrame:
    """What the renderer produced, plus what it measured about itself."""

    image: np.ndarray
    render_ms: float
    edge_metrics: dict
    thermal_metrics: dict
    flame_metrics: dict = field(default_factory=dict)


class HUDRenderer:
    def __init__(self, config: Optional[HUDConfig] = None) -> None:
        self.config = config or HUDConfig()
        self._flame_detector = VisibleFlameDetector()
        immediate = min(
            self.config.temporal_config.immediate_threshold,
            self.config.edge_filter.high_threshold,
        )
        temporal = replace(
            self.config.temporal_config,
            immediate_threshold=immediate,
            support_threshold=min(self.config.temporal_config.support_threshold, immediate),
        )
        self._temporal_edges = TemporalEdgeStabilizer(temporal)

    def reset_temporal(self) -> None:
        self._temporal_edges.reset()

    def _placement(self, image: np.ndarray) -> tuple[int, int, int, int]:
        """Where the scaled image sits in the output canvas: (x, y, w, h)."""
        target = self.config.output_size
        height, width = image.shape[:2]
        if target is None:
            return 0, 0, width, height
        target_width, target_height = target
        scale = min(target_width / width, target_height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return (
            (target_width - new_width) // 2,
            (target_height - new_height) // 2,
            new_width,
            new_height,
        )

    def _fit(self, image: np.ndarray) -> np.ndarray:
        """Scale into the output size, preserving aspect ratio with letterboxing."""
        target = self.config.output_size
        if target is None:
            return image
        target_width, target_height = target
        height, width = image.shape[:2]
        if (width, height) == (target_width, target_height):
            return image
        scale = min(target_width / width, target_height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=interpolation
        )
        canvas = np.full(
            (target_height, target_width, 3),
            self.config.letterbox_color,
            dtype=np.uint8,
        )
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        canvas[y : y + new_height, x : x + new_width] = resized
        return canvas

    def _fit_mask(
        self, mask: np.ndarray, placement: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Stretch the mask onto wherever the background landed.

        The model saw the whole frame squeezed into its own input size, so the
        mask covers the same field of view as the background and must be
        stretched to the background's rectangle - not letterboxed by its own
        aspect ratio, which would shift every edge inward.
        """
        target = self.config.output_size
        x, y, width, height = placement
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if target is None:
            return resized
        canvas = np.zeros((target[1], target[0]), dtype=mask.dtype)
        canvas[y : y + height, x : x + width] = resized
        return canvas

    def _draw_panel(self, image: np.ndarray, stats: dict) -> None:
        # Text is read on a virtual screen a short distance from the eye, so it
        # must scale with the output, not cap out at desktop-window size.
        scale = min(max(image.shape[1] / 640.0, 0.55), 3.0)
        line_height = int(20 * scale)
        lines = [
            "TEED  encoder {:5.1f} ms | head {:5.1f} ms | model {:5.1f} ms".format(
                stats.get("encoder", 0.0),
                stats.get("edge_head", 0.0),
                stats.get("model", 0.0),
            ),
            "PIPE  pre {:4.1f} | post {:4.1f} | hud {:4.1f} | e2e {:5.1f} ms".format(
                stats.get("preprocess", 0.0),
                stats.get("postprocess", 0.0),
                stats.get("hud", 0.0),
                stats.get("e2e", 0.0),
            ),
            "RATE  {:4.1f} fps | p95 model {:5.1f} ms | p95 e2e {:5.1f} ms".format(
                stats.get("fps", 0.0),
                stats.get("p95_model", 0.0),
                stats.get("p95_e2e", 0.0),
            ),
            "SRC   {} | frame {}".format(
                stats.get("source", "-"), stats.get("frame", 0)
            ),
        ]
        panel_height = line_height * len(lines) + int(10 * scale)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        for index, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (int(8 * scale), int(16 * scale) + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * scale,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        legend = "GREEN = TEED EDGE"
        if stats.get("hot_pixels", 0) > 0:
            legend += "   YELLOW/ORANGE/RED = TEMPERATURE CONTOURS"
        # Dark strip behind the legend: without it the white text disappears
        # wherever the frame underneath is bright.
        strip_top = image.shape[0] - int(28 * scale)
        strip = image[strip_top:, :]
        cv2.addWeighted(np.zeros_like(strip), 0.55, strip, 0.45, 0, strip)
        cv2.putText(
            image,
            legend,
            (int(8 * scale), image.shape[0] - int(10 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        if stats.get("recording"):
            cv2.circle(
                image,
                (image.shape[1] - int(28 * scale), int(20 * scale)),
                int(7 * scale),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                image,
                "REC",
                (image.shape[1] - int(90 * scale), int(26 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46 * scale,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    def render(
        self,
        frame_bgr: np.ndarray,
        edge: EdgeResult,
        stats: Optional[dict] = None,
        pixel_origin: str = "bgr8",
        temperature_c: Optional[np.ndarray] = None,
        depth_m: Optional[np.ndarray] = None,
        ignore_regions: tuple = (),
        frame_index: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> HUDFrame:
        import time

        started = time.perf_counter()
        stats = dict(stats or {})
        base = edge.source_view
        for name, array in (("temperature_c", temperature_c), ("depth_m", depth_m)):
            if array is not None and array.shape != frame_bgr.shape[:2]:
                raise ValueError(name + " must be registered to the original frame")
        if self.config.filter_edges:
            probability = edge.probability
            if self.config.temporal_edges:
                probability, temporal_metrics = self._temporal_edges.update(
                    probability, base, ignore_regions,
                    frame_index=frame_index, timestamp_s=timestamp_s,
                )
            else:
                self._temporal_edges.reset()
                temporal_metrics = {"status": "disabled"}
            depth_small = (
                None
                if depth_m is None
                else cv2.resize(depth_m, edge.size, interpolation=cv2.INTER_NEAREST)
            )
            selection = select_edges(
                probability,
                base,
                self.config.edge_filter,
                ignore_regions,
                depth_small,
            )
            mask, alpha, edge_metrics = (
                selection.mask,
                selection.alpha,
                {**selection.metrics, "temporal": temporal_metrics},
            )
        else:
            self._temporal_edges.reset()
            mask = edge.mask
            alpha = (mask > 0).astype(np.float32)
            edge_metrics = {
                "mode": "legacy",
                "selected_pixels": int(np.count_nonzero(mask)),
            }

        use_full_res = (
            self.config.full_res_background
            and self.config.background_scale > 0
            and self.config.output_size is not None
        )
        background = frame_bgr if use_full_res else base
        composited = np.clip(
            self._fit(background).astype(np.float32) * self.config.background_scale,
            0,
            255,
        ).astype(np.uint8)
        placement = self._placement(background)
        x, y, width, height = placement
        flame_mask = np.zeros(base.shape[:2], np.uint8)
        if not self.config.visible_flame_candidates:
            self._flame_detector.reset()
            flame_metrics = {"status": "disabled", "candidate_pixels": 0}
        elif pixel_origin != "bgr8" or temperature_c is not None:
            self._flame_detector.reset()
            flame_metrics = {"status": "unsupported_input", "candidate_pixels": 0}
        else:
            flame_mask, flame_metrics = self._flame_detector.update(
                base, ignore_regions
            )
        edge_up = self._fit_mask(mask, placement)
        alpha_up = self._fit_mask(alpha, placement)
        flame_up = self._fit_mask(flame_mask, placement)
        selected = (edge_up > 0) & (flame_up == 0)
        amount = alpha_up[selected, None]
        composited[selected] = np.rint(
            composited[selected] * (1 - amount)
            + np.asarray(self.config.edge_color) * amount
        ).astype(np.uint8)

        if np.any(flame_up):
            flame_pixels = flame_up > 0
            composited[flame_pixels] = np.rint(
                composited[flame_pixels] * 0.85
                + np.array((0, 0, 255)) * 0.15
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                flame_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(composited, contours, -1, (0, 0, 255), 2)
            left, top, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label_x = min(left, max(0, composited.shape[1] - 70))
            label_y = max(15, top - 5)
            cv2.putText(
                composited, "FLAME?", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                composited, "FLAME?", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA,
            )

        view = composited[y : y + height, x : x + width]
        temperature_view = (
            None
            if temperature_c is None
            else cv2.resize(
                temperature_c, (width, height), interpolation=cv2.INTER_NEAREST
            )
        )
        thermal_metrics = draw_temperature(
            view, temperature_view, self.config.thermal, ignore_regions
        )
        stats.update(thermal_metrics)
        if self.config.show_panel:
            self._draw_panel(composited, stats)
        # A compact availability cue avoids silently presenting absent heat data as safe.
        if thermal_metrics["status"] == "temperature_unavailable":
            cv2.putText(
                composited,
                "TEMP N/A",
                (6, composited.shape[0] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
        return HUDFrame(
            image=composited,
            render_ms=(time.perf_counter() - started) * 1000,
            edge_metrics=edge_metrics,
            thermal_metrics=thermal_metrics,
            flame_metrics=flame_metrics,
        )
