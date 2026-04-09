"""
Fall Detection Engine — Bounding Box Aspect Ratio Method
==========================================================
ENGG1101 Project — HKU

Technique:
  - When standing, a person's bounding box is TALLER than WIDE  → ratio > 1.0
  - When falling/lying, the bounding box is WIDER than TALL     → ratio < 1.0

  aspect_ratio = box_height / box_width

  If aspect_ratio < threshold (default 0.7), classify as FALLING.

Author : ENGG1101 Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class FallAlert:
    """Describes a detected fall event."""
    timestamp: float
    aspect_ratio: float
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    reason: str


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class FallDetectorConfig:
    # Aspect ratio threshold — below this = falling
    aspect_ratio_threshold: float = 0.7

    # Require N consecutive "fall" frames before triggering alert
    confirmation_frames: int = 10

    # Cooldown between alerts (seconds)
    post_alert_cooldown: float = 5.0


# ──────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────
class FallDetector:
    """
    Per-person fall detector using bounding box aspect ratio.

    Create one instance per tracked person ID.
    Feed it the person's bounding box each frame.
    """

    def __init__(self, config: FallDetectorConfig | None = None):
        self.cfg = config or FallDetectorConfig()
        self._consecutive_fall_frames = 0
        self._last_alert_time: float = 0.0
        self._latest_ratio: float = 1.5   # assume standing initially
        self._latest_bbox: Optional[Tuple[int, int, int, int]] = None

    def update(self, bbox: Tuple[int, int, int, int],
               timestamp: float | None = None) -> Optional[FallAlert]:
        """
        Process one frame for this person.

        Parameters
        ----------
        bbox : (x1, y1, x2, y2) — bounding box coordinates
        timestamp : epoch seconds (defaults to time.time())

        Returns
        -------
        FallAlert or None
        """
        ts = timestamp if timestamp is not None else time.time()
        x1, y1, x2, y2 = bbox

        # Calculate aspect ratio
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        ratio = box_h / box_w

        self._latest_ratio = ratio
        self._latest_bbox = bbox

        # Cooldown check
        if ts - self._last_alert_time < self.cfg.post_alert_cooldown:
            self._consecutive_fall_frames = 0
            return None

        # Is this frame a "fall" frame?
        is_falling = ratio < self.cfg.aspect_ratio_threshold

        if is_falling:
            self._consecutive_fall_frames += 1
        else:
            # Decay slowly so brief glitches don't reset entirely
            self._consecutive_fall_frames = max(
                0, self._consecutive_fall_frames - 2
            )

        # Not enough consecutive frames yet
        if self._consecutive_fall_frames < self.cfg.confirmation_frames:
            return None

        # ── Emit alert ──
        self._last_alert_time = ts
        self._consecutive_fall_frames = 0

        return FallAlert(
            timestamp=ts,
            aspect_ratio=ratio,
            bbox=bbox,
            reason=f"Aspect ratio {ratio:.2f} < {self.cfg.aspect_ratio_threshold} "
                   f"for {self.cfg.confirmation_frames} consecutive frames",
        )

    def get_latest_metrics(self) -> dict:
        """Return the most recent metrics for the overlay."""
        return {
            "aspect_ratio": self._latest_ratio,
            "bbox": self._latest_bbox,
            "threshold": self.cfg.aspect_ratio_threshold,
        }
