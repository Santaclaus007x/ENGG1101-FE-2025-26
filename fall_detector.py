"""
Fall Detection Engine — Aspect Ratio + Torso Angle (Two-Signal)
================================================================
ENGG1101 Project — HKU

Two complementary signals are combined to decide if a person is falling:

  Signal 1 — Bounding Box Aspect Ratio (H/W)
    Standing  → bounding box is TALLER than wide  → ratio > 1.0
    Falling   → bounding box is WIDER than tall   → ratio < threshold (0.7)

  Signal 2 — Torso Angle from Vertical
    Standing  → mid-shoulder directly above mid-hip → angle ≈ 0–20°
    Falling   → torso tilts toward horizontal       → angle > threshold (50°)
    Computed from COCO keypoints: shoulders (5,6) and hips (11,12)

  A frame is classified as a FALL only when BOTH signals fire simultaneously.
  If keypoint confidence is too low to compute the torso angle, the system
  falls back to aspect-ratio only (backward compatible).

  This dual-signal approach eliminates most false positives from:
    - Bending over to pick something up   (ratio low, angle low → no alert)
    - Sitting down quickly                (ratio low, angle moderate → no alert)
    - Someone lying on a bed/sofa         (ratio low, angle low → no alert)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class FallAlert:
    """Describes a confirmed fall event."""
    timestamp:    float
    aspect_ratio: float
    torso_angle:  Optional[float]            # None if keypoints unavailable
    bbox:         Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    reason:       str


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class FallDetectorConfig:
    # Signal 1 — aspect ratio
    aspect_ratio_threshold: float = 0.7    # H/W below this = potential fall

    # Signal 2 — torso angle
    torso_angle_threshold:  float = 50.0   # degrees from vertical above this = potential fall

    # Confirmation — require N consecutive dual-signal frames
    confirmation_frames:    int   = 10

    # Cooldown between alerts (seconds)
    post_alert_cooldown:    float = 5.0


# ──────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────
class FallDetector:
    """
    Per-person fall detector using bounding box aspect ratio + torso angle.

    Create one instance per tracked person ID.
    Call update() every frame with the bounding box and (optionally) the
    torso angle computed from pose keypoints.
    """

    def __init__(self, config: FallDetectorConfig | None = None):
        self.cfg = config or FallDetectorConfig()
        self._consecutive_fall_frames: int   = 0
        self._last_alert_time:         float = 0.0
        self._latest_ratio:            float = 1.5    # assume standing
        self._latest_torso_angle:      Optional[float] = None
        self._latest_bbox:             Optional[Tuple[int, int, int, int]] = None

    def update(
        self,
        bbox:        Tuple[int, int, int, int],
        timestamp:   float | None = None,
        torso_angle: float | None = None,
    ) -> Optional[FallAlert]:
        """
        Process one frame for this person.

        Parameters
        ----------
        bbox        : (x1, y1, x2, y2) bounding box coordinates
        timestamp   : epoch seconds (defaults to time.time())
        torso_angle : degrees from vertical computed from pose keypoints.
                      Pass None if keypoints aren't available / confident.

        Returns
        -------
        FallAlert if a fall is confirmed, else None.
        """
        ts = timestamp if timestamp is not None else time.time()
        x1, y1, x2, y2 = bbox

        # ── Signal 1: aspect ratio ─────────────────────────────────────
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        ratio = box_h / box_w

        self._latest_ratio       = ratio
        self._latest_torso_angle = torso_angle
        self._latest_bbox        = bbox

        # ── Cooldown ───────────────────────────────────────────────────
        if ts - self._last_alert_time < self.cfg.post_alert_cooldown:
            self._consecutive_fall_frames = 0
            return None

        # ── Two-signal fall decision ───────────────────────────────────
        ratio_falling = ratio < self.cfg.aspect_ratio_threshold

        if torso_angle is not None:
            # Both signals available → require both to fire
            angle_falling = torso_angle > self.cfg.torso_angle_threshold
            is_falling    = ratio_falling and angle_falling
        else:
            # No keypoint data → fall back to aspect ratio alone
            is_falling = ratio_falling

        if is_falling:
            self._consecutive_fall_frames += 1
        else:
            # Decay slowly so brief glitches don't fully reset the counter
            self._consecutive_fall_frames = max(
                0, self._consecutive_fall_frames - 2
            )

        if self._consecutive_fall_frames < self.cfg.confirmation_frames:
            return None

        # ── Emit alert ─────────────────────────────────────────────────
        self._last_alert_time         = ts
        self._consecutive_fall_frames = 0

        if torso_angle is not None:
            reason = (
                f"H/W={ratio:.2f} < {self.cfg.aspect_ratio_threshold}  AND  "
                f"torso={torso_angle:.1f}° > {self.cfg.torso_angle_threshold}°  "
                f"for {self.cfg.confirmation_frames} consecutive frames"
            )
        else:
            reason = (
                f"H/W={ratio:.2f} < {self.cfg.aspect_ratio_threshold}  "
                f"for {self.cfg.confirmation_frames} consecutive frames "
                f"(no keypoint data)"
            )

        return FallAlert(
            timestamp=ts,
            aspect_ratio=ratio,
            torso_angle=torso_angle,
            bbox=bbox,
            reason=reason,
        )

    def get_latest_metrics(self) -> dict:
        """Return the most recent metrics for the overlay and buzzer logic."""
        ratio  = self._latest_ratio
        angle  = self._latest_torso_angle
        r_fall = ratio < self.cfg.aspect_ratio_threshold
        a_fall = (angle is not None and angle > self.cfg.torso_angle_threshold)

        return {
            "aspect_ratio":       ratio,
            "torso_angle":        angle,
            "bbox":               self._latest_bbox,
            "threshold":          self.cfg.aspect_ratio_threshold,
            "angle_threshold":    self.cfg.torso_angle_threshold,
            # True when both signals agree the person is currently falling
            "is_currently_falling": r_fall and (a_fall if angle is not None else True),
        }
