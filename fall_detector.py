"""
Fall Detection Engine — Tuned Dual-Signal
==========================================
ENGG1101 Project — HKU

Two complementary signals are combined to decide if a person is falling:

  Signal 1 — Bounding Box Aspect Ratio (H/W)
    Standing  → bounding box is TALLER than wide  → ratio > 1.0
    Falling   → bounding box is WIDER than tall   → ratio < threshold (0.7)

  Signal 2 — Torso Angle from Vertical
    Standing  → mid-shoulder directly above mid-hip → angle ≈ 0–20°
    Falling   → torso tilts toward horizontal       → angle > threshold (50°)
    Computed from COCO keypoints: shoulders (5,6) and hips (11,12)

  A frame is classified as FALLING only when BOTH signals fire simultaneously.
  If keypoint confidence is too low to compute the torso angle, the system
  falls back to aspect-ratio only (backward compatible).

What's new in this tuned version
--------------------------------
  • EMA smoothing on raw signals        → reduces bounding-box / keypoint jitter
  • Angle velocity tracking (°/sec)     → rapid falls confirm faster
  • Adaptive confirmation window        → severe falls need fewer frames
  • Weighted counter boost on severity  → momentum builds faster for clear falls
  • Split update() into private helpers → easier to read and tune

All public API stays the same: FallAlert, FallDetectorConfig, FallDetector,
update(), get_latest_metrics().  Existing callers don't need any changes.

This dual-signal approach eliminates most false positives from:
  - Bending over to pick something up   (ratio low, angle low → no alert)
  - Sitting down quickly                (ratio low, angle moderate → no alert)
  - Someone lying on a bed/sofa         (ratio low, angle low → no alert)
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
    # ── Primary thresholds (same behaviour as before) ──
    aspect_ratio_threshold: float = 0.7    # H/W below this = potential fall
    torso_angle_threshold:  float = 50.0   # degrees from vertical above this = potential fall

    # ── Severe thresholds — when BOTH are crossed, we fast-confirm ──
    # A "severe" fall is unambiguous: ratio clearly horizontal AND torso
    # clearly flat.  No need to wait a full 0.5 s before alerting.
    severe_ratio_threshold: float = 0.50   # clearly horizontal silhouette
    severe_angle_threshold: float = 65.0   # torso clearly past horizontal

    # ── Confirmation window (consecutive frames) ──
    confirmation_frames:      int = 10     # standard dual-signal confirmation
    fast_confirmation_frames: int = 3      # used when a severe fall is detected

    # ── Angle velocity (°/sec) — rapid rotation = unmistakable fall ──
    angle_velocity_threshold: float = 120.0   # above this, boost counter

    # ── EMA smoothing on raw inputs (α closer to 1 = snappier) ──
    # Reduces one-frame noise without adding much lag.
    signal_smoothing_alpha: float = 0.6

    # ── Decay behaviour when the signal drops ──
    decay_per_frame: int = 2               # counter steps lost per non-falling frame

    # ── Cooldown between alerts (seconds) ──
    post_alert_cooldown: float = 5.0


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

        # Counter + cooldown state
        self._consecutive_fall_frames: int   = 0
        self._last_alert_time:         float = 0.0

        # Smoothed signals (what the decision logic actually reads)
        self._smooth_ratio:       float           = 1.5    # assume standing
        self._smooth_torso_angle: Optional[float] = None

        # Angle velocity tracking
        self._last_raw_angle:   Optional[float] = None
        self._last_angle_ts:    Optional[float] = None
        self._angle_velocity:   float           = 0.0      # °/sec

        # Latest data for overlay / metrics
        self._latest_bbox: Optional[Tuple[int, int, int, int]] = None

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────
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

        # ── 1. Update all signals (smoothed ratio + angle + velocity) ──
        self._update_signals(bbox, torso_angle, ts)

        # ── 2. Cooldown gate — no alerts during recovery window ──
        if ts - self._last_alert_time < self.cfg.post_alert_cooldown:
            self._consecutive_fall_frames = 0
            return None

        # ── 3. Evaluate current fall state ──
        is_falling, is_severe, fast_velocity = self._evaluate_fall_state()

        # ── 4. Update confirmation counter ──
        self._update_confirmation_counter(is_falling, is_severe, fast_velocity)

        # ── 5. Decide whether to emit an alert ──
        required = (self.cfg.fast_confirmation_frames
                    if is_severe else self.cfg.confirmation_frames)

        if self._consecutive_fall_frames < required:
            return None

        return self._emit_alert(ts, is_severe, fast_velocity)

    def get_latest_metrics(self) -> dict:
        """Return the most recent metrics for the overlay and buzzer logic."""
        ratio = self._smooth_ratio
        angle = self._smooth_torso_angle

        r_fall = ratio < self.cfg.aspect_ratio_threshold
        a_fall = (angle is not None and angle > self.cfg.torso_angle_threshold)

        return {
            # ── Original keys (do not remove — visualizer relies on them) ──
            "aspect_ratio":       ratio,
            "torso_angle":        angle,
            "bbox":               self._latest_bbox,
            "threshold":          self.cfg.aspect_ratio_threshold,
            "angle_threshold":    self.cfg.torso_angle_threshold,
            "is_currently_falling": r_fall and (a_fall if angle is not None else True),
            # ── New debug / tuning keys ──
            "angle_velocity":        self._angle_velocity,
            "confirmation_progress": self._consecutive_fall_frames,
            "confirmation_required": self.cfg.confirmation_frames,
        }

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────
    def _update_signals(
        self,
        bbox:        Tuple[int, int, int, int],
        torso_angle: Optional[float],
        ts:          float,
    ) -> None:
        """Compute raw signals, apply EMA smoothing, update angle velocity."""
        x1, y1, x2, y2 = bbox
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        raw_ratio = box_h / box_w

        a = self.cfg.signal_smoothing_alpha

        # EMA on aspect ratio
        self._smooth_ratio = a * raw_ratio + (1.0 - a) * self._smooth_ratio

        # EMA on torso angle (only when we have data this frame)
        if torso_angle is not None:
            if self._smooth_torso_angle is None:
                self._smooth_torso_angle = torso_angle            # seed
            else:
                self._smooth_torso_angle = (a * torso_angle
                                            + (1.0 - a) * self._smooth_torso_angle)

            # Angle velocity (°/sec) — use RAW angle, not smoothed, or we
            # lose the very signal we're trying to detect.
            if (self._last_raw_angle is not None and
                    self._last_angle_ts is not None):
                dt = ts - self._last_angle_ts
                if dt > 1e-3:
                    self._angle_velocity = (torso_angle - self._last_raw_angle) / dt
            self._last_raw_angle = torso_angle
            self._last_angle_ts  = ts
        else:
            # Lost keypoints — keep the last smoothed angle, but stop
            # reporting a stale velocity.
            self._angle_velocity = 0.0
            self._last_raw_angle = None
            self._last_angle_ts  = None

        self._latest_bbox = bbox

    def _evaluate_fall_state(self) -> Tuple[bool, bool, bool]:
        """
        Decide if the CURRENT frame looks like a fall.

        Returns
        -------
        is_falling     : both signals fire (ratio + angle), or ratio alone
                         when no angle is available
        is_severe      : both signals deeply exceeded — high-confidence fall
        fast_velocity  : torso is rotating rapidly (°/sec above threshold)
        """
        ratio = self._smooth_ratio
        angle = self._smooth_torso_angle

        ratio_falling = ratio < self.cfg.aspect_ratio_threshold
        severe_ratio  = ratio < self.cfg.severe_ratio_threshold

        if angle is not None:
            angle_falling = angle > self.cfg.torso_angle_threshold
            severe_angle  = angle > self.cfg.severe_angle_threshold
            is_falling    = ratio_falling and angle_falling
            is_severe     = severe_ratio  and severe_angle
        else:
            # Fall back to ratio-only (original behaviour)
            is_falling = ratio_falling
            is_severe  = severe_ratio

        fast_velocity = abs(self._angle_velocity) > self.cfg.angle_velocity_threshold
        return is_falling, is_severe, fast_velocity

    def _update_confirmation_counter(
        self,
        is_falling:    bool,
        is_severe:     bool,
        fast_velocity: bool,
    ) -> None:
        """
        Advance or decay the consecutive-fall counter.
        Severe / fast-velocity frames boost the counter by 2 instead of 1
        so a clearly-falling person alerts in a handful of frames.
        """
        if is_falling:
            boost = 2 if (is_severe or fast_velocity) else 1
            self._consecutive_fall_frames += boost
        else:
            self._consecutive_fall_frames = max(
                0, self._consecutive_fall_frames - self.cfg.decay_per_frame
            )

    def _emit_alert(
        self,
        ts:            float,
        is_severe:     bool,
        fast_velocity: bool,
    ) -> FallAlert:
        """Build and return a FallAlert, and reset the counter/cooldown."""
        self._last_alert_time         = ts
        self._consecutive_fall_frames = 0

        ratio = self._smooth_ratio
        angle = self._smooth_torso_angle

        tags = []
        if is_severe:
            tags.append("SEVERE")
        if fast_velocity:
            tags.append(f"FAST {self._angle_velocity:.0f} deg/s")

        if angle is not None:
            base = (f"H/W={ratio:.2f} < {self.cfg.aspect_ratio_threshold}  AND  "
                    f"torso={angle:.1f} deg > {self.cfg.torso_angle_threshold} deg")
        else:
            base = (f"H/W={ratio:.2f} < {self.cfg.aspect_ratio_threshold}  "
                    f"(no keypoint data)")

        required = (self.cfg.fast_confirmation_frames
                    if is_severe else self.cfg.confirmation_frames)
        suffix = f"  for {required} consecutive frames"
        if tags:
            suffix += f"  [{', '.join(tags)}]"

        return FallAlert(
            timestamp=ts,
            aspect_ratio=ratio,
            torso_angle=angle,
            bbox=self._latest_bbox,
            reason=base + suffix,
        )
