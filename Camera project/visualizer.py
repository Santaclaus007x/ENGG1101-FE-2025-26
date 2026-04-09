"""
Visualization overlay for the Fall + Wave Detection System
============================================================
Draws bounding boxes (green = standing, orange = warning, red = falling,
blue = waving), aspect ratio labels, wave status, and alert banners.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import cv2
import numpy as np

from fall_detector import FallAlert


# ──────────────────────────────────────
# Color palette  (BGR for OpenCV)
# ──────────────────────────────────────
COL_BOX_STANDING    = (0, 220, 100)      # green
COL_BOX_FALLING     = (0, 0, 255)        # red
COL_BOX_WARNING     = (0, 180, 255)      # orange
COL_BOX_WAVING      = (255, 180, 0)      # blue-cyan
COL_BOX_ENV         = (150, 150, 150)    # gray for environment objects
COL_TEXT_BG         = (30, 30, 30)       # dark gray
COL_TEXT_WHITE      = (255, 255, 255)
COL_ALERT_BG        = (0, 0, 180)        # dark red
COL_WAVE_BANNER_BG  = (180, 120, 0)      # dark blue-teal
COL_ALERT_TEXT      = (255, 255, 255)
COL_PANEL_BG        = (0, 0, 0)


def draw_fall_overlay(frame: np.ndarray,
                      metrics: Dict[str, Any],
                      alert: Optional[FallAlert] = None,
                      person_id: int = 0,
                      wave_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Draw bounding box + aspect ratio + wave overlay onto the frame (in-place).

    Parameters
    ----------
    frame      : BGR image (H, W, 3)
    metrics    : dict from FallDetector.get_latest_metrics()
    alert      : FallAlert if one was just triggered
    person_id  : for multi-person labeling
    wave_info  : optional dict with keys 'is_waving', 'duration', 'confirmed'

    Returns
    -------
    frame with overlays drawn
    """
    if not metrics:
        return frame

    bbox = metrics.get("bbox")
    ratio = metrics.get("aspect_ratio", 1.5)
    threshold = metrics.get("threshold", 0.7)

    if bbox is None:
        return frame

    x1, y1, x2, y2 = [int(v) for v in bbox]

    # ── 1. Choose color based on state ──
    is_wave_confirmed = wave_info and wave_info.get("confirmed", False)

    if ratio < threshold:
        col = COL_BOX_FALLING
        status = "FALLING"
    elif ratio < threshold + 0.3:
        col = COL_BOX_WARNING
        status = "WARNING"
    else:
        col = COL_BOX_STANDING
        status = "Standing"

    # If waving and not falling, highlight with waving color
    if is_wave_confirmed and status != "FALLING":
        col = COL_BOX_WAVING

    # ── 2. Draw bounding box ──
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

    # ── 3. Label: Person ID + aspect ratio + status ──
    label = f"#{person_id}  H/W={ratio:.2f}  [{status}]"
    if is_wave_confirmed:
        label += "  [WAVING]"
    _draw_label(frame, label, (x1, y1 - 10), col)

    # ── 4. Draw dimensions on the box ──
    box_w = x2 - x1
    box_h = y2 - y1
    # Width label (bottom)
    w_label = f"W={box_w}"
    mid_x = (x1 + x2) // 2
    cv2.putText(frame, w_label, (mid_x - 20, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    # Height label (right side)
    h_label = f"H={box_h}"
    mid_y = (y1 + y2) // 2
    cv2.putText(frame, h_label, (x2 + 5, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    # ── 5. HUD panel (top-left) ──
    _draw_hud_panel(frame, ratio, threshold, person_id, status, wave_info)

    # ── 6. Alert banner (fall) ──
    if alert is not None:
        _draw_alert_banner(frame, alert)

    # ── 7. Wave banner (bottom) ──
    if is_wave_confirmed:
        _draw_wave_banner(frame, person_id, wave_info.get("duration", 0.0))

    return frame


def draw_env_object(frame: np.ndarray, bbox: tuple, label: str) -> np.ndarray:
    """Draw a bounding box and label for a non-person environment object."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    col = COL_BOX_ENV
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    _draw_label(frame, label, (x1, y1 - 10), col)
    return frame


def _draw_hud_panel(frame: np.ndarray,
                    ratio: float, threshold: float,
                    person_id: int, status: str,
                    wave_info: Optional[Dict[str, Any]] = None):
    """Compact semi-transparent metrics panel in the top-left corner."""
    has_wave = wave_info is not None
    panel_w, panel_h = 260, 90 if has_wave else 70
    x0, y0 = 10, 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Border
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (80, 80, 80), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    line_h = 20

    lines = [
        f"Person #{person_id}   Status: {status}",
        f"Aspect Ratio (H/W): {ratio:.2f}",
        f"Threshold:          {threshold:.2f}",
    ]

    if has_wave:
        is_waving = wave_info.get("is_waving", False)
        duration = wave_info.get("duration", 0.0)
        confirmed = wave_info.get("confirmed", False)
        wave_str = "YES" if confirmed else (f"Raising hand ({duration:.1f}s)" if is_waving else "No")
        lines.append(f"Wave:               {wave_str}")

    for i, text in enumerate(lines):
        col = COL_TEXT_WHITE
        if i == 1 and ratio < threshold:
            col = COL_BOX_FALLING
        elif i == 1 and ratio < threshold + 0.3:
            col = COL_BOX_WARNING
        elif i == 3 and has_wave and wave_info.get("confirmed", False):
            col = COL_BOX_WAVING
        cv2.putText(frame, text, (x0 + 8, y0 + 18 + i * line_h),
                    font, scale, col, 1, cv2.LINE_AA)


def _draw_alert_banner(frame: np.ndarray, alert: FallAlert):
    """Large alert banner across the top of the frame."""
    h, w = frame.shape[:2]
    banner_h = 50

    cv2.rectangle(frame, (0, 0), (w, banner_h), COL_ALERT_BG, -1)

    text = f"!! FALL DETECTED !!   H/W = {alert.aspect_ratio:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    tx = (w - tw) // 2
    ty = (banner_h + th) // 2
    cv2.putText(frame, text, (tx, ty), font, scale, COL_ALERT_TEXT, 2,
                cv2.LINE_AA)


def _draw_wave_banner(frame: np.ndarray, person_id: int, duration: float):
    """Wave alert banner across the bottom of the frame (above the status bar)."""
    h, w = frame.shape[:2]
    banner_h = 40
    y_start = h - 28 - banner_h  # above the status bar

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + banner_h),
                  COL_WAVE_BANNER_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    text = f"WAVE DETECTED — Person #{person_id} ({duration:.1f}s)"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    tx = (w - tw) // 2
    ty = y_start + (banner_h + th) // 2
    cv2.putText(frame, text, (tx, ty), font, scale, COL_ALERT_TEXT, 2,
                cv2.LINE_AA)


def _draw_label(frame: np.ndarray, text: str,
                pos: tuple, color: tuple):
    """Text label with dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + 4),
                  COL_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
