"""
servo_tracker.py
================
Closed-loop pan/tilt controller for STS3215 servo motors.

Keeps a detected person centred in the camera frame using a
Proportional (P) controller with FOV-based angle maths and
Exponential Moving Average (EMA) smoothing.

Hardware setup
--------------
  Servo ID 1  →  pan  axis (rotates camera left / right)
  Servo ID 4  →  tilt axis (rotates camera up   / down)

  Connect via USB-to-serial adapter.
  Default baud rate: 115200  (must match value stored in each servo)
  Default port:      /dev/ttyUSB0  (check with: dmesg | grep tty)

Why FOV-based maths?
---------------------
  A naive normalised-error approach (error / half_frame * CENTER_POS * Kp)
  scales the correction to the servo's full range, not to real-world
  angles.  Result: the same pixel error gives a very different angular
  correction depending on frame resolution.

  FOV-based maths instead converts the pixel error to an actual angle:
      angle_error = pixel_error * (camera_fov / frame_size)   [degrees]
      step_delta  = Kp * angle_error * STEPS_PER_DEGREE

  This is precise and resolution-independent.  Kp=1.0 means
  "correct 100 % of the angle error in one frame step".

Why EMA smoothing?
-------------------
  Without smoothing, each frame can produce a slightly different
  target because the bounding box jitters.  EMA damps this:
      smooth_t = alpha * raw_t  +  (1 - alpha) * smooth_{t-1}
  alpha=0.35 gives a good balance between lag and jitter suppression.

Tuning guide
------------
  Kp  (default 0.5):  Higher = faster chase, lower = slower but stable.
                      Oscillation → reduce Kp.  Sluggish → increase Kp.
  alpha (0–1):        Higher = less smoothing (snappier).
                      Lower  = more smoothing (slower but silky).
  deadband_px:        Zone where servos stay still.
                      Larger = less hunting, but less precise centering.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

# ── Locate scservo_sdk ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS
except ImportError as e:
    raise ImportError(
        f"Cannot import scservo_sdk from:\n  {_HERE}\n"
        f"Make sure the scservo_sdk/ folder is in the same directory as servo_tracker.py.\n"
        f"Original error: {e}"
    ) from e

# ── Servo constants ────────────────────────────────────────────────────────
MIN_POS    = 0
MAX_POS    = 4095
CENTER_POS = 2048
PAN_HOME   = 2374   # calibrated "camera faces forward" for pan
TILT_HOME  = 2050   # calibrated "camera faces forward" for tilt

# STS3215: 4096 steps / 360° = 11.377… steps per degree
STEPS_PER_DEG: float = 4096.0 / 360.0


class ServoTracker:
    """
    Drives a pan/tilt servo pair to keep a target centred in frame.

    Parameters
    ----------
    port : str
        Serial port (e.g. '/dev/ttyUSB0' or 'COM3').
    pan_id : int
        Servo ID for the pan  (left/right) axis. Default 1.
    tilt_id : int
        Servo ID for the tilt (up/down)    axis. Default 4.
    Kp : float
        Proportional gain applied to the angle error.
        Kp=1.0 corrects 100 % of the error per frame.
    deadband_px : int
        Pixel radius around frame centre where servos stay still.
    smooth_alpha : float  (0–1)
        EMA factor — higher = snappier, lower = smoother.
    cam_hfov : float
        Camera horizontal field of view in degrees (default 60°).
    cam_vfov : float
        Camera vertical field of view in degrees (default 40°).
    speed : int
        Servo movement speed (0–4095).
    acc : int
        Servo acceleration (0–255).
    baudrate : int
        Serial baud rate. Must match the servo's stored value.
    travel_deg : float
        Max degrees each servo travels from home in either direction.
    """

    def __init__(
        self,
        port:         str,
        pan_id:       int   = 1,
        tilt_id:      int   = 4,
        Kp:           float = 0.5,
        deadband_px:  int   = 20,
        smooth_alpha: float = 0.35,
        cam_hfov:     float = 60.0,
        cam_vfov:     float = 40.0,
        speed:        int   = 1200,
        acc:          int   = 20,
        baudrate:     int   = 115200,
        travel_deg:   float = 150.0,
    ) -> None:
        self.pan_id       = pan_id
        self.tilt_id      = tilt_id
        self.Kp           = Kp
        self.deadband_px  = deadband_px
        self._alpha       = smooth_alpha
        self.cam_hfov     = cam_hfov
        self.cam_vfov     = cam_vfov
        self.speed        = speed
        self.acc          = acc

        # Soft travel limits centred on home positions
        travel_steps   = int(travel_deg * STEPS_PER_DEG)
        self._pan_min  = max(MIN_POS, PAN_HOME  - travel_steps)
        self._pan_max  = min(MAX_POS, PAN_HOME  + travel_steps)
        self._tilt_min = max(MIN_POS, TILT_HOME - travel_steps)
        self._tilt_max = min(MAX_POS, TILT_HOME + travel_steps)

        # Cached + smoothed positions
        self._pan_pos     = PAN_HOME
        self._tilt_pos    = TILT_HOME
        self._smooth_pan  = float(PAN_HOME)
        self._smooth_tilt = float(TILT_HOME)

        # ── Open serial port ──
        self._port_handler   = PortHandler(port)
        self._packet_handler = sms_sts(self._port_handler)

        if not self._port_handler.openPort():
            raise RuntimeError(f"[ServoTracker] Failed to open port '{port}'")
        if not self._port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"[ServoTracker] Failed to set baud rate {baudrate}")

        print(f"[ServoTracker] Connected on {port} @ {baudrate} baud")
        print(f"[ServoTracker] Kp={Kp}  alpha={smooth_alpha}  "
              f"deadband={deadband_px}px  FOV={cam_hfov}°x{cam_vfov}°")

        # Move to home and seed EMA with the real read-back positions
        self.center()
        time.sleep(0.5)
        pan0, tilt0 = self._read_positions()
        self._pan_pos     = pan0
        self._tilt_pos    = tilt0
        self._smooth_pan  = float(pan0)
        self._smooth_tilt = float(tilt0)
        print(f"[ServoTracker] Ready — pan={pan0}, tilt={tilt0}")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        target_cx: int,
        target_cy: int,
        frame_w:   int,
        frame_h:   int,
    ) -> None:
        """
        Steer servos to keep (target_cx, target_cy) at the frame centre.

        Call once per frame.

        Parameters
        ----------
        target_cx, target_cy : pixel coordinates of the target centre.
        frame_w, frame_h     : frame dimensions in pixels.
        """
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0

        error_x = target_cx - frame_cx   # + = target is right of centre
        error_y = target_cy - frame_cy   # + = target is below centre

        # Deadband — skip if close enough
        if abs(error_x) <= self.deadband_px and abs(error_y) <= self.deadband_px:
            return

        # ── FOV-based angle error (degrees) ──────────────────────────────
        # pixel → angle using camera's actual field of view
        angle_x = error_x * (self.cam_hfov / frame_w)   # degrees, + = right
        angle_y = error_y * (self.cam_vfov / frame_h)   # degrees, + = down

        # ── Read actual servo positions (closed-loop feedback) ────────────
        actual_pan, actual_tilt = self._read_positions()

        # ── Proportional step correction ─────────────────────────────────
        # Kp=1.0 → fully correct the angle in one step
        delta_pan  = int(self.Kp * angle_x * STEPS_PER_DEG)
        delta_tilt = int(self.Kp * angle_y * STEPS_PER_DEG)

        raw_pan  = _clamp(actual_pan  + delta_pan,  self._pan_min,  self._pan_max)
        raw_tilt = _clamp(actual_tilt + delta_tilt, self._tilt_min, self._tilt_max)

        # ── EMA smoothing — prevents jitter from bounding-box noise ──────
        self._smooth_pan  = self._alpha * raw_pan  + (1.0 - self._alpha) * self._smooth_pan
        self._smooth_tilt = self._alpha * raw_tilt + (1.0 - self._alpha) * self._smooth_tilt

        cmd_pan  = _clamp(int(round(self._smooth_pan)),  self._pan_min,  self._pan_max)
        cmd_tilt = _clamp(int(round(self._smooth_tilt)), self._tilt_min, self._tilt_max)

        self._write_position(self.pan_id,  cmd_pan)
        self._write_position(self.tilt_id, cmd_tilt)

        self._pan_pos  = cmd_pan
        self._tilt_pos = cmd_tilt

    def center(self) -> None:
        """Move both servos to home (camera faces straight ahead)."""
        self._write_position(self.pan_id,  PAN_HOME)
        self._write_position(self.tilt_id, TILT_HOME)
        self._pan_pos     = PAN_HOME
        self._tilt_pos    = TILT_HOME
        self._smooth_pan  = float(PAN_HOME)
        self._smooth_tilt = float(TILT_HOME)
        print("[ServoTracker] Homed.")

    def close(self) -> None:
        """Centre servos and close the serial port."""
        self.center()
        time.sleep(0.3)
        self._port_handler.closePort()
        print("[ServoTracker] Port closed.")

    # ── Private helpers ────────────────────────────────────────────────────

    def _write_position(self, servo_id: int, position: int) -> None:
        result, error = self._packet_handler.WritePosEx(
            servo_id, position, self.speed, self.acc
        )
        if result != COMM_SUCCESS:
            print(f"[ServoTracker] Write error servo {servo_id}: "
                  f"{self._packet_handler.getTxRxResult(result)}")
        elif error != 0:
            print(f"[ServoTracker] Servo {servo_id} error byte: "
                  f"{self._packet_handler.getRxPacketError(error)}")

    def _read_positions(self) -> Tuple[int, int]:
        return (
            self._read_one(self.pan_id,  self._pan_pos),
            self._read_one(self.tilt_id, self._tilt_pos),
        )

    def _read_one(self, servo_id: int, fallback: int) -> int:
        pos, _speed, result, error = self._packet_handler.ReadPosSpeed(servo_id)
        if result != COMM_SUCCESS or error != 0:
            return fallback
        return pos


# ── Utility ────────────────────────────────────────────────────────────────

def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ── Smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Servo tracker smoke-test")
    p.add_argument("--port", default="/dev/ttyUSB0")
    args = p.parse_args()

    t = ServoTracker(port=args.port)
    print("Centred. Waiting 2 s …")
    time.sleep(2)
    t.close()
    print("Done.")
