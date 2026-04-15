"""
servo_tracker.py
================
Closed-loop pan/tilt controller for STS3215 servo motors.

Keeps a detected person centred in the camera frame using a simple
Proportional (P) controller.

Hardware setup
--------------
  Servo ID 1  →  pan  axis (rotates camera left / right)
  Servo ID 4  →  tilt axis (rotates camera up   / down)

  Connect both servos via the USB-to-serial adapter.
  Default baud rate: 115200 (must match the baud rate stored in each servo).
  Default port:      /dev/ttyUSB0  (check with `dmesg | grep tty` on the Pi)

How closed-loop works here
--------------------------
  Each call to update() does:
    1. READ the actual current positions from the servo (the feedback).
    2. Compute the pixel error between the target centre and the frame centre.
    3. WRITE new target = actual_position + proportional_correction.

  Because we always start from the real position, the command can never
  run away from reality — that's what makes it "closed-loop".

Tuning the P gain (Kp)
-----------------------
  Kp controls how aggressively the servo chases the target:
    - Too low  → slow to respond, person drifts out of frame
    - Too high → servo oscillates / jitters around the target
  Start at Kp=0.3 and adjust in steps of 0.05.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

# ── Locate scservo_sdk ─────────────────────────────────────────────────────
# scservo_sdk/ lives in the same directory as this file.
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
MIN_POS    = 0      # Minimum servo position
MAX_POS    = 4095   # Maximum servo position (~300° range for STS3215)
CENTER_POS = 2048   # Absolute midpoint of servo range
PAN_HOME   = 2374   # Camera "front" position for pan  (calibrated)
TILT_HOME  = 2050   # Camera "front" position for tilt (calibrated)

# STS3215: 4096 steps per full revolution → ~11.38 steps per degree
STEPS_PER_DEG: float = 4096 / 360


class ServoTracker:
    """
    Drives a pan/tilt servo pair to keep a target centred in frame.

    Parameters
    ----------
    port : str
        Serial port the servos are connected to (e.g. '/dev/ttyUSB0' or 'COM3').
    pan_id : int
        Servo ID for the pan  (left/right) axis. Default 1.
    tilt_id : int
        Servo ID for the tilt (up/down)    axis. Default 4.
    Kp : float
        Proportional gain. Higher = faster but potentially oscillatory.
    deadband_px : int
        If the target is within this many pixels of centre, do nothing.
        Prevents jitter when the person is roughly centred.
    speed : int
        Servo movement speed (0–4095). Lower is smoother.
    acc : int
        Servo acceleration (0–255). Lower gives gentler starts/stops.
    baudrate : int
        Serial baud rate. Must match the value stored in the servo.
    travel_deg : float
        Max degrees each servo can travel from the home position.
    """

    def __init__(
        self,
        port:        str,
        pan_id:      int   = 1,
        tilt_id:     int   = 4,
        Kp:          float = 0.3,
        deadband_px: int   = 30,
        speed:       int   = 1500,
        acc:         int   = 30,
        baudrate:    int   = 115200,
        travel_deg:  float = 180.0,
    ) -> None:
        self.pan_id      = pan_id
        self.tilt_id     = tilt_id
        self.Kp          = Kp
        self.deadband_px = deadband_px
        self.speed       = speed
        self.acc         = acc

        # Soft travel limits centred on home positions
        travel_steps   = int(travel_deg * STEPS_PER_DEG)
        self._pan_min  = max(MIN_POS, PAN_HOME  - travel_steps)
        self._pan_max  = min(MAX_POS, PAN_HOME  + travel_steps)
        self._tilt_min = max(MIN_POS, TILT_HOME - travel_steps)
        self._tilt_max = min(MAX_POS, TILT_HOME + travel_steps)

        # Cached positions (used as fallback when a read fails)
        self._pan_pos  = PAN_HOME
        self._tilt_pos = TILT_HOME

        # ── Open serial port ──
        self._port_handler   = PortHandler(port)
        self._packet_handler = sms_sts(self._port_handler)

        if not self._port_handler.openPort():
            raise RuntimeError(f"[ServoTracker] Failed to open port '{port}'")
        if not self._port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"[ServoTracker] Failed to set baud rate {baudrate}")

        print(f"[ServoTracker] Connected on {port} @ {baudrate} baud")

        # Move to home and read back real starting positions
        self.center()
        time.sleep(0.5)
        self._pan_pos, self._tilt_pos = self._read_positions()
        print(f"[ServoTracker] Initial positions — pan={self._pan_pos}, tilt={self._tilt_pos}")

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        target_cx: int,
        target_cy: int,
        frame_w:   int,
        frame_h:   int,
    ) -> None:
        """
        Move servos to keep (target_cx, target_cy) at the frame centre.

        Call once per frame after selecting the target person.

        Parameters
        ----------
        target_cx, target_cy : int
            Pixel coordinates of the target's bounding-box centre.
        frame_w, frame_h : int
            Width and height of the video frame in pixels.
        """
        frame_cx = frame_w / 2
        frame_cy = frame_h / 2

        error_x = target_cx - frame_cx   # positive → target is to the right
        error_y = target_cy - frame_cy   # positive → target is below centre

        # Deadband: ignore small errors to prevent jitter
        if abs(error_x) < self.deadband_px and abs(error_y) < self.deadband_px:
            return

        # Normalise error to [-1, +1]
        norm_x = error_x / frame_cx
        norm_y = error_y / frame_cy

        # Read actual servo positions (the closed-loop feedback)
        actual_pan, actual_tilt = self._read_positions()

        # Proportional correction
        delta_pan  = int(self.Kp * norm_x * CENTER_POS)
        delta_tilt = int(self.Kp * norm_y * CENTER_POS)

        new_pan  = _clamp(actual_pan  + delta_pan,  self._pan_min,  self._pan_max)
        new_tilt = _clamp(actual_tilt + delta_tilt, self._tilt_min, self._tilt_max)

        self._write_position(self.pan_id,  new_pan)
        self._write_position(self.tilt_id, new_tilt)

        self._pan_pos  = new_pan
        self._tilt_pos = new_tilt

    def center(self) -> None:
        """Move both servos to the home position (camera faces straight ahead)."""
        self._write_position(self.pan_id,  PAN_HOME)
        self._write_position(self.tilt_id, TILT_HOME)
        self._pan_pos  = PAN_HOME
        self._tilt_pos = TILT_HOME
        print("[ServoTracker] Homed.")

    def close(self) -> None:
        """Centre the servos and close the serial port cleanly."""
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
        pan  = self._read_one(self.pan_id,  self._pan_pos)
        tilt = self._read_one(self.tilt_id, self._tilt_pos)
        return pan, tilt

    def _read_one(self, servo_id: int, fallback: int) -> int:
        pos, _speed, result, error = self._packet_handler.ReadPosSpeed(servo_id)
        if result != COMM_SUCCESS or error != 0:
            return fallback
        return pos


# ── Utility ────────────────────────────────────────────────────────────────

def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ── Smoke-test ─────────────────────────────────────────────────────────────
# Run directly to verify the servo connection:
#   python servo_tracker.py --port /dev/ttyUSB0
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Servo tracker smoke-test")
    p.add_argument("--port", default="/dev/ttyUSB0",
                   help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    args = p.parse_args()

    print(f"Testing ServoTracker on {args.port} …")
    tracker = ServoTracker(port=args.port)
    print("Servos centred. Waiting 2 s …")
    time.sleep(2)
    tracker.close()
    print("Done.")
