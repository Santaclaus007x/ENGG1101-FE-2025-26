"""
tilt_adjust.py
==============
Moves the tilt servo 10 degrees anti-clockwise and reports all positions.

Run on Pi:
    python tilt_adjust.py

Run with different port or servo ID:
    python tilt_adjust.py --port /dev/ttyUSB1 --tilt-id 4
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS

# ── Constants (must match servo_tracker.py) ───────────────────────────────────
STEPS_PER_DEG = 4096.0 / 360.0   # 11.377 steps per degree
ADJUST_DEG    = 10.0              # degrees to move anti-clockwise
ADJUST_STEPS  = int(ADJUST_DEG * STEPS_PER_DEG)   # ≈ 114 steps


def run(port: str, tilt_id: int, baudrate: int = 115200):
    # ── Open serial port ──────────────────────────────────────────────────────
    ph = PortHandler(port)
    pk = sms_sts(ph)

    if not ph.openPort():
        print(f"[ERROR] Cannot open port '{port}'")
        print("        Check USB connection and port name.")
        sys.exit(1)
    if not ph.setBaudRate(baudrate):
        print(f"[ERROR] Cannot set baud rate {baudrate}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Tilt Servo Adjustment  —  ID {tilt_id}  on {port}")
    print(f"{'='*50}")

    # ── Read current position ─────────────────────────────────────────────────
    pos_before, speed, result, error = pk.ReadPosSpeed(tilt_id)
    if result != COMM_SUCCESS or error != 0:
        print(f"[ERROR] Cannot read servo ID {tilt_id}")
        print(f"        result={pk.getTxRxResult(result)}  error={error}")
        print("        Is the servo powered and connected?")
        ph.closePort()
        sys.exit(1)

    angle_before = pos_before / STEPS_PER_DEG

    print(f"\n  BEFORE")
    print(f"    Position  : {pos_before} steps")
    print(f"    Angle     : {angle_before:.2f}°  (from servo zero)")

    # ── Calculate new position ────────────────────────────────────────────────
    # Anti-clockwise = decreasing position for STS3215
    pos_after = max(0, pos_before - ADJUST_STEPS)
    angle_after = pos_after / STEPS_PER_DEG
    actual_change_steps = pos_before - pos_after
    actual_change_deg   = actual_change_steps / STEPS_PER_DEG

    print(f"\n  MOVE")
    print(f"    Direction : Anti-clockwise")
    print(f"    Amount    : {ADJUST_DEG:.1f}° ({ADJUST_STEPS} steps requested)")
    print(f"    Actual    : {actual_change_deg:.2f}° ({actual_change_steps} steps applied)")

    # ── Write new position ────────────────────────────────────────────────────
    speed_val = 300   # slow and deliberate for calibration
    acc_val   = 10
    result, error = pk.WritePosEx(tilt_id, pos_after, speed_val, acc_val)
    if result != COMM_SUCCESS or error != 0:
        print(f"\n[ERROR] Write failed: {pk.getTxRxResult(result)}  error={error}")
        ph.closePort()
        sys.exit(1)

    # Wait for movement to complete
    print(f"\n  Moving ... ", end="", flush=True)
    time.sleep(1.5)
    print("done.")

    # ── Read back actual position ─────────────────────────────────────────────
    pos_readback, _, result2, error2 = pk.ReadPosSpeed(tilt_id)
    if result2 != COMM_SUCCESS or error2 != 0:
        pos_readback = pos_after   # fallback to commanded value

    angle_readback = pos_readback / STEPS_PER_DEG

    print(f"\n  AFTER")
    print(f"    Position  : {pos_readback} steps")
    print(f"    Angle     : {angle_readback:.2f}°  (from servo zero)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"    Before    : {pos_before} steps  ({angle_before:.2f}°)")
    print(f"    After     : {pos_readback} steps  ({angle_readback:.2f}°)")
    print(f"    Moved     : {pos_before - pos_readback} steps  "
          f"({(pos_before - pos_readback) / STEPS_PER_DEG:.2f}° anti-clockwise)")
    print(f"\n  ► If this position looks correct, update TILT_HOME in servo_tracker.py:")
    print(f"      TILT_HOME = {pos_readback}")
    print(f"{'='*50}\n")

    ph.closePort()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Move tilt servo 10° anti-clockwise")
    p.add_argument("--port",     default="/dev/ttyUSB0",
                   help="Serial port (default: /dev/ttyUSB0)")
    p.add_argument("--tilt-id", type=int, default=4,
                   help="Servo ID for tilt axis (default: 4)")
    p.add_argument("--baudrate", type=int, default=115200)
    args = p.parse_args()

    run(port=args.port, tilt_id=args.tilt_id, baudrate=args.baudrate)
