"""
servo_test.py — Manual servo movement test
===========================================
Use this to verify the pan/tilt servos move in the right direction
and that the gears are working, WITHOUT running the full detection system.

How to run:
    python servo_test.py                  # default port /dev/ttyUSB0
    python servo_test.py --port /dev/ttyUSB1

Available commands (type and press Enter):
    up 15        — tilt camera UP   15 degrees
    down 10      — tilt camera DOWN 10 degrees
    left 30      — pan  camera LEFT 30 degrees
    right 45     — pan  camera RIGHT 45 degrees
    home         — return both servos to center (straight ahead)
    status       — show current position in degrees
    q / quit     — center servos and exit

Tip: just type  up  with no number to move 5 degrees (default step).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── Locate scservo_sdk ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS
except ImportError:
    print("ERROR: Cannot find scservo_sdk.")
    print("       Make sure the scservo_sdk/ folder is in the same directory as this script.")
    sys.exit(1)

# ── Servo constants (must match servo_tracker.py) ─────────────────────────────
STEPS_PER_DEG: float = 4096.0 / 360.0   # 11.377 steps per degree

PAN_HOME  = 2374    # steps — camera faces straight ahead (pan)
TILT_HOME = 1383    # steps — camera faces straight ahead (tilt, 121.55° from servo zero)

# Travel limits (same defaults as main.py)
PAN_LIMIT_DEG  = 90.0    # max degrees left or right from home
TILT_LIMIT_DEG = 45.0    # max degrees up or down from home

MIN_POS = 0
MAX_POS = 4095

DEFAULT_STEP_DEG = 5.0   # degrees moved when no number is given


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def steps_to_deg_from_home(steps: int, home: int) -> float:
    """Convert raw step position → degrees offset from home (+ = one dir, - = other)."""
    return (steps - home) / STEPS_PER_DEG


def write_pos(packet_handler, servo_id: int, position: int,
              speed: int = 800, acc: int = 20) -> bool:
    result, error = packet_handler.WritePosEx(servo_id, position, speed, acc)
    if result != COMM_SUCCESS:
        print(f"  [!] Write error on servo {servo_id}: "
              f"{packet_handler.getTxRxResult(result)}")
        return False
    if error != 0:
        print(f"  [!] Servo {servo_id} error byte: "
              f"{packet_handler.getRxPacketError(error)}")
        return False
    return True


def read_pos(packet_handler, servo_id: int, fallback: int) -> int:
    pos, _speed, result, error = packet_handler.ReadPosSpeed(servo_id)
    if result != COMM_SUCCESS or error != 0:
        return fallback
    return pos


def print_status(pan_steps: int, tilt_steps: int):
    pan_deg  = steps_to_deg_from_home(pan_steps,  PAN_HOME)
    tilt_deg = steps_to_deg_from_home(tilt_steps, TILT_HOME)

    # Human-readable direction labels
    pan_dir  = "RIGHT" if pan_deg  > 0 else ("LEFT"  if pan_deg  < 0 else "CENTER")
    tilt_dir = "DOWN"  if tilt_deg > 0 else ("UP"    if tilt_deg < 0 else "LEVEL")

    print()
    print(f"  Current position:")
    print(f"    Pan  (left/right): {abs(pan_deg):5.1f}° {pan_dir}  "
          f"  (limit: ±{PAN_LIMIT_DEG:.0f}°)")
    print(f"    Tilt (up/down)   : {abs(tilt_deg):5.1f}° {tilt_dir}  "
          f"  (limit: ±{TILT_LIMIT_DEG:.0f}°)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Manual servo movement test")
    p.add_argument("--port",    default="/dev/ttyUSB0",
                   help="Serial port for servos (default: /dev/ttyUSB0)")
    p.add_argument("--pan-id",  type=int, default=1,
                   help="Servo ID for pan  axis (default: 1)")
    p.add_argument("--tilt-id", type=int, default=4,
                   help="Servo ID for tilt axis (default: 4)")
    p.add_argument("--speed",   type=int, default=800,
                   help="Servo movement speed 0-4095 (default: 800)")
    args = p.parse_args()

    # ── Soft travel limits in steps ──
    pan_limit_steps  = int(PAN_LIMIT_DEG  * STEPS_PER_DEG)
    tilt_limit_steps = int(TILT_LIMIT_DEG * STEPS_PER_DEG)
    pan_min  = max(MIN_POS, PAN_HOME  - pan_limit_steps)
    pan_max  = min(MAX_POS, PAN_HOME  + pan_limit_steps)
    tilt_min = max(MIN_POS, TILT_HOME - tilt_limit_steps)  # up   = fewer steps
    tilt_max = min(MAX_POS, TILT_HOME + tilt_limit_steps)  # down = more  steps

    # ── Open serial port ──
    port_handler   = PortHandler(args.port)
    packet_handler = sms_sts(port_handler)

    if not port_handler.openPort():
        print(f"ERROR: Cannot open port '{args.port}'")
        print("       Check the USB cable and that the servo driver board is powered.")
        sys.exit(1)
    if not port_handler.setBaudRate(115200):
        print("ERROR: Cannot set baud rate.")
        port_handler.closePort()
        sys.exit(1)

    print()
    print("=" * 56)
    print("  SERVO MANUAL TEST  — connected on", args.port)
    print(f"  Pan  servo ID: {args.pan_id}   |   Tilt servo ID: {args.tilt_id}")
    print("=" * 56)
    print()
    print("  Commands:")
    print("    up [deg]    — tilt camera UP   (e.g.  up 15)")
    print("    down [deg]  — tilt camera DOWN (e.g.  down 10)")
    print("    left [deg]  — pan  camera LEFT (e.g.  left 30)")
    print("    right [deg] — pan  camera RIGHT(e.g.  right 45)")
    print("    home        — return to center (straight ahead)")
    print("    status      — show current position in degrees")
    print("    q / quit    — center and exit")
    print()
    print(f"  Default step if no number given: {DEFAULT_STEP_DEG:.0f} degrees")
    print()

    # Move to home to start from a known position
    print("  Moving to HOME position...")
    write_pos(packet_handler, args.pan_id,  PAN_HOME,  speed=args.speed)
    write_pos(packet_handler, args.tilt_id, TILT_HOME, speed=args.speed)
    time.sleep(0.8)

    # Read actual positions back
    pan_steps  = read_pos(packet_handler, args.pan_id,  PAN_HOME)
    tilt_steps = read_pos(packet_handler, args.tilt_id, TILT_HOME)

    print_status(pan_steps, tilt_steps)

    # ── Command loop ──
    while True:
        try:
            raw = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0]

        # Parse optional degree argument
        deg = DEFAULT_STEP_DEG
        if len(parts) >= 2:
            try:
                deg = float(parts[1])
                if deg <= 0:
                    print("  [!] Degrees must be a positive number.")
                    continue
            except ValueError:
                print(f"  [!] '{parts[1]}' is not a valid number.")
                continue

        if cmd in ("q", "quit", "exit"):
            break

        elif cmd == "home":
            print("  Returning to HOME — camera faces straight ahead.")
            write_pos(packet_handler, args.pan_id,  PAN_HOME,  speed=args.speed)
            write_pos(packet_handler, args.tilt_id, TILT_HOME, speed=args.speed)
            time.sleep(0.5)
            pan_steps  = read_pos(packet_handler, args.pan_id,  PAN_HOME)
            tilt_steps = read_pos(packet_handler, args.tilt_id, TILT_HOME)
            print_status(pan_steps, tilt_steps)

        elif cmd == "status":
            pan_steps  = read_pos(packet_handler, args.pan_id,  pan_steps)
            tilt_steps = read_pos(packet_handler, args.tilt_id, tilt_steps)
            print_status(pan_steps, tilt_steps)

        elif cmd == "up":
            delta = int(deg * STEPS_PER_DEG)
            new_tilt = _clamp(tilt_steps - delta, tilt_min, tilt_max)
            actual_deg = (tilt_steps - new_tilt) / STEPS_PER_DEG
            if actual_deg < 0.5:
                print(f"  [!] Already at the UP limit ({TILT_LIMIT_DEG:.0f}°). Cannot go further.")
            else:
                print(f"  Tilting UP {actual_deg:.1f}°  ...")
                write_pos(packet_handler, args.tilt_id, new_tilt, speed=args.speed)
                time.sleep(0.3)
                tilt_steps = read_pos(packet_handler, args.tilt_id, new_tilt)
                print_status(pan_steps, tilt_steps)

        elif cmd == "down":
            delta = int(deg * STEPS_PER_DEG)
            new_tilt = _clamp(tilt_steps + delta, tilt_min, tilt_max)
            actual_deg = (new_tilt - tilt_steps) / STEPS_PER_DEG
            if actual_deg < 0.5:
                print(f"  [!] Already at the DOWN limit ({TILT_LIMIT_DEG:.0f}°). Cannot go further.")
            else:
                print(f"  Tilting DOWN {actual_deg:.1f}°  ...")
                write_pos(packet_handler, args.tilt_id, new_tilt, speed=args.speed)
                time.sleep(0.3)
                tilt_steps = read_pos(packet_handler, args.tilt_id, new_tilt)
                print_status(pan_steps, tilt_steps)

        elif cmd == "left":
            delta = int(deg * STEPS_PER_DEG)
            new_pan = _clamp(pan_steps - delta, pan_min, pan_max)
            actual_deg = (pan_steps - new_pan) / STEPS_PER_DEG
            if actual_deg < 0.5:
                print(f"  [!] Already at the LEFT limit ({PAN_LIMIT_DEG:.0f}°). Cannot go further.")
            else:
                print(f"  Panning LEFT {actual_deg:.1f}°  ...")
                write_pos(packet_handler, args.pan_id, new_pan, speed=args.speed)
                time.sleep(0.3)
                pan_steps = read_pos(packet_handler, args.pan_id, new_pan)
                print_status(pan_steps, tilt_steps)

        elif cmd == "right":
            delta = int(deg * STEPS_PER_DEG)
            new_pan = _clamp(pan_steps + delta, pan_min, pan_max)
            actual_deg = (new_pan - pan_steps) / STEPS_PER_DEG
            if actual_deg < 0.5:
                print(f"  [!] Already at the RIGHT limit ({PAN_LIMIT_DEG:.0f}°). Cannot go further.")
            else:
                print(f"  Panning RIGHT {actual_deg:.1f}°  ...")
                write_pos(packet_handler, args.pan_id, new_pan, speed=args.speed)
                time.sleep(0.3)
                pan_steps = read_pos(packet_handler, args.pan_id, new_pan)
                print_status(pan_steps, tilt_steps)

        else:
            print(f"  [?] Unknown command '{cmd}'.")
            print("      Try: up, down, left, right, home, status, quit")

    # ── Cleanup ──
    print()
    print("  Centering servos before exit...")
    write_pos(packet_handler, args.pan_id,  PAN_HOME,  speed=500)
    write_pos(packet_handler, args.tilt_id, TILT_HOME, speed=500)
    time.sleep(0.8)
    port_handler.closePort()
    print("  Done. Servo port closed.")
    print()


if __name__ == "__main__":
    main()
