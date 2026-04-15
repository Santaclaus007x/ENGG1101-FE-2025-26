"""
Real-Time Fall + Wave Detection
=================================
ENGG1101 Camera Project — HKU

Uses YOLO26 Nano Pose for person detection + pose keypoints.
  - Fall detection:  bounding box aspect ratio (H/W < threshold)
  - Wave detection:  wrist above shoulder for N seconds

Usage
-----
  # Live webcam (default — just run it)
  python main.py

  # Specific camera index
  python main.py --source 1

  # Tune sensitivity
  python main.py --aspect-ratio 0.8 --wave-seconds 5

  # Save output video
  python main.py --output result.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("=" * 50)
    print("  ultralytics is not installed.")
    print("  Run:  pip install ultralytics opencv-python")
    print("=" * 50)
    sys.exit(1)

# Custom ByteTrack config — lives next to this file
_BYTETRACK_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")

from fall_detector import FallDetector, FallDetectorConfig, FallAlert
from visualizer import draw_fall_overlay, draw_env_object
from buzzer import Buzzer

try:
    from servo_tracker import ServoTracker
    _SERVO_AVAILABLE = True
except ImportError:
    _SERVO_AVAILABLE = False


# ──────────────────────────────────────
# COCO Keypoint indices + names
# ──────────────────────────────────────
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


# ──────────────────────────────────────
# CLI arguments
# ──────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-Time Fall + Wave Detection (ENGG1101)"
    )
    p.add_argument("--source", type=str, default="0",
                   help="Camera index (0, 1, ...) or path to video file")
    p.add_argument("--model", type=str, default="yolo26n-pose.pt",
                   help="YOLO pose model file (default: yolo26n-pose.pt)")
    p.add_argument("--output", type=str, default=None,
                   help="Save annotated video to this path")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Detection confidence threshold (default: 0.4)")
    p.add_argument("--imgsz", type=int, default=384,
                   help="YOLO input resolution (lower = faster, default: 384)")
    p.add_argument("--aspect-ratio", type=float, default=0.7,
                   help="Aspect ratio (H/W) threshold — below = falling (default: 0.7)")
    p.add_argument("--wave-seconds", type=float, default=2.0,
                   help="Seconds wrist must stay above shoulder to trigger wave (default: 2.0)")
    p.add_argument("--show-env", action="store_true",
                   help="Show the environmental visual (pose skeleton and keypoints) on the video feed.")
    p.add_argument("--no-show", action="store_true",
                   help="Disable display window (headless mode)")
    p.add_argument("--buzzer-pin", type=int, default=17,
                   help="BCM GPIO pin for the buzzer (default: 17)")
    p.add_argument("--no-buzzer", action="store_true",
                   help="Disable buzzer output")
    p.add_argument("--servo-port", type=str, default="/dev/ttyUSB0",
                   help="Serial port for STS3215 servos (default: /dev/ttyUSB0)")
    p.add_argument("--pan-id",  type=int, default=1,
                   help="Servo ID for pan  axis (default: 1)")
    p.add_argument("--tilt-id", type=int, default=4,
                   help="Servo ID for tilt axis (default: 4)")
    p.add_argument("--no-servo", action="store_true",
                   help="Disable pan/tilt servo tracking")
    p.add_argument("--servo-kp", type=float, default=0.5,
                   help="Servo P-gain — fraction of angle error to correct per frame (default: 0.5)")
    p.add_argument("--servo-deadband", type=int, default=20,
                   help="Deadband radius in pixels before servos move (default: 20)")
    p.add_argument("--servo-alpha", type=float, default=0.35,
                   help="EMA smoothing factor 0–1, higher=snappier (default: 0.35)")
    p.add_argument("--cam-hfov", type=float, default=60.0,
                   help="Camera horizontal field of view in degrees (default: 60)")
    p.add_argument("--cam-vfov", type=float, default=40.0,
                   help="Camera vertical field of view in degrees (default: 40)")
    return p.parse_args()


# ──────────────────────────────────────
# Wave detection helper
# ──────────────────────────────────────
def check_waving(person_kps, conf_threshold: float = 0.5) -> bool:
    """
    Check if a person is raising either wrist above the corresponding shoulder.

    Parameters
    ----------
    person_kps : tensor of shape (17, 3) — x, y, confidence per keypoint
    conf_threshold : minimum keypoint confidence to consider

    Returns
    -------
    True if wrist is above shoulder (smaller Y = higher in image)
    """
    l_sh = person_kps[KP_LEFT_SHOULDER]
    r_sh = person_kps[KP_RIGHT_SHOULDER]
    l_wr = person_kps[KP_LEFT_WRIST]
    r_wr = person_kps[KP_RIGHT_WRIST]

    # Left wrist above left shoulder
    if float(l_wr[2]) > conf_threshold and float(l_sh[2]) > conf_threshold:
        if float(l_wr[1]) < float(l_sh[1]):
            return True

    # Right wrist above right shoulder
    if float(r_wr[2]) > conf_threshold and float(r_sh[2]) > conf_threshold:
        if float(r_wr[1]) < float(r_sh[1]):
            return True

    return False


# ──────────────────────────────────────
# Main real-time loop
# ──────────────────────────────────────
def main():
    args = parse_args()

    # ── Load YOLO Pose ──
    print(f"[INIT] Loading model: {args.model}")
    model = YOLO(args.model)
    print("[INIT] Model loaded successfully.")

    # ── Open camera / video ──
    source = int(args.source) if args.source.isdigit() else args.source

    # Try multiple camera indices — use platform-appropriate backend
    import platform
    cap = None
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        for idx in [source] + [i for i in range(5) if i != source]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"[INIT] Camera opened on index {idx}")
                break
            cap.release()
            cap = None
        # Fallback: try default backend if V4L2 failed
        if cap is None and platform.system() != "Windows":
            for idx in [source] + [i for i in range(5) if i != source]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    print(f"[INIT] Camera opened on index {idx} (default backend)")
                    break
                cap.release()
                cap = None
    else:
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        print("        Make sure your webcam is connected.")
        sys.exit(1)

    # Webcam optimizations
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INIT] Stream: {frame_w}x{frame_h} @ {cam_fps:.0f} FPS")

    # ── Video writer (optional) ──
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, cam_fps,
                                 (frame_w, frame_h))
        print(f"[INIT] Recording to: {args.output}")

    # ── Buzzer ──
    buzzer = None if args.no_buzzer else Buzzer(pin=args.buzzer_pin)

    # ── Servo tracker ──
    tracker = None
    if not args.no_servo and _SERVO_AVAILABLE:
        try:
            tracker = ServoTracker(
                port=args.servo_port,
                pan_id=args.pan_id,
                tilt_id=args.tilt_id,
                Kp=args.servo_kp,
                deadband_px=args.servo_deadband,
                smooth_alpha=args.servo_alpha,
                cam_hfov=args.cam_hfov,
                cam_vfov=args.cam_vfov,
            )
        except Exception as e:
            print(f"[WARN] Servo tracker disabled: {e}")
            tracker = None
    elif not _SERVO_AVAILABLE:
        print("[WARN] servo_tracker could not be imported — servo disabled")

    # ── Fall detector config ──
    config = FallDetectorConfig(
        aspect_ratio_threshold=args.aspect_ratio,
        confirmation_frames=int(cam_fps * 0.5),  # ~0.5 seconds
    )

    # ── Per-person state (keyed by ByteTrack ID) ──
    detectors: Dict[int, FallDetector] = {}
    last_seen: Dict[int, float] = {}
    active_alerts: Dict[int, tuple] = {}

    # ── Per-person wave state ──
    wave_start_times: Dict[int, float | None] = {}   # when wrist first raised
    wave_confirmed: Dict[int, bool] = {}              # True once threshold met
    wave_confirm_times: Dict[int, float] = {}         # timestamp of latest confirmation

    wave_threshold = args.wave_seconds  # seconds required

    # ── Servo target — sticky: tracks the most-recently-waving person ──
    servo_target_tid: int | None = None
    servo_target_wave_time: float = 0.0

    # ── Occlusion handling (Plans A + B) ──
    servo_last_cx:         int | None   = None   # last confirmed target centre X
    servo_last_cy:         int | None   = None   # last confirmed target centre Y
    servo_target_lost_time: float | None = None  # when target ID first disappeared
    SERVO_HOLD_SECS = 1.5   # hold at last position before attempting recovery

    # ── FPS measurement ──
    frame_count = 0
    fps_timer = time.time()
    display_fps = 0.0

    print()
    print("=" * 55)
    print("  FALL + WAVE DETECTION RUNNING — press 'q' to quit")
    print(f"  Fall:  Bounding Box Aspect Ratio < {args.aspect_ratio}")
    print(f"  Wave:  Wrist above shoulder for {wave_threshold:.0f}s")
    print("=" * 55)
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, int):
                print("[WARN] Camera frame grab failed, retrying...")
                continue
            else:
                print("[INFO] End of video file.")
                break

        now = time.time()
        frame_count += 1

        # ── FPS (update every 20 frames) ──
        if frame_count % 20 == 0:
            elapsed = now - fps_timer
            display_fps = 20.0 / elapsed if elapsed > 0 else 0
            fps_timer = now

        # ────────────────────────────────────────────
        # YOLO Pose Detection + ByteTrack
        # ────────────────────────────────────────────
        results = model.track(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            persist=True,
            tracker=_BYTETRACK_CFG,
            # Tracker handles all classes now (person, chair, bed, etc.)
        )

        result = results[0]

        # ── Draw environment visual (Skeleton & Labels) ──
        if args.show_env:
            frame = result.plot()   # Draws YOLO skeletons and boxes natively
            if result.keypoints is not None and result.keypoints.data is not None:
                for person_kps in result.keypoints.data:
                    for kp_idx, kp in enumerate(person_kps):
                        x, y, conf = int(kp[0]), int(kp[1]), float(kp[2])
                        if conf > 0.5:
                            label = f"{KEYPOINT_NAMES[kp_idx]}"
                            cv2.putText(
                                frame, label, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                                cv2.LINE_AA
                            )

        # ── Extract bounding boxes + track IDs ──
        has_boxes = (result.boxes is not None and
                     len(result.boxes) > 0)
        has_ids = (result.boxes is not None and
                   result.boxes.id is not None)

        # Keypoints from pose model
        has_keypoints = (result.keypoints is not None and
                         result.keypoints.data is not None and
                         len(result.keypoints.data) > 0)

        current_track_ids = set()
        wave_count = 0
        any_fall_active = False
        any_wave_active = False
        person_positions: Dict[int, tuple] = {}   # tid → (cx, cy) this frame

        if has_boxes and has_ids:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)   # (N, 4)
            track_ids = result.boxes.id.cpu().numpy().astype(int)  # (N,)
            classes = result.boxes.cls.cpu().numpy().astype(int)   # (N,)
            names = model.names

            # Get keypoints if available
            keypoints_data = None
            if has_keypoints:
                keypoints_data = result.keypoints.data  # (N, 17, 3)

            # ── Process each tracked object ──
            for i, (box, tid, cls) in enumerate(zip(boxes, track_ids, classes)):
                current_track_ids.add(tid)
                last_seen[tid] = now

                # If it's not a person, draw it as an environment object
                if cls != 0:
                    label = f"{names.get(cls, 'Object')} #{tid}"
                    draw_env_object(frame, tuple(box), label)
                    continue

                # ── Record centre position for servo target lookup ──
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                person_positions[tid] = (cx, cy)

                # ────────────────────────────
                # FALL DETECTION (bounding box for PERSON)
                # ────────────────────────────
                if tid not in detectors:
                    detectors[tid] = FallDetector(config=config)

                detector = detectors[tid]
                bbox = tuple(box)  # (x1, y1, x2, y2)
                alert = detector.update(bbox, timestamp=now)

                if alert is not None:
                    active_alerts[tid] = (alert, now + 3.0)
                    _log_alert(alert, tid)

                # Continuous fall state — beep while person is actually falling
                metrics = detector.get_latest_metrics()
                if metrics.get('aspect_ratio', 1.5) < metrics.get('threshold', 0.7):
                    any_fall_active = True

                # ────────────────────────────
                # WAVE DETECTION (keypoints)
                # ────────────────────────────
                wave_info = None
                if keypoints_data is not None and i < len(keypoints_data):
                    person_kps = keypoints_data[i]
                    is_waving_now = check_waving(person_kps)

                    if is_waving_now:
                        if wave_start_times.get(tid) is None:
                            wave_start_times[tid] = now
                        duration = now - wave_start_times[tid]
                        confirmed = duration >= wave_threshold
                        if confirmed and not wave_confirmed.get(tid, False):
                            wave_confirmed[tid] = True
                            wave_confirm_times[tid] = now
                            _log_wave(tid, duration)
                            # Switch servo target if this wave is more recent
                            if now >= servo_target_wave_time:
                                servo_target_tid = tid
                                servo_target_wave_time = now
                                print(f"[SERVO] New target: Person #{tid}")
                        wave_info = {
                            "is_waving": True,
                            "duration": duration,
                            "confirmed": confirmed,
                        }
                        if confirmed:
                            wave_count += 1
                            any_wave_active = True
                    else:
                        wave_start_times[tid] = None
                        wave_confirmed[tid] = False
                        wave_info = {
                            "is_waving": False,
                            "duration": 0.0,
                            "confirmed": False,
                        }

                # ── Draw overlay for this person ──
                metrics = detector.get_latest_metrics()
                current_alert = None
                if tid in active_alerts:
                    a, expire = active_alerts[tid]
                    if now < expire:
                        current_alert = a
                    else:
                        del active_alerts[tid]

                draw_fall_overlay(frame, metrics, current_alert, tid,
                                  wave_info=wave_info)

                # ── Servo target marker ──
                if tid == servo_target_tid:
                    _draw_servo_target(frame, cx, cy)

        # ── Servo: occlusion-robust sticky tracking ──────────────────────
        if tracker:
            if servo_target_tid is not None and servo_target_tid not in current_track_ids:
                # Target ID vanished — could be a passerby occlusion or brief drop
                if servo_target_lost_time is None:
                    servo_target_lost_time = now

                time_lost = now - servo_target_lost_time

                if time_lost < SERVO_HOLD_SECS:
                    # Plan B: hold at last known position — passerby will clear in < 1.5 s
                    if servo_last_cx is not None and person_positions:
                        tracker.update(servo_last_cx, servo_last_cy, frame_w, frame_h)
                else:
                    # Plan A: position-based recovery — pick person closest to last known pos
                    if person_positions and servo_last_cx is not None:
                        lx, ly = servo_last_cx, servo_last_cy
                        # Prefer people with wave history; otherwise anyone in frame
                        wave_pool = [t for t in person_positions if t in wave_confirm_times]
                        pool = wave_pool if wave_pool else list(person_positions.keys())
                        new_tid = min(pool, key=lambda t: (
                            (person_positions[t][0] - lx) ** 2 +
                            (person_positions[t][1] - ly) ** 2
                        ))
                        print(f"[SERVO] Position recovery → #{new_tid} "
                              f"(lost {time_lost:.1f}s)")
                        servo_target_tid = new_tid
                        servo_target_wave_time = wave_confirm_times.get(
                            new_tid, servo_target_wave_time)
                    servo_target_lost_time = None
            else:
                servo_target_lost_time = None  # target visible — reset timer

            if servo_target_tid in person_positions:
                tcx, tcy = person_positions[servo_target_tid]
                servo_last_cx, servo_last_cy = tcx, tcy   # save for Plans A/B
                tracker.update(tcx, tcy, frame_w, frame_h)

        # ── Buzzer: continuous beep while condition is active ──
        if buzzer:
            if any_fall_active:
                buzzer.start_fall_beep()
            elif any_wave_active:
                buzzer.start_wave_beep()
            else:
                buzzer.stop()

        # ── Cleanup stale trackers (not seen for 5s) ──
        stale = [tid for tid, t in last_seen.items() if now - t > 5.0]
        for tid in stale:
            detectors.pop(tid, None)
            last_seen.pop(tid, None)
            active_alerts.pop(tid, None)
            wave_start_times.pop(tid, None)
            wave_confirmed.pop(tid, None)
            wave_confirm_times.pop(tid, None)

        # ── Draw status bar ──
        _draw_status_bar(frame, display_fps, len(current_track_ids),
                         wave_count, frame_w, frame_h)

        # ── Output ──
        if writer:
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("Fall + Wave Detection - ENGG1101", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Quit requested.")
                break

    # ── Cleanup ──
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if tracker:
        tracker.close()
    if buzzer:
        buzzer.cleanup()
    print("[INFO] Done.")


# ──────────────────────────────────────
# Helpers
# ──────────────────────────────────────
def _draw_status_bar(frame, fps, n_persons, n_waving, w, h):
    """Bottom status bar with FPS, person count, and wave count."""
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    text = (f"FPS: {fps:.0f}  |  Tracking: {n_persons} person(s)  |  "
            f"Waving: {n_waving}  |  Press 'q' to quit")
    cv2.putText(frame, text, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)


def _log_alert(alert: FallAlert, person_id: int):
    """Print fall alert to console."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] !! FALL DETECTED !!  Person #{person_id}  "
          f"H/W={alert.aspect_ratio:.2f}  "
          f"bbox={alert.bbox}")


def _draw_servo_target(frame, cx: int, cy: int):
    """Yellow crosshair + ring to mark the person the servo is tracking."""
    col = (0, 220, 255)   # yellow
    r = 28
    cv2.circle(frame, (cx, cy), r, col, 2, cv2.LINE_AA)
    cv2.line(frame, (cx - r - 8, cy), (cx + r + 8, cy), col, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r - 8), (cx, cy + r + 8), col, 1, cv2.LINE_AA)
    cv2.putText(frame, "TRACKING", (cx - 35, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def _log_wave(person_id: int, duration: float):
    """Print wave detection to console."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] ** WAVE DETECTED **  Person #{person_id}  "
          f"duration={duration:.1f}s")


if __name__ == "__main__":
    main()
