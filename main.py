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

from fall_detector import FallDetector, FallDetectorConfig, FallAlert
from visualizer import draw_fall_overlay, draw_env_object
from buzzer import Buzzer


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

    wave_threshold = args.wave_seconds  # seconds required

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
            tracker="bytetrack.yaml",
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
                            _log_wave(tid, duration)
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


def _log_wave(person_id: int, duration: float):
    """Print wave detection to console."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] ** WAVE DETECTED **  Person #{person_id}  "
          f"duration={duration:.1f}s")


if __name__ == "__main__":
    main()
