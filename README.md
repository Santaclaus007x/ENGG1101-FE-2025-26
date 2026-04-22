# Vision-Based Elderly Monitoring System

Real-time fall detection + wave-to-help alerts + auto camera tracking.
Runs on a Raspberry Pi 5 with a webcam and two STS3215 servos.

## What it does
- Detects falls using YOLO-Pose (bounding box + spine angle)
- Sends Discord alerts with a snapshot when a fall or wave happens
- Camera auto-tracks whoever waves or gives a thumbs-up

---

## Quick start

### 1. Clone the repo
```bash
git clone https://github.com/Santaclaus007x/ENGG1101-FE-2025-26.git
cd ENGG1101-FE-2025-26
```

### 2. Install Python packages
```bash
pip install -r requirements.txt
```

On the **Raspberry Pi only**, also run:
```bash
sudo apt install swig liblgpio-dev
pip install lgpio
```

### 3. Plug in the hardware
- USB webcam → any Pi USB port
- Servo driver board → Pi USB port (shows up as `/dev/ttyUSB0`)
- Buzzer → Pi GPIO pin 17 (+) and GND (−)

### 4. Run it
```bash
python main.py
```

That's it. A window pops up showing the live feed. Press **q** to quit.

---

## Common run variations

| Command | What it does |
|---|---|
| `python main.py` | Normal run (everything on) |
| `python main.py --no-servo` | PC test — no servos connected |
| `python main.py --no-buzzer` | PC test — no buzzer |
| `python main.py --show-env` | Show the 17-point skeleton overlay (for the demo/poster) |
| `python main.py --source 1` | Use a different webcam index |
| `python main.py --output demo.mp4` | Save the session to a video file |

---

## Gestures (during a live run)

| Gesture | Hold time | What happens |
|---|---|---|
| 🙋 Wave (wrist above shoulder) | 2 seconds | Buzzer + Discord alert |
| 👍 Thumbs-up (arm straight up) | 1 second | Camera starts following you |
| 🤸 Fall (on the floor) | ~0.5 s | Buzzer + Discord alert with photo |

---

## Troubleshooting

**"No module named serial"** → `pip install pyserial`

**"No such file /dev/ttyUSB0"** → the servo board isn't plugged in, or Linux gave it a different name. Run `ls /dev/ttyUSB*` and use the right one:
```bash
python main.py --servo-port /dev/ttyUSB1
```

**"Cannot open source: 0"** → webcam isn't detected. Try `--source 1` or `--source 2`.

**Discord alerts not arriving** → make sure `config.py` exists with:
```python
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
```

**Servo isn't moving** → check the USB cable isn't a charge-only one, and that the servo driver board's power LED is on.

---

## Files you might touch

| File | What's in it |
|---|---|
| `main.py` | Main loop — run this |
| `config.py` | Discord webhook URL (not on GitHub — ask a teammate) |
| `tilt_info.txt` | Tilt servo calibration numbers |
| `bytetrack.yaml` | Multi-person tracker settings |

Everything else (`fall_detector.py`, `servo_tracker.py`, `notifier.py`, `buzzer.py`, `visualizer.py`) is internal — you don't need to edit those to run the system.
