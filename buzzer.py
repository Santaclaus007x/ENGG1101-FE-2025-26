"""
Buzzer Controller for Fall + Wave Detection System
====================================================
Controls an active buzzer connected to a Raspberry Pi GPIO pin.
Falls back gracefully on non-Pi platforms (Windows/Mac development).

Wiring (default):
  Buzzer +  →  BCM GPIO 17  (physical pin 11)
  Buzzer −  →  GND

Override the pin with --buzzer-pin N when running main.py.
"""

from __future__ import annotations

import threading
import time

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    _GPIO_AVAILABLE = False


class Buzzer:
    """
    Active-buzzer driver.  Beep patterns run in daemon threads so
    they never block the main detection loop.
    """

    DEFAULT_PIN = 17  # BCM numbering

    def __init__(self, pin: int = DEFAULT_PIN):
        self.pin = pin
        self._lock = threading.Lock()

        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"[BUZZER] GPIO ready on BCM pin {self.pin}")
        else:
            print("[BUZZER] RPi.GPIO not available — buzzer disabled (non-Pi platform)")

    # ── public API ────────────────────────────────────────────────────

    def beep_fall(self):
        """3 rapid urgent beeps — triggered on fall detection."""
        pattern = [
            (0.12, 0.08),   # on 120 ms, off 80 ms
            (0.12, 0.08),
            (0.12, 0.00),
        ]
        self._start(pattern)

    def beep_wave(self):
        """2 friendly longer beeps — triggered when wave is confirmed."""
        pattern = [
            (0.30, 0.12),   # on 300 ms, off 120 ms
            (0.30, 0.00),
        ]
        self._start(pattern)

    def cleanup(self):
        """Release GPIO resources on shutdown."""
        if _GPIO_AVAILABLE:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup(self.pin)
            print("[BUZZER] GPIO cleanup done.")

    # ── internal ──────────────────────────────────────────────────────

    def _start(self, pattern: list):
        t = threading.Thread(
            target=self._run_pattern,
            args=(pattern,),
            daemon=True,
        )
        t.start()

    def _run_pattern(self, pattern: list):
        if not _GPIO_AVAILABLE:
            return
        with self._lock:
            try:
                for on_t, off_t in pattern:
                    GPIO.output(self.pin, GPIO.HIGH)
                    time.sleep(on_t)
                    GPIO.output(self.pin, GPIO.LOW)
                    if off_t > 0:
                        time.sleep(off_t)
            finally:
                GPIO.output(self.pin, GPIO.LOW)
