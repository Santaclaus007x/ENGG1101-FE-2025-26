"""
Buzzer Controller for Fall + Wave Detection System
====================================================
Uses gpiozero (pre-installed on Raspberry Pi OS) to drive an active buzzer.
Falls back gracefully on non-Pi platforms (Windows/Mac development).

Wiring (default):
  Buzzer +  →  BCM GPIO 17  (physical pin 11)
  Buzzer −  →  GND
"""

from __future__ import annotations

import threading
import time

try:
    from gpiozero import OutputDevice
    _GPIO_AVAILABLE = True
except (ImportError, Exception):
    _GPIO_AVAILABLE = False


class Buzzer:
    """
    Active-buzzer driver using gpiozero.
    Beep patterns run in daemon threads so they never block detection.
    """

    DEFAULT_PIN = 17  # BCM numbering

    def __init__(self, pin: int = DEFAULT_PIN):
        self.pin = pin
        self._lock = threading.Lock()
        self._device = None

        if _GPIO_AVAILABLE:
            try:
                self._device = OutputDevice(pin, active_high=True, initial_value=False)
                print(f"[BUZZER] GPIO ready on BCM pin {self.pin}")
            except Exception as e:
                print(f"[BUZZER] GPIO init failed: {e} — buzzer disabled")
                self._device = None
        else:
            print("[BUZZER] gpiozero not available — buzzer disabled (non-Pi platform)")

    # ── public API ────────────────────────────────────────────────────

    def beep_fall(self):
        """3 rapid urgent beeps — triggered on fall detection."""
        pattern = [
            (0.12, 0.08),
            (0.12, 0.08),
            (0.12, 0.00),
        ]
        self._start(pattern)

    def beep_wave(self):
        """2 friendly longer beeps — triggered when wave is confirmed."""
        pattern = [
            (0.30, 0.12),
            (0.30, 0.00),
        ]
        self._start(pattern)

    def cleanup(self):
        """Release GPIO resources on shutdown."""
        if self._device:
            self._device.off()
            self._device.close()
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
        if not self._device:
            return
        with self._lock:
            try:
                for on_t, off_t in pattern:
                    self._device.on()
                    time.sleep(on_t)
                    self._device.off()
                    if off_t > 0:
                        time.sleep(off_t)
            finally:
                self._device.off()
