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
    Supports continuous looping beep that runs until explicitly stopped.
    """

    DEFAULT_PIN = 17  # BCM numbering

    def __init__(self, pin: int = DEFAULT_PIN):
        self.pin = pin
        self._device = None
        self._stop_event = threading.Event()
        self._current_mode: str | None = None  # 'fall', 'wave', or None

        if _GPIO_AVAILABLE:
            try:
                self._device = OutputDevice(pin, active_high=True, initial_value=False)
                print(f"[BUZZER] GPIO ready on BCM pin {self.pin}")
            except Exception as e:
                print(f"[BUZZER] GPIO init failed: {e} — buzzer disabled")
        else:
            print("[BUZZER] gpiozero not available — buzzer disabled (non-Pi platform)")

    # ── public API ────────────────────────────────────────────────────

    def start_fall_beep(self):
        """Rapid continuous beeping while a fall is active."""
        if self._current_mode == 'fall':
            return  # already beeping in this mode
        self._start_loop(on_t=0.1, off_t=0.1, mode='fall')

    def start_wave_beep(self):
        """Slower continuous beeping while a wave is active."""
        if self._current_mode == 'wave':
            return  # already beeping in this mode
        self._start_loop(on_t=0.3, off_t=0.3, mode='wave')

    def stop(self):
        """Stop all beeping."""
        if self._current_mode is None:
            return  # already silent
        self._current_mode = None
        self._stop_event.set()

    def cleanup(self):
        """Release GPIO resources on shutdown."""
        self.stop()
        if self._device:
            self._device.off()
            self._device.close()
            print("[BUZZER] GPIO cleanup done.")

    # ── internal ──────────────────────────────────────────────────────

    def _start_loop(self, on_t: float, off_t: float, mode: str):
        # Signal old thread to stop, then give it a fresh event
        self._stop_event.set()
        self._stop_event = threading.Event()
        self._current_mode = mode
        t = threading.Thread(
            target=self._loop,
            args=(on_t, off_t, self._stop_event),
            daemon=True,
        )
        t.start()

    def _loop(self, on_t: float, off_t: float, stop_event: threading.Event):
        if not self._device:
            return
        while not stop_event.is_set():
            self._device.on()
            stop_event.wait(on_t)   # wakes immediately if stopped
            self._device.off()
            stop_event.wait(off_t)
        self._device.off()
