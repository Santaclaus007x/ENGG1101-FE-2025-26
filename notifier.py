"""
Discord Webhook Notifier
=========================
Sends a photo + embed to a Discord channel when a fall is detected.

Uses Discord Webhooks — no bot account, no library, just a URL.
Only requires the `requests` package.

Setup
-----
1. Open Discord → right-click any text channel → Edit Channel
2. Integrations → Webhooks → New Webhook → Copy Webhook URL
3. Pass the URL when running main.py:
       python main.py --discord-webhook https://discord.com/api/webhooks/...
"""

from __future__ import annotations

import json
import threading
import time

import cv2
import requests


class DiscordNotifier:
    """
    Fire-and-forget Discord alerts via webhook.
    Sends an embedded message + photo on fall detection.
    Runs in a daemon thread — never blocks the detection loop.
    Per-person cooldown prevents alert spam.
    """

    def __init__(self, webhook_url: str, cooldown: float = 30.0):
        self.webhook_url = webhook_url
        self.cooldown    = cooldown
        self._last_sent: dict[int, float] = {}
        print(f"[Discord] Notifier ready")
        self._test_connection()

    # ── Public API ─────────────────────────────────────────────────────────

    def send_fall_alert(self, person_id: int, frame=None) -> None:
        """Send a fall alert with an optional photo."""
        now = time.time()
        if now - self._last_sent.get(person_id, 0) < self.cooldown:
            return  # still in cooldown for this person
        self._last_sent[person_id] = now

        ts = time.strftime("%d %b %Y  %H:%M:%S")

        # Encode frame now on main thread before it gets overwritten
        jpeg_bytes = None
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpeg_bytes = buf.tobytes()

        t = threading.Thread(
            target=self._send,
            args=(person_id, ts, jpeg_bytes),
            daemon=True,
        )
        t.start()

    # ── Internal ───────────────────────────────────────────────────────────

    def _send(self, person_id: int, ts: str, jpeg_bytes: bytes | None) -> None:
        embed = {
            "title": "🚨 FALL DETECTED",
            "color": 0xFF0000,   # red
            "fields": [
                {"name": "Person",    "value": f"#{person_id}", "inline": True},
                {"name": "Time",      "value": ts,              "inline": True},
                {"name": "Action",    "value": "Please check immediately!", "inline": False},
            ],
            "footer": {"text": "ENGG1101 Fall Detection System"},
        }

        payload = {"embeds": [embed]}

        try:
            if jpeg_bytes:
                # Send photo + embed together
                requests.post(
                    self.webhook_url,
                    data={"payload_json": json.dumps(payload)},
                    files={"file": ("alert.jpg", jpeg_bytes, "image/jpeg")},
                    timeout=15,
                )
            else:
                requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=15,
                )
            print(f"[Discord] Fall alert sent for Person #{person_id}")
        except Exception as e:
            print(f"[Discord] Send failed: {e}")

    def _test_connection(self) -> None:
        """Send a startup message to confirm the webhook works."""
        payload = {
            "embeds": [{
                "title": "✅ Fall Detection System Online",
                "color": 0x00CC44,
                "description": "Monitoring started. Fall alerts will appear here.",
                "footer": {"text": "ENGG1101 Fall Detection System"},
            }]
        }
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            if r.status_code in (200, 204):
                print("[Discord] Connection confirmed — startup message sent.")
            else:
                print(f"[Discord] Webhook returned {r.status_code} — check the URL.")
        except Exception as e:
            print(f"[Discord] Connection test failed: {e}")
