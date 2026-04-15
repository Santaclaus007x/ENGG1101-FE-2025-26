"""
Telegram Alert Notifier
========================
Sends a photo + caption to a Telegram chat when a fall is detected.

Uses the Telegram Bot HTTP API directly — no extra library needed,
only the `requests` package (already in most Python environments).

Setup
-----
1. Open Telegram → search @BotFather → /newbot → follow steps → copy token.
2. Start a chat with your new bot (search its username, press Start).
3. Send the bot any message, then open in a browser:
       https://api.telegram.org/bot<TOKEN>/getUpdates
   Find "chat" → "id" — that is your chat_id.
4. Pass both when running main.py:
       python main.py --telegram-token <TOKEN> --telegram-chat <CHAT_ID>
"""

from __future__ import annotations

import threading
import time

import cv2
import requests

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    """
    Fire-and-forget Telegram alerts.

    Each alert runs in a daemon thread so it never blocks detection.
    Per-person cooldown prevents alert spam.
    """

    def __init__(self, token: str, chat_id: str, cooldown: float = 30.0):
        self.token    = token
        self.chat_id  = chat_id
        self.cooldown = cooldown
        self._last_sent: dict[int, float] = {}   # tid → last alert timestamp
        print(f"[Telegram] Notifier ready  chat_id={chat_id}")

    # ── Public API ─────────────────────────────────────────────────────────

    def send_fall_alert(self, person_id: int, frame=None) -> None:
        """Send a fall alert (with photo if frame is provided)."""
        now = time.time()
        if now - self._last_sent.get(person_id, 0) < self.cooldown:
            return  # still in cooldown for this person
        self._last_sent[person_id] = now

        ts      = time.strftime("%d %b %Y  %H:%M:%S")
        caption = (
            f"🚨 *FALL DETECTED*\n"
            f"Person: `#{person_id}`\n"
            f"Time:   `{ts}`\n"
            f"Please check immediately\!"
        )

        # Snapshot — encode now (on main thread) before the frame is reused
        jpeg_bytes = None
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpeg_bytes = buf.tobytes()

        t = threading.Thread(
            target=self._send,
            args=(caption, jpeg_bytes),
            daemon=True,
        )
        t.start()

    # ── Internal ───────────────────────────────────────────────────────────

    def _send(self, caption: str, jpeg_bytes: bytes | None) -> None:
        try:
            if jpeg_bytes:
                self._send_photo(caption, jpeg_bytes)
            else:
                self._send_text(caption)
            print("[Telegram] Alert sent.")
        except Exception as e:
            print(f"[Telegram] Send failed: {e}")

    def _send_photo(self, caption: str, jpeg_bytes: bytes) -> None:
        url = _TELEGRAM_API.format(token=self.token, method="sendPhoto")
        requests.post(
            url,
            data={"chat_id": self.chat_id, "caption": caption,
                  "parse_mode": "MarkdownV2"},
            files={"photo": ("alert.jpg", jpeg_bytes, "image/jpeg")},
            timeout=15,
        )

    def _send_text(self, text: str) -> None:
        url = _TELEGRAM_API.format(token=self.token, method="sendMessage")
        requests.post(
            url,
            data={"chat_id": self.chat_id, "text": text,
                  "parse_mode": "MarkdownV2"},
            timeout=15,
        )
