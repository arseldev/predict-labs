import os
import requests
import time
import threading
from loguru import logger

class TelegramNotifier:
    """
    Sederhana dan handal untuk mengirim notifikasi event-based ke Telegram.
    Dilengkapi minimal delay (rate limiting) agar tidak membombardir API Telegram.
    """
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._last_send_time = 0.0
        self._lock = threading.Lock()

    def send(self, text: str):
        """Kirim pesan langsung ke Telegram dengan parsing HTML."""
        with self._lock:
            # Cegah spam berlebih (rate limit minimal 1 detik)
            elapsed = time.time() - self._last_send_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            
            try:
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML"
                }
                res = requests.post(self.api_url, json=payload, timeout=8)
                res.raise_for_status()
                self._last_send_time = time.time()
            except Exception as e:
                # Gunakan print biasa untuk mencegah infinite loops dengan logging handlers
                print(f"Error sending Telegram notification: {e}")

