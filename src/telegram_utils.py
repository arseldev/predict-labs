import os
import requests
import threading

class TelegramBufferedHandler:
    """
    Custom log handler that buffers log messages and flushes them as a consolidated
    beautifully formatted HTML message to Telegram at the end of the prediction cycle.
    """
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.buffer = []
        self._lock = threading.Lock()

    def write(self, message):
        """Callback for loguru to write messages into our buffer."""
        msg_str = message.strip()
        if msg_str:
            with self._lock:
                self.buffer.append(msg_str)

    def flush_to_telegram(self, header: str = ""):
        """Sends the buffered logs to Telegram and clears the buffer."""
        with self._lock:
            if not self.buffer:
                return
            logs_to_send = list(self.buffer)
            self.buffer = []

        # Format logs beautifully
        formatted_lines = []
        in_code_block = False
        
        for line in logs_to_send:
            # Handle multi-line strings split
            for subline in line.split("\n"):
                subline_str = subline.strip()
                if not subline_str:
                    continue
                
                # Check if it matches loguru's default format "time | level | location - message"
                parts = subline_str.split(" | ", 2)
                if len(parts) >= 3:
                    # Close existing code block if active
                    if in_code_block:
                        formatted_lines.append("</code>")
                        in_code_block = False
                    
                    timestamp, level_str, rest = parts
                    level = level_str.strip()
                    
                    # Split location and message
                    rest_parts = rest.split(" - ", 1)
                    msg = rest_parts[1] if len(rest_parts) == 2 else rest
                    
                    emoji = "ℹ️"
                    if "ERROR" in level or "CRITICAL" in level:
                        emoji = "🔴"
                    elif "WARNING" in level:
                        emoji = "🟡"
                    elif "INFO" in level:
                        emoji = "🟢"
                    elif "SUCCESS" in level:
                        emoji = "✅"
                        
                    escaped_msg = (
                        msg.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    formatted_lines.append(f"{emoji} <b>{level}</b>: {escaped_msg}")
                else:
                    # Traceback / raw log extension
                    if not in_code_block:
                        formatted_lines.append("<code>")
                        in_code_block = True
                    escaped_subline = (
                        subline_str.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    formatted_lines.append(escaped_subline)
                    
        if in_code_block:
            formatted_lines.append("</code>")
            
        joined_body = "\n".join(formatted_lines)
        message_text = f"<b>{header}</b>\n\n{joined_body}"
        
        # Handle Telegram's max character limit (4096)
        if len(message_text) > 4000:
            message_text = message_text[:3900] + "\n... (truncated)"

        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message_text,
                "parse_mode": "HTML"
            }
            res = requests.post(self.api_url, json=payload, timeout=10)
            res.raise_for_status()
        except Exception as e:
            # Direct print to avoid recursion issues in loguru
            print(f"Error sending log to Telegram: {e}")
