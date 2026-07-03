"""
simulation_engine.py — Engine Simulasi Trading Real-time
Mensimulasikan entry, TP/SL, dan exit tanpa menggunakan Binance API untuk order execution.
Mencatat hasil simulasi trade ke SQLite db.
"""

from datetime import datetime, timezone
import os
import sqlite3
import pandas as pd
from loguru import logger

class SimulationEngine:
    def __init__(self, config: dict, pred_logger=None):
        self.config = config
        self.symbol = config["binance"].get("symbol", "BTCUSDT")
        
        # Risk protection parameters
        self.position_size_pct = config["trading"].get("position_size_pct", 0.02)
        self.tp_pct = config["trading"].get("profit_target_pct", 0.006)
        self.sl_pct = config["trading"].get("stop_loss_pct", 0.004)
        self.max_hold_candles = config["trading"].get("max_hold_candles", 12)
        self.fee_pct = config["trading"].get("fee_taker", 0.001) + config["trading"].get("slippage_pct", 0.0002)
        
        self._pred_logger = pred_logger
        self._position = None  # Menyimpan posisi aktif
        self.simulated_balance = 10000.0  # Balance simulasi awal $10,000
        
        # Inisialisasi DB untuk menyimpan balance simulasi harian/transaksi
        self.db_path = config["logging"]["db_path"]
        self._init_sim_balance()
        
    def _init_sim_balance(self):
        """Inisialisasi tabel balance simulasi jika belum ada."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sim_balance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                balance REAL NOT NULL
            )
        """)
        # Cek apakah sudah ada balance tercatat, jika belum masukkan default
        cursor.execute("SELECT balance FROM sim_balance ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            self.simulated_balance = row[0]
            logger.info(f"Loaded simulated balance from DB: ${self.simulated_balance:.2f}")
        else:
            cursor.execute("INSERT INTO sim_balance (balance) VALUES (?)", (self.simulated_balance,))
            conn.commit()
            logger.info(f"Initialized simulated balance in DB: ${self.simulated_balance:.2f}")
        conn.close()

    def update_balance(self, new_balance: float):
        self.simulated_balance = new_balance
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sim_balance (balance) VALUES (?)", (new_balance,))
        conn.commit()
        conn.close()

    def submit_order(self, direction: str, proba: float, entry_price: float, timestamp):
        """
        Simulasi masuk posisi (UP / DOWN) di awal candle baru.
        entry_price adalah harga close dari candle 5m yang baru ditutup (paling mendekati harga open candle baru).
        """
        if self._position is not None:
            logger.debug("Sudah ada posisi simulasi aktif, lewati sinyal baru.")
            return

        # Hitung quantity simulasi berdasarkan flat $1.00 USD per trade
        position_size_usd = 1.0
        quantity = position_size_usd / entry_price
        
        # Hitung level TP/SL
        if direction == "UP":
            tp_price = entry_price * (1.0 + self.tp_pct)
            sl_price = entry_price * (1.0 - self.sl_pct)
        else:  # DOWN (Short)
            tp_price = entry_price * (1.0 - self.tp_pct)
            sl_price = entry_price * (1.0 + self.sl_pct)
            
        self._position = {
            "entry_time": timestamp,
            "entry_price": entry_price,
            "quantity": quantity,
            "direction": direction,
            "predicted_proba": proba,
            "fee_paid": position_size_usd * self.fee_pct,
            "candles_held": 0,
            "tp_price": round(tp_price, 2),
            "sl_price": round(sl_price, 2),
        }
        
        logger.info(
            f"🚀 [SIMULATION ENTRY] Masuk {direction} @ {entry_price:.2f} USDT | "
            f"TP: {self._position['tp_price']:.2f} | SL: {self._position['sl_price']:.2f} | "
            f"Quantity: {quantity:.5f} BTC"
        )
        
        # Kirim notifikasi Telegram jika handler terdaftar
        self._send_telegram_notification(
            f"🚀 <b>SIMULASI ENTRY: {direction}</b>\n"
            f"Harga Entry: ${entry_price:,.2f}\n"
            f"Target TP: ${self._position['tp_price']:,.2f} (+{self.tp_pct:.2%})\n"
            f"Target SL: ${self._position['sl_price']:,.2f} (-{self.sl_pct:.2%})\n"
            f"Confidence: {proba:.2%}"
        )

    def process_candle(self, candle_close_data: dict):
        """
        Dipanggil setiap kali candle 5m ditutup untuk memperbarui status posisi simulasi aktif.
        candle_close_data: dict berisi open, high, low, close, close_time
        """
        if self._position is None:
            return

        pos = self._position
        pos["candles_held"] += 1
        
        high = float(candle_close_data["high"])
        low = float(candle_close_data["low"])
        close = float(candle_close_data["close"])
        close_time = candle_close_data["close_time"]
        
        exit_price = None
        exit_reason = None
        
        # Cek TP/SL
        if pos["direction"] == "UP":
            # Jika high menyentuh TP dan low menyentuh SL pada candle yang sama, kita asumsikan SL kena duluan (konservatif)
            if high >= pos["tp_price"] and low <= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "stop_loss"
            elif low <= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "stop_loss"
            elif high >= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_reason = "take_profit"
        else:  # DOWN (Short)
            if low <= pos["tp_price"] and high >= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "stop_loss"
            elif high >= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "stop_loss"
            elif low <= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_reason = "take_profit"
                
        # Cek Timeout
        if exit_price is None and pos["candles_held"] >= self.max_hold_candles:
            exit_price = close
            exit_reason = "timeout"
            
        # Jika exit terdeteksi, catat trade
        if exit_price is not None:
            self._close_position(exit_price, exit_reason, close_time)

    def _close_position(self, exit_price: float, exit_reason: str, exit_time):
        pos = self._position
        
        # Hitung PnL
        if pos["direction"] == "UP":
            gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
        else:  # DOWN
            gross_pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            
        fee_exit = exit_price * pos["quantity"] * self.fee_pct
        net_pnl = gross_pnl - pos["fee_paid"] - fee_exit
        
        # Update balance
        new_balance = self.simulated_balance + net_pnl
        self.update_balance(new_balance)
        
        trade_data = {
            "signal_timestamp": str(pos["entry_time"]),
            "entry_time": str(pos["entry_time"]),
            "exit_time": str(exit_time),
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "quantity": pos["quantity"],
            "direction": pos["direction"].lower(),
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "fee_paid": pos["fee_paid"] + fee_exit,
            "exit_reason": exit_reason,
            "predicted_proba": pos["predicted_proba"],
        }
        
        if self._pred_logger:
            self._pred_logger.log_trade(trade_data)
            logger.info(
                f"✅ [SIMULATION EXIT] {exit_reason.upper()} | Direction: {pos['direction']} | "
                f"Entry: {pos['entry_price']:.2f} → Exit: {exit_price:.2f} | "
                f"Net PnL: ${net_pnl:.2f} | Balance Baru: ${new_balance:.2f}"
            )
        else:
            logger.warning("Warning: pred_logger tidak di-set di SimulationEngine, trade tidak disimpan ke DB!")
            
        # Kirim notifikasi Telegram
        pnl_icon = "🟢" if net_pnl > 0 else "🔴"
        self._send_telegram_notification(
            f"{pnl_icon} <b>SIMULASI EXIT: {exit_reason.upper()}</b>\n"
            f"Arah Trade: {pos['direction']}\n"
            f"Harga Entry: ${pos['entry_price']:,.2f}\n"
            f"Harga Exit: ${exit_price:,.2f}\n"
            f"Net PnL: <b>${net_pnl:+.2f}</b>\n"
            f"Balance Simulasi: ${new_balance:,.2f}"
        )
        
        # Reset posisi
        self._position = None

    def _send_telegram_notification(self, text: str):
        bot_token = os.getenv("BOT_TOKEN")
        bot_target_id = os.getenv("BOT_TARGET_ID")
        if bot_token and bot_target_id:
            import requests
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            try:
                payload = {
                    "chat_id": bot_target_id,
                    "text": text,
                    "parse_mode": "HTML"
                }
                res = requests.post(url, json=payload, timeout=5)
                res.raise_for_status()
            except Exception as e:
                logger.error(f"Gagal mengirim notifikasi Telegram: {e}")
