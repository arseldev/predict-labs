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

        # Sinyal simulasi baru dengan flat $1.00 USD
        self._position = {
            "entry_time": timestamp,
            "entry_price": entry_price,
            "direction": direction,
            "predicted_proba": proba,
            "candles_held": 0
        }
        
        logger.info(
            f"🚀 [SIMULATION ENTRY] Prediksi {direction} @ {entry_price:.2f} USDT (Bet Size: $1.00 USD)"
        )
        
        # Kirim notifikasi Telegram jika handler terdaftar
        self._send_telegram_notification(
            f"🚀 <b>SIMULASI ENTRY PREDIKSI 5M: {direction}</b>\n"
            f"Harga Entry: ${entry_price:,.2f}\n"
            f"Bet Size: $1.00 USD\n"
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
        
        close = float(candle_close_data["close"])
        open_val = float(candle_close_data["open"])
        close_time = candle_close_data["close_time"]
        
        # Tentukan tebakan benar atau salah
        if pos["direction"] == "UP":
            win = close > open_val
        else: # DOWN
            win = close < open_val
            
        draw = close == open_val
        
        if draw:
            exit_reason = "DRAW"
            net_pnl = 0.0
        elif win:
            exit_reason = "WIN"
            # Profit bersih $0.85 jika tebakan benar (asumsi payout multiplier 1.85x)
            net_pnl = 0.85
        else:
            exit_reason = "LOSS"
            # Rugi modal $1.00 jika tebakan salah
            net_pnl = -1.00
            
        self._close_position(close, exit_reason, net_pnl, close_time)

    def _close_position(self, exit_price: float, outcome: str, net_pnl: float, exit_time):
        pos = self._position
        
        # Update balance
        new_balance = self.simulated_balance + net_pnl
        self.update_balance(new_balance)
        
        trade_data = {
            "signal_timestamp": str(pos["entry_time"]),
            "entry_time": str(pos["entry_time"]),
            "exit_time": str(exit_time),
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "quantity": 1.0, # Taruhan flat $1.00
            "direction": pos["direction"].lower(),
            "gross_pnl": net_pnl,
            "net_pnl": net_pnl,
            "fee_paid": 0.0,
            "exit_reason": outcome,
            "predicted_proba": pos["predicted_proba"],
        }
        
        if self._pred_logger:
            self._pred_logger.log_trade(trade_data)
            logger.info(
                f"✅ [SIMULATION EXIT] {outcome} | Direction: {pos['direction']} | "
                f"Entry Price: {pos['entry_price']:.2f} → Exit Price: {exit_price:.2f} | "
                f"Net PnL: {net_pnl:+.2f} USD | Balance: ${new_balance:.2f}"
            )
        else:
            logger.warning("Warning: pred_logger tidak di-set di SimulationEngine, trade tidak disimpan ke DB!")
            
        # Kirim notifikasi Telegram
        pnl_icon = "🟢" if net_pnl > 0 else ("🟡" if net_pnl == 0 else "🔴")
        self._send_telegram_notification(
            f"{pnl_icon} <b>SIMULASI PREDIKSI 5M SELESAI: {outcome}</b>\n"
            f"Tebakan Arah: {pos['direction']}\n"
            f"Harga Entry: ${pos['entry_price']:,.2f}\n"
            f"Harga Exit (5m): ${exit_price:,.2f}\n"
            f"Hasil PnL: <b>{net_pnl:+.2f} USD</b>\n"
            f"Saldo Terkini: ${new_balance:,.2f}"
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
