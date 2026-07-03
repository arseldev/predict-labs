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
        self.simulated_balance = 50.0  # Balance simulasi awal $50

        
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

    def _fetch_live_pool_ratio(self) -> float:
        """
        Mengambil rasio pool UP / (UP + DOWN) secara real-time dari API Binance Predict.
        Menggunakan requests ke endpoint public prediction.
        """
        import requests
        try:
            # Menggunakan endpoint Binance Predict (Web3 / Prediction Market)
            # Karena Binance menggunakan base URL api.binance.com atau endpoint spesifik
            # Kita targetkan API ticker / order book prediction
            url = "https://api.binance.com/sapi/v1/w3w/wallet/prediction/market/search"
            # Note: Dalam real-world production, kita butuh API keys & marketId spesifik.
            # Di sini kita fetch dengan parameter dummy atau get public tickers
            params = {"symbol": self.symbol, "status": "TRADING"}
            response = requests.get(url, params=params, timeout=5, verify=False)
            if response.status_code == 200:
                data = response.json()
                # Parsing rasio UP vs DOWN dari data pool API
                # Misal: marketInfo -> poolRatioUp atau similar keys
                if "markets" in data and len(data["markets"]) > 0:
                    market = data["markets"][0]
                    # Cari total value UP vs total value DOWN
                    total_up = float(market.get("poolUp", 0))
                    total_down = float(market.get("poolDown", 0))
                    if total_up + total_down > 0:
                        return total_up / (total_up + total_down)
            
            # Fallback ke public ticker order book untuk estimasi rasio pasar jika endpoint prediction restricted
            # Ticker harga pasar token prediksi (biasanya berupa pair instrumen biner di market orderbook)
            ticker_url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={self.symbol}"
            ticker_res = requests.get(ticker_url, timeout=5, verify=False)
            if ticker_res.status_code == 200:
                # Simulasi rasio berdasarkan supply bid/ask jika REST API prediction butuh API signature
                tick = ticker_res.json()
                bid_qty = float(tick.get("bidQty", 1))
                ask_qty = float(tick.get("askQty", 1))
                return bid_qty / (bid_qty + ask_qty)
        except Exception as e:
            logger.debug(f"Failed to fetch live pool ratio: {e}")
        return None

    def submit_order(self, direction: str, proba: float, entry_price: float, timestamp, market_ratio_up: float = None):
        """
        Simulasi masuk posisi (UP / DOWN) di awal candle baru.
        entry_price adalah harga close dari candle 5m yang baru ditutup.
        """
        if self._position is not None:
            logger.debug("Sudah ada posisi simulasi aktif, lewati sinyal baru.")
            return None

        # Ambil parameter dari config
        pm_cfg = self.config["trading"].get("predict_market", {})
        bet_size = pm_cfg.get("bet_size_usd", 1.0)
        pool_source = pm_cfg.get("pool_ratio_source", "model_proba")
        
        # Tentukan market ratio up secara dinamis dari API riil jika memungkinkan
        if market_ratio_up is None:
            if pool_source == "fixed":
                market_ratio_up = pm_cfg.get("fixed_ratio_up", 0.50)
            else:
                # Mengambil rasio pool real-time langsung dari Binance Predict API
                market_ratio_up = self._fetch_live_pool_ratio()
                if market_ratio_up is None:
                    # Fallback jika API error/tidak tersedia
                    market_ratio_up = pm_cfg.get("fixed_ratio_up", 0.50)
                    logger.debug(f"Using fallback fixed ratio: {market_ratio_up}")
                else:
                    logger.info(f"Successfully fetched real-time pool ratio from Binance: {market_ratio_up:.2f}")
                
        # Hitung entry_cost (harga token) per $1 bet
        if direction == "UP":
            entry_cost = market_ratio_up
        else:
            entry_cost = 1.0 - market_ratio_up
            
        # Batasi entry cost agar tidak 0 atau >= 1.0 (karena minimal di Binance 0.01 dan max 0.99)
        entry_cost = max(0.01, min(0.99, entry_cost))
        
        self._position = {
            "entry_time": timestamp,
            "entry_price": entry_price,
            "direction": direction,
            "predicted_proba": proba,
            "market_ratio_up": market_ratio_up,
            "entry_cost": entry_cost,
            "bet_size": bet_size,
            "candles_held": 0
        }
        
        logger.info(
            f"🚀 [SIMULATION ENTRY] Prediksi {direction} @ {entry_price:.2f} USDT "
            f"(Token Price: {entry_cost:.2f} USDT, Bet: ${bet_size:.2f})"
        )
        return self._position

    def process_candle(self, candle_close_data: dict):
        """
        Dipanggil setiap kali candle 5m ditutup untuk memperbarui status posisi simulasi aktif.
        """
        if self._position is None:
            return None

        pos = self._position
        close = float(candle_close_data["close"])
        open_val = float(candle_close_data["open"])
        close_time = candle_close_data["close_time"]
        
        # Tentukan apakah tebakan benar
        if pos["direction"] == "UP":
            win = close > open_val
        else: # DOWN
            win = close < open_val
            
        draw = close == open_val
        
        # Payout pool-ratio (Parimutuel Betting Model):
        # Jika menang -> payout = bet_size * Multiplier. Profit = payout - bet_size - fee
        # Jika kalah -> P&L = -bet_size (rugi 100%)
        pm_cfg = self.config["trading"].get("predict_market", {})
        fee_pct = pm_cfg.get("platform_fee_pct", 0.01)
        fee = pos["bet_size"] * fee_pct
        
        # Hitung multiplier dinamis berdasarkan odds pool entry_cost (harga token)
        # Multiplier = (Total Pool * (1 - fee_pct)) / Pool Sisi Terpilih
        # Karena entry_cost merepresentasikan proporsi pool sisi terpilih (mis. 0.50 atau 0.40):
        # Multiplier = (1.0 - fee_pct) / entry_cost
        entry_cost = pos["entry_cost"]
        multiplier = (1.0 - fee_pct) / entry_cost
        
        if draw:
            exit_reason = "DRAW"
            net_pnl = 0.0
        elif win:
            exit_reason = "WIN"
            # Profit bersih = taruhan * (multiplier - 1.0)
            net_pnl = pos["bet_size"] * (multiplier - 1.0)
        else:
            exit_reason = "LOSS"
            # Kalah -> Kehilangan 100% taruhan
            net_pnl = -pos["bet_size"]
            
        return self._close_position(close, exit_reason, net_pnl, close_time)

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
            "quantity": pos["bet_size"],
            "direction": pos["direction"].lower(),
            "gross_pnl": net_pnl, # Simplified gross = net
            "net_pnl": net_pnl,
            "fee_paid": pos["bet_size"] * self.config["trading"].get("predict_market", {}).get("platform_fee_pct", 0.01),
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
            
        # Simpan state lama untuk dikembalikan ke pemanggil
        closed_position_info = {
            "pos": pos,
            "outcome": outcome,
            "net_pnl": net_pnl,
            "new_balance": new_balance,
            "exit_price": exit_price
        }
        
        # Reset posisi
        self._position = None
        return closed_position_info
