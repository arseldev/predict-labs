# Phase 6 — Paper Trading (Testnet)

> **Tujuan:** Jalankan sistem end-to-end secara live di Binance Testnet (uang virtual). Ini adalah validasi final sebelum modal riil — karena live trading punya banyak kejutan yang tidak ada di backtest.

---

## 6.1 Kenapa Paper Trading Itu Penting (Bukan Opsional)

Backtest, betapapun bagusnya, **tidak bisa memodelkan**:
- **Latency eksekusi**: prediksi dibuat jam 12:00:00.000 tapi order sampai ke exchange jam 12:00:00.150 — harga sudah bergerak
- **Partial fills**: order mungkin tidak terisi sepenuhnya
- **API errors**: koneksi putus, timeout, order rejection
- **Market impact**: kalau posisi cukup besar, order sendiri bisa mempengaruhi harga
- **Bugs di kode produksi**: bug yang tidak muncul di "dry run" sering muncul saat jam-jam sibuk

Paper trading di testnet adalah **defense terakhir** sebelum uang riil.

---

## 6.2 Arsitektur Sistem Live

```
┌─────────────────────────────────────────────────────────────┐
│                     live_predict.py                         │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ StreamManager│───▶│FeatureEngine │───▶│   Model      │  │
│  │(data_stream) │    │(features.py) │    │ predict_proba│  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                 │           │
│                                          ┌──────▼───────┐  │
│                                          │SignalFilter  │  │
│                                          │(threshold)   │  │
│                                          └──────┬───────┘  │
│                                                 │           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐  │
│  │  PredLog     │◀───│  Executor    │◀───│RiskManager   │  │
│  │ (predictions │    │(executor.py) │    │(kill-switch) │  │
│  │   .db)       │    └──────────────┘    └──────────────┘  │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 6.3 `src/live_predict.py` — Real-time Prediction Loop

```python
"""
live_predict.py — Prediction loop untuk paper trading dan live trading.

Flow:
1. Inisialisasi: load config, load model, connect ke WebSocket
2. Stream loop: handle setiap candle yang closed
3. Setiap candle closed:
   a. Update rolling DataFrame dengan data terbaru
   b. Hitung fitur terbaru (hanya dari data closed!)
   c. predict_proba → cek threshold
   d. Jika signal: kirim ke Executor
   e. Log prediksi ke database
4. Executor: kirim order ke testnet API
5. Monitor: cek status order, update posisi, cek kill-switch
"""

import threading
import queue
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from loguru import logger

from src.data_stream import StreamManager
from src.features import build_all_features
from src.executor import OrderExecutor
from src.models import load_model


class LivePredictor:
    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.model = load_model(model_path)
        self.feature_cols = config["features"]["feature_columns"]
        self.threshold = config["trading"]["probability_threshold"]
        
        # Rolling data storage (in-memory)
        self._kline_buffer: dict[str, pd.DataFrame] = {}  # per timeframe
        self._orderbook_buffer: pd.DataFrame = pd.DataFrame()
        self._trades_buffer: pd.DataFrame = pd.DataFrame()
        self._lookback = config["features"]["lookback_candles"]
        
        # Signal queue (thread-safe)
        self._signal_queue: queue.Queue = queue.Queue()
        
        # Stream manager
        self.stream_manager = StreamManager(
            symbol=config["binance"]["symbol"],
            config=config,
            storage=None,  # Untuk live predict, storage opsional
            on_candle_closed=self._on_candle_closed
        )
        
        # Executor
        self.executor = OrderExecutor(config)
        
        # Prediction logger
        self._pred_logger = PredictionLogger(config["logging"]["db_path"])
        
        self._running = False
    
    def start(self):
        """Start semua komponen sistem."""
        logger.info("🚀 Starting LivePredictor...")
        
        # Load historical data untuk warm-up
        self._warmup()
        
        self._running = True
        self.stream_manager.start()
        
        # Execution thread terpisah
        self._exec_thread = threading.Thread(
            target=self._execution_loop, daemon=True
        )
        self._exec_thread.start()
        
        logger.info("✅ LivePredictor started. Waiting for candle close events...")
    
    def _warmup(self):
        """
        Load data historis untuk mengisi rolling buffer sebelum stream dimulai.
        
        Tanpa warmup, fitur yang butuh lookback 100 candle tidak bisa dihitung
        untuk beberapa jam pertama setelah start.
        """
        from src.data_fetch import fetch_klines_rest, get_binance_client
        
        client = get_binance_client(testnet=self.config["binance"]["testnet"])
        lookback_str = f"{self._lookback + 50} candles ago UTC"  # buffer extra
        
        for tf in [self.config["data"]["timeframes"]["primary"]] + \
                  self.config["data"]["timeframes"]["context"]:
            df = fetch_klines_rest(client, self.config["binance"]["symbol"], tf,
                                   start_str="2 days ago UTC")
            self._kline_buffer[tf] = df
            logger.info(f"Warmed up {tf}: {len(df)} candles")
    
    def _on_candle_closed(self, candle_data: dict, timeframe: str):
        """
        Callback dipanggil setiap candle closed oleh StreamManager.
        
        ⚠️ PENTING: Fungsi ini berjalan di thread WebSocket.
        Lakukan minimal processing di sini — hanya update buffer dan queue sinyal.
        Semua heavy computation di thread terpisah.
        """
        # Append candle baru ke buffer
        new_row = _parse_candle(candle_data)
        self._kline_buffer[timeframe] = pd.concat([
            self._kline_buffer.get(timeframe, pd.DataFrame()),
            pd.DataFrame([new_row])
        ]).tail(self._lookback + 100)  # Keep last N candles
        
        # Hanya generate sinyal untuk timeframe primary (5m)
        if timeframe == self.config["data"]["timeframes"]["primary"]:
            self._signal_queue.put(("predict", pd.Timestamp.now()))
    
    def _execution_loop(self):
        """
        Loop di thread terpisah untuk processing signal dan eksekusi order.
        Dipisah dari WebSocket thread untuk menghindari blocking.
        """
        while self._running:
            try:
                msg = self._signal_queue.get(timeout=1.0)
                if msg[0] == "predict":
                    self._process_signal()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
    
    def _process_signal(self):
        """
        Hitung fitur terbaru dan generate sinyal trading.
        
        PENTING — Urutan yang benar:
        1. Ambil data kline yang SUDAH CLOSED (TIDAK pakai candle yang masih open)
        2. Hitung fitur dari data tersebut
        3. predict_proba
        4. Cek threshold
        5. Log ke database (termasuk kalau tidak ada sinyal!)
        """
        try:
            df_5m = self._kline_buffer.get("5m", pd.DataFrame())
            df_1h = self._kline_buffer.get("1h", pd.DataFrame())
            
            if len(df_5m) < 50:  # Butuh minimal 50 candle untuk fitur
                logger.warning("Insufficient warmup data, skipping prediction")
                return
            
            # Build features — HANYA dari candle yang sudah closed
            df_features = build_all_features(
                df_5m=df_5m,
                df_1h=df_1h,
                orderbook_df=self._orderbook_buffer if not self._orderbook_buffer.empty else None,
                trades_df=self._trades_buffer if not self._trades_buffer.empty else None,
                config=self.config.get("features", {})
            )
            
            if len(df_features) == 0:
                logger.warning("No features generated")
                return
            
            # Predict pada baris terakhir (candle 5m yang baru closed)
            latest_features = df_features[self.feature_cols].iloc[-1:].copy()
            proba = self.model.predict_proba(latest_features)[0, 1]
            
            timestamp = df_features.index[-1]
            
            # Log prediksi (selalu, bahkan tanpa sinyal)
            self._pred_logger.log(
                timestamp=timestamp,
                proba_up=proba,
                signal=proba > self.threshold,
                features=latest_features.to_dict()
            )
            
            logger.debug(f"[{timestamp}] P(up)={proba:.4f} | "
                        f"{'SIGNAL' if proba > self.threshold else 'skip'}")
            
            # Send to executor if signal
            if proba > self.threshold:
                self.executor.submit_order(
                    direction="long",
                    proba=proba,
                    timestamp=timestamp
                )
        
        except Exception as e:
            logger.error(f"Error in _process_signal: {e}", exc_info=True)
    
    def stop(self):
        self._running = False
        self.stream_manager.stop()
        self.executor.close_all_positions()
        logger.info("LivePredictor stopped")
```

---

## 6.4 `src/executor.py` — Order Execution

```python
"""
executor.py — Order execution untuk testnet dan live trading.

PENTING: Mode testnet WAJIB aktif untuk semua testing.
Hanya ubah ke mode live setelah lolos kriteria Phase 6 sepenuhnya.
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
import time

class OrderExecutor:
    def __init__(self, config: dict):
        self.config = config
        self.testnet = config["binance"]["testnet"]
        self.symbol = config["binance"]["symbol"]
        
        # Inisialisasi client
        if self.testnet:
            self.client = Client(
                api_key=config["binance"]["testnet_api_key"],
                api_secret=config["binance"]["testnet_api_secret"],
                testnet=True
            )
            logger.info("🧪 Executor running in TESTNET mode")
        else:
            self.client = Client(
                api_key=config["binance"]["api_key"],
                api_secret=config["binance"]["api_secret"]
            )
            logger.warning("⚠️ Executor running in LIVE mode — real money!")
        
        self._active_order = None
        self._position = None
        
        # Risk parameters
        self.max_daily_loss_usdt = config["trading"].get("max_daily_loss_usdt", 50.0)
        self.daily_loss_today = 0.0
        self.today = None
    
    def submit_order(self, direction: str, proba: float, timestamp):
        """
        Submit order ke Binance (testnet atau live).
        
        Args:
            direction: 'long' atau 'short'
            proba: probabilitas dari model (untuk logging)
            timestamp: waktu sinyal di-generate
        
        Flow:
        1. Kill-switch check
        2. Cek tidak ada posisi aktif
        3. Hitung position size
        4. Submit market order (atau limit order)
        5. Set take-profit dan stop-loss order
        6. Log order
        """
        # Kill-switch check
        if not self._check_kill_switch():
            logger.warning("Kill-switch active, order rejected")
            return
        
        # Cek posisi aktif
        if self._position is not None:
            logger.debug("Already in position, skipping new order")
            return
        
        try:
            # Hitung position size
            balance = self._get_usdt_balance()
            position_size_usdt = balance * self.config["trading"]["position_size_pct"]
            current_price = self._get_current_price()
            quantity = round(position_size_usdt / current_price, 5)  # round ke 5 desimal
            
            logger.info(f"Submitting {direction.upper()} order: "
                       f"{quantity} BTC @ ~{current_price:.2f} USDT "
                       f"(P={proba:.3f})")
            
            # Market order (cepat tapi ada slippage)
            order = self.client.order_market_buy(
                symbol=self.symbol,
                quantity=quantity
            )
            
            logger.info(f"Order filled: {order}")
            
            # Set TP/SL sebagai OCO order
            self._set_tp_sl_order(
                entry_price=float(order["fills"][0]["price"]),
                quantity=quantity
            )
            
            self._position = {
                "order_id": order["orderId"],
                "entry_time": timestamp,
                "entry_price": float(order["fills"][0]["price"]),
                "quantity": quantity,
                "predicted_proba": proba
            }
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=True)
    
    def _set_tp_sl_order(self, entry_price: float, quantity: float):
        """
        Set Take-Profit dan Stop-Loss menggunakan OCO (One-Cancels-the-Other) order.
        
        OCO: jika salah satu order tereksekusi, yang lain otomatis dibatalkan.
        Ini adalah cara yang benar untuk mengelola risiko di Binance.
        """
        tp_price = round(entry_price * (1 + self.config["trading"]["profit_target_pct"]), 2)
        sl_price = round(entry_price * (1 - self.config["trading"]["stop_loss_pct"]), 2)
        sl_limit_price = round(sl_price * 0.999, 2)  # Sedikit di bawah SL trigger
        
        try:
            oco_order = self.client.order_oco_sell(
                symbol=self.symbol,
                quantity=quantity,
                price=tp_price,              # Limit sell di TP
                stopPrice=sl_price,          # Stop trigger
                stopLimitPrice=sl_limit_price,  # Stop limit price
                stopLimitTimeInForce="GTC"
            )
            logger.info(f"OCO order set: TP={tp_price}, SL={sl_price}")
        except Exception as e:
            logger.error(f"Failed to set OCO order: {e}")
            # Fallback: set manual stop loss
    
    def _check_kill_switch(self) -> bool:
        """
        Cek kill-switch kondisi.
        Return False jika trading harus dihentikan.
        
        Kondisi stop:
        1. Daily loss melebihi max_daily_loss_usdt
        2. API error berulang
        3. Drawdown melebihi batas
        """
        from datetime import date
        today = date.today()
        
        if self.today != today:
            self.today = today
            self.daily_loss_today = 0.0  # Reset daily loss counter
        
        if self.daily_loss_today < -self.max_daily_loss_usdt:
            logger.warning(f"Daily loss limit hit: {self.daily_loss_today:.2f} USDT")
            return False
        
        return True
    
    def close_all_positions(self):
        """Force-close semua posisi (untuk shutdown atau kill-switch)."""
        if self._position:
            try:
                # Cancel semua open orders dulu
                self.client.cancel_open_orders(symbol=self.symbol)
                # Market sell
                self.client.order_market_sell(
                    symbol=self.symbol,
                    quantity=self._position["quantity"]
                )
                logger.info("All positions closed")
                self._position = None
            except Exception as e:
                logger.error(f"Error closing positions: {e}")
```

---

## 6.5 Prediction Logger (Database)

```python
"""
Menyimpan setiap prediksi ke SQLite untuk analisis performa live vs backtest.
Ini adalah sumber kebenaran untuk mengevaluasi apakah sistem benar-benar bekerja.
"""

import sqlite3
import json
from datetime import datetime

class PredictionLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Buat tabel jika belum ada."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                proba_up REAL NOT NULL,
                signal INTEGER NOT NULL,      -- 1 jika signal, 0 jika tidak
                features_json TEXT,           -- Nilai fitur saat prediksi
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_timestamp TEXT,        -- Waktu sinyal di-generate
                entry_time TEXT,
                exit_time TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                direction TEXT,
                gross_pnl REAL,
                net_pnl REAL,
                fee_paid REAL,
                exit_reason TEXT,
                predicted_proba REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def log(self, timestamp, proba_up: float, signal: bool, features: dict = None):
        """Log satu prediksi."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO predictions (timestamp, proba_up, signal, features_json) "
            "VALUES (?, ?, ?, ?)",
            (str(timestamp), proba_up, int(signal),
             json.dumps(features) if features else None)
        )
        conn.commit()
        conn.close()
    
    def log_trade(self, trade_data: dict):
        """Log hasil trade."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO trades 
            (signal_timestamp, entry_time, exit_time, entry_price, exit_price,
             quantity, direction, gross_pnl, net_pnl, fee_paid, exit_reason, predicted_proba)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(trade_data.values()))
        conn.commit()
        conn.close()
    
    def get_live_performance(self, days_back: int = 30) -> dict:
        """
        Ambil statistik performa live untuk perbandingan dengan backtest.
        
        Returns dict dengan: win_rate, avg_pnl, total_trades, daily_pnl_series
        """
        conn = sqlite3.connect(self.db_path)
        trades = pd.read_sql(
            f"SELECT * FROM trades WHERE entry_time >= datetime('now', '-{days_back} days')",
            conn
        )
        conn.close()
        
        if trades.empty:
            return {"message": "No trades yet"}
        
        return {
            "total_trades": len(trades),
            "win_rate": (trades["net_pnl"] > 0).mean(),
            "avg_net_pnl": trades["net_pnl"].mean(),
            "total_net_pnl": trades["net_pnl"].sum(),
            "total_fee_paid": trades["fee_paid"].sum(),
        }
```

---

## 6.6 Monitoring & Alerting

```python
"""
Monitoring harian untuk paper trading.
Jalankan ini sebagai cron job atau di thread terpisah.
"""

def daily_monitoring_report(pred_logger: PredictionLogger, config: dict) -> str:
    """
    Generate laporan harian untuk monitoring.
    Kirim ke Telegram/Email/Slack untuk alerting.
    
    Isi laporan:
    - Total prediksi hari ini
    - Total sinyal
    - Win rate trades hari ini
    - PnL hari ini (vs backtest expectation)
    - Status kill-switch
    - Koneksi WebSocket OK/NOK
    
    Red flags yang butuh perhatian segera:
    - Win rate < 40% dalam 24 jam (lebih buruk dari random)
    - System offline > 5 menit (disconnect)
    - Daily loss melebihi 50% dari batas kill-switch
    """
    perf = pred_logger.get_live_performance(days_back=1)
    
    report = f"""
📊 DAILY PAPER TRADING REPORT
Date: {pd.Timestamp.now().date()}
---
Trades today: {perf.get('total_trades', 0)}
Win rate: {perf.get('win_rate', 0):.1%}
Net PnL: ${perf.get('total_net_pnl', 0):.2f}
Fee paid: ${perf.get('total_fee_paid', 0):.2f}
---
Status: {'✅ OK' if perf.get('total_trades', 0) > 0 else '⚠️ No trades'}
    """
    
    return report
```

---

## 6.7 Retraining Schedule

Model harus diretrain secara berkala karena data finansial non-stasioner:

```python
"""
Retraining pipeline — jalankan mingguan atau saat performa turun.
"""

def scheduled_retrain(config: dict, db_path: str):
    """
    Retrain model dengan data terbaru.
    
    Trigger retraining jika:
    1. Setiap minggu (scheduled)
    2. Win rate live < (backtest win rate - 5%) selama 3 hari berturut-turut
    3. Model drift detected (input feature distribution berubah signifikan)
    
    Steps:
    1. Load data kline terbaru (termasuk data baru sejak training terakhir)
    2. Rebuild features
    3. Relabel dengan triple-barrier
    4. Run walk-forward validation
    5. Jika performa baru >= performa model lama: deploy model baru
    6. Jika tidak: pertahankan model lama, log sebagai warning
    
    PENTING: Jangan auto-deploy tanpa validasi. Selalu bandingkan
    model baru vs lama menggunakan out-of-sample period yang sama.
    """
    pass  # Implementasi detail di sini
```

---

## 6.8 Entry Point — `main.py`

```python
"""
main.py — Entry point untuk menjalankan sistem.

Usage:
    python main.py --mode paper     # Paper trading (testnet)
    python main.py --mode live      # Live trading (HATI-HATI!)
    python main.py --mode backtest  # Backtest saja
    python main.py --mode retrain   # Retrain model
"""
import argparse
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="BTC 5m Prediction System")
    parser.add_argument("--mode", choices=["paper", "live", "backtest", "retrain"],
                       required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default="models/latest.pkl")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == "live":
        # Double-confirmation untuk mode live
        confirm = input("⚠️ WARNING: LIVE MODE dengan uang riil! Ketik 'CONFIRM' untuk lanjut: ")
        if confirm != "CONFIRM":
            logger.info("Live mode cancelled")
            return
        config["binance"]["testnet"] = False
    
    if args.mode in ["paper", "live"]:
        predictor = LivePredictor(config, args.model)
        try:
            predictor.start()
            # Keep alive
            while True:
                import time
                time.sleep(60)
                # Print monitoring tiap menit
        except KeyboardInterrupt:
            predictor.stop()

if __name__ == "__main__":
    main()
```

---

## 6.9 Kriteria Lolos ke Phase 7 (Live Trading)

```
MUST PASS (paper trading minimal 4 minggu):
  ✅ Win rate paper trading >= backtest win rate - 5%
     (boleh sedikit lebih rendah karena latency, tapi tidak jauh)
  ✅ EV paper trading positif selama 4 minggu
  ✅ Tidak ada bug kritikal (order gagal, posisi stuck, dll)
  ✅ Sistem jalan 24/7 tanpa disconnect > 5 menit selama 2 minggu terakhir
  ✅ Kill-switch pernah di-test: simulasikan rugi besar dan cek apakah berhenti

COMPARISON PAPER VS BACKTEST:
  ✅ Win rate selisih <= 5%
  ✅ EV paper >= 50% dari EV backtest
  ✅ Max drawdown paper tidak jauh lebih besar dari backtest

RED FLAGS — JANGAN ke Phase 7:
  ❌ Win rate paper trading << backtest (selisih > 10%)
     → Ada overfitting atau implementasi bug di live code
  ❌ Sistem sering offline atau order gagal
  ❌ PnL paper trading konsisten negatif
  ❌ Jumlah trade paper trading << backtest
     → Threshold terlalu tinggi atau ada bug di signal generation
```

---

## 6.10 Checklist Selesai Phase 6

- [ ] `live_predict.py` diimplementasikan dengan `LivePredictor` class
- [ ] `executor.py` diimplementasikan dengan mode testnet/live
- [ ] OCO order (TP + SL) diset otomatis setiap entry
- [ ] Kill-switch (daily loss) aktif dan sudah di-test
- [ ] `PredictionLogger` menyimpan semua prediksi ke SQLite
- [ ] Sistem berjalan 24/7 di testnet selama minimal 4 minggu
- [ ] Monitoring report harian dihasilkan
- [ ] Performa paper vs backtest dibandingkan secara kuantitatif
- [ ] Retraining schedule direncanakan dan ditest sekali

**→ Lanjut ke [Phase 7 — Live Trading](./phase-7-live.md)**
