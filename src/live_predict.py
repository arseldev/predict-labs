"""
live_predict.py — Real-time Prediction Loop untuk paper trading dan live trading.
"""

import os
import time
import threading
import queue
import pickle
import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.data_stream import StreamManager
from src.features import build_all_features
from src.executor import OrderExecutor
from src.models import load_model

def _parse_candle(candle_data: dict) -> dict:
    """Parse raw kline data dari websocket ke dict format internal."""
    k = candle_data.get("k", {})
    return {
        "open_time": pd.to_datetime(k["t"], unit="ms", utc=True),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
        "close_time": pd.to_datetime(k["T"], unit="ms", utc=True),
        "quote_volume": float(k["q"]),
        "trades": int(k["n"]),
        "taker_buy_base": float(k["V"]),
        "taker_buy_quote": float(k["Q"])
    }

class PredictionLogger:
    """Menyimpan data prediksi dan trade ke SQLite."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabel predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                proba_up REAL NOT NULL,
                signal INTEGER NOT NULL,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                outcome TEXT,
                features_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Migrasi jika kolom tidak ada
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        if "direction" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN direction TEXT")
        if "entry_price" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN entry_price REAL")
        if "exit_price" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN exit_price REAL")
        if "outcome" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN outcome TEXT")
        
        # Tabel trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_timestamp TEXT,
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
        
    def log(self, timestamp, proba_up: float, signal: bool, direction: str = "NEUTRAL", entry_price: float = None, features: dict = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, proba_up, signal, direction, entry_price, features_json) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(timestamp), 
            float(proba_up), 
            int(signal), 
            str(direction),
            float(entry_price) if entry_price else None,
            json.dumps(features) if features else None
        ))
        conn.commit()
        conn.close()
        
    def log_trade(self, trade_data: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades 
            (signal_timestamp, entry_time, exit_time, entry_price, exit_price,
             quantity, direction, gross_pnl, net_pnl, fee_paid, exit_reason, predicted_proba)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(trade_data.get("signal_timestamp", "")),
            str(trade_data.get("entry_time", "")),
            str(trade_data.get("exit_time", "")),
            float(trade_data.get("entry_price", 0.0)),
            float(trade_data.get("exit_price", 0.0)) if trade_data.get("exit_price") else None,
            float(trade_data.get("quantity", 0.0)),
            str(trade_data.get("direction", "long")),
            float(trade_data.get("gross_pnl", 0.0)),
            float(trade_data.get("net_pnl", 0.0)),
            float(trade_data.get("fee_paid", 0.0)),
            str(trade_data.get("exit_reason", "")),
            float(trade_data.get("predicted_proba", 0.0))
        ))
        conn.commit()
        conn.close()
        
    def get_live_performance(self, days_back: int = 30) -> dict:
        conn = sqlite3.connect(self.db_path)
        try:
            # Query trades ke DataFrame
            trades = pd.read_sql_query(
                f"SELECT * FROM trades WHERE entry_time >= datetime('now', '-{days_back} days')",
                conn
            )
        except Exception as e:
            logger.error(f"Error querying performance: {e}")
            trades = pd.DataFrame()
        finally:
            conn.close()
            
        if trades.empty:
            return {"total_trades": 0, "win_rate": 0.0, "total_net_pnl": 0.0, "total_fee_paid": 0.0}
            
        return {
            "total_trades": len(trades),
            "win_rate": float((trades["net_pnl"] > 0).mean()),
            "avg_net_pnl": float(trades["net_pnl"].mean()),
            "total_net_pnl": float(trades["net_pnl"].sum()),
            "total_fee_paid": float(trades["fee_paid"].sum()),
        }

class LivePredictor:
    def __init__(self, config: dict, model_path: str):
        self.config = config
        model_result = load_model(model_path)
        self.model, feature_cols_from_model = model_result

        # Prioritas pengambilan feature_cols:
        # 1. Dari pickle model (paling akurat, disimpan saat training)
        # 2. Dari config YAML runtime (jika tersedia)
        # 3. Dari model.feature_name_ (LightGBM introspection)
        # 4. Hardcoded fallback (hanya 33 fitur — SUDAH TIDAK DIGUNAKAN)
        if feature_cols_from_model:
            self.feature_cols = feature_cols_from_model
            logger.info(f"Feature cols loaded from model pickle: {len(self.feature_cols)} features")
        elif config["features"].get("feature_columns"):
            self.feature_cols = config["features"]["feature_columns"]
            logger.info(f"Feature cols loaded from config: {len(self.feature_cols)} features")
        elif hasattr(self.model, "feature_name_"):
            self.feature_cols = list(self.model.feature_name_)
            logger.warning(f"Feature cols extracted from model introspection: {len(self.feature_cols)} features")
        else:
            logger.error(
                "CRITICAL: Cannot determine feature_cols! Model was not saved with feature_cols. "
                "Please retrain and save model using updated save_model(model, path, feature_cols)."
            )
            self.feature_cols = []

        self.threshold = config["trading"]["probability_threshold"]
        
        self._kline_buffer: dict[str, pd.DataFrame] = {}
        self._orderbook_buffer: pd.DataFrame = pd.DataFrame()
        self._trades_buffer: pd.DataFrame = pd.DataFrame()
        self._lookback = config["features"]["lookback_candles"]
        
        self._signal_queue = queue.Queue()
        
        # Inisialisasi stream manager
        self.stream_manager = StreamManager(
            symbol=config["binance"]["symbol"],
            config=config,
            storage=None,  # Opsional untuk live predict
            on_candle_closed=self._on_candle_closed
        )
        
        self._pred_logger = PredictionLogger(config["logging"]["db_path"])
        self.simulate_mode = config.get("simulate_mode", False)
        
        if self.simulate_mode:
            from src.simulation_engine import SimulationEngine
            self.executor = SimulationEngine(config, pred_logger=self._pred_logger)
            logger.info("🎮 LivePredictor running in pure SIMULATION mode (No Binance API orders)")
        else:
            self.executor = OrderExecutor(config, pred_logger=self._pred_logger)
        
        # Setup Telegram logging if credentials exist in env
        bot_token = os.getenv("BOT_TOKEN")
        bot_target_id = os.getenv("BOT_TARGET_ID")
        self.telegram_handler = None
        self._pred_counter = 0
        if bot_token and bot_target_id:
            from src.telegram_utils import TelegramBufferedHandler
            self.telegram_handler = TelegramBufferedHandler(bot_token, str(bot_target_id))
            logger.add(self.telegram_handler.write, level="INFO")
            logger.info("📢 Telegram logging sink initialized successfully!")
        else:
            logger.warning("⚠️ BOT_TOKEN or BOT_TARGET_ID not found in env. Telegram logs disabled.")
            
        self._running = False
        
    def start(self):
        logger.info("Starting LivePredictor...")
        self._warmup()
        
        self._running = True
        self.stream_manager.start()
        
        self._exec_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._exec_thread.start()
        logger.info("LivePredictor successfully running. Waiting for closed candles...")
        
    def _warmup(self):
        """Warm up in-memory buffer dengan REST data."""
        from src.data_fetch import get_binance_client, fetch_klines_rest
        
        # Gunakan testnet=True agar terhubung ke testnet.binance.vision yang bebas dari SSL block di PC user
        client = get_binance_client(testnet=True)
        
        timeframes = [self.config["data"]["timeframes"]["primary"]] + self.config["data"]["timeframes"]["context"]
        
        for tf in timeframes:
            # Ambil data kline 2 days ago
            df = fetch_klines_rest(client, self.config["binance"]["symbol"], tf, start_str="2 days ago UTC")
            self._kline_buffer[tf] = df
            logger.info(f"Warmed up kline buffer for {tf}: {len(df)} candles")

    def _on_candle_closed(self, candle_data: dict, timeframe: str):
        """Callback dipanggil saat candle ditutup oleh StreamManager."""
        new_row = _parse_candle(candle_data)
        new_df = pd.DataFrame([new_row]).set_index("open_time")
        
        # Append ke buffer
        if timeframe not in self._kline_buffer:
            self._kline_buffer[timeframe] = new_df
        else:
            self._kline_buffer[timeframe] = pd.concat([self._kline_buffer[timeframe], new_df])
            # Batasi buffer
            self._kline_buffer[timeframe] = self._kline_buffer[timeframe].tail(self._lookback + 100)
            
        # Trigger prediksi jika timeframe utama (5m) closed
        if timeframe == self.config["data"]["timeframes"]["primary"]:
            # Jika dalam mode simulasi, update status trade aktif terlebih dahulu menggunakan lilin baru yang ditutup ini
            if self.simulate_mode and hasattr(self, "executor") and hasattr(self.executor, "process_candle"):
                self.executor.process_candle({
                    "open": new_row["open"],
                    "high": new_row["high"],
                    "low": new_row["low"],
                    "close": new_row["close"],
                    "close_time": new_row["close_time"]
                })
            self._signal_queue.put(("predict", pd.Timestamp.now(tz=None)))

    def _execution_loop(self):
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
        """Orchestrator real-time feature calculation dan inference."""
        try:
            df_5m = self._kline_buffer.get("5m", pd.DataFrame())
            df_1h = self._kline_buffer.get("1h", pd.DataFrame())
            
            if len(df_5m) < 50:
                logger.warning("Insufficient warmup data. Skipping signal check.")
                return
                
            # Build features HANYA dari data closed
            df_features = build_all_features(
                df_5m=df_5m,
                df_1h=df_1h,
                orderbook_df=None, # Opsional untuk awal MVP
                trades_df=None,
                config=self.config.get("features", {})
            )
            
            if df_features.empty:
                logger.warning("No features built")
                return
            
            # Validasi eksplisit: pastikan semua feature_cols tersedia sebelum predict
            if not self.feature_cols:
                logger.error("feature_cols is empty! Cannot predict. Re-train model or fix load_model.")
                return

            missing_feats = [f for f in self.feature_cols if f not in df_features.columns]
            if missing_feats:
                logger.error(
                    f"FEATURE MISMATCH: Model butuh {len(self.feature_cols)} features, "
                    f"tapi {len(missing_feats)} tidak ada di runtime data: {missing_feats}"
                )
                return
                
            # Prediksi pada lilin closed terakhir
            latest_features = df_features[self.feature_cols].iloc[-1:].copy()
            proba = self.model.predict_proba(latest_features)[0, 1]
            timestamp = df_features.index[-1]
            entry_price = float(df_5m["close"].iloc[-1]) # harga entry adalah close candle 5m yang barusan ditutup
            
            # Tentukan arah berdasarkan threshold
            threshold_up = self.threshold
            threshold_down = 1.0 - self.threshold
            
            if proba >= threshold_up:
                direction = "UP"
                signal = True
            elif proba <= threshold_down:
                direction = "DOWN"
                signal = True
            else:
                direction = "NEUTRAL"
                signal = False
            
            # Log prediksi ke SQLite
            self._pred_logger.log(
                timestamp=timestamp,
                proba_up=proba,
                signal=signal,
                direction=direction,
                entry_price=entry_price,
                features=latest_features.iloc[0].to_dict()
            )
            
            # Log ke terminal
            logger.info(f"[{timestamp}] P(up)={proba:.4f} | DIRECTION={direction} | SIGNAL={signal}")
            
            if signal:
                if self.simulate_mode:
                    self.executor.submit_order(
                        direction=direction,
                        proba=proba,
                        entry_price=entry_price,
                        timestamp=timestamp
                    )
                else:
                    # Di real order executor (order_market_buy dll) hanya long yang didukung.
                    # Kita tetap pass ke executor.
                    self.executor.submit_order(
                        direction="long" if direction == "UP" else "short",
                        proba=proba,
                        timestamp=timestamp
                    )
                
        except Exception as e:
            logger.error(f"Error processing signal: {e}", exc_info=True)
        finally:
            if self.telegram_handler:
                self._pred_counter += 1
                if self._pred_counter >= 3:
                    try:
                        # Ambil timestamp dari try block if available, else current time
                        t_str = str(pd.Timestamp.now(tz=None))
                        msg_time = locals().get("timestamp", t_str)
                        self.telegram_handler.flush_to_telegram(header=f"🤖 BTC 5m Predictions (Past 15m) @ {msg_time}")
                        self._pred_counter = 0
                    except Exception as e:
                        print(f"Failed to flush telegram handler: {e}")

    def stop(self):
        self._running = False
        self.stream_manager.stop()
        if hasattr(self.executor, "close_all_positions"):
            self.executor.close_all_positions()
        logger.info("LivePredictor gracefully stopped")
