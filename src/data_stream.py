"""
data_stream.py — WebSocket Live Stream Manager

Kelas utama:
- StorageManager: Handle I/O ke parquet + SQLite
- StreamManager: Manage semua WebSocket stream (kline, depth, aggTrade)
  - _handle_kline(): Callback untuk kline closed event
  - _handle_depth(): Callback untuk order book depth updates
  - _handle_trade(): Callback untuk aggTrade events
"""

import os
import time
import threading
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from unicorn_binance_websocket_api import BinanceWebSocketApiManager

class StorageManager:
    """
    Handle I/O ke storage (parquet files).
    """
    def __init__(self, raw_path: str = "data/raw", compression: str = "snappy"):
        self.raw_path = raw_path
        self.compression = compression
        
        # Buat folder-folder yang diperlukan
        Path(self.raw_path).mkdir(parents=True, exist_ok=True)
        (Path(self.raw_path) / "klines").mkdir(parents=True, exist_ok=True)
        (Path(self.raw_path) / "orderbook").mkdir(parents=True, exist_ok=True)
        (Path(self.raw_path) / "trades").mkdir(parents=True, exist_ok=True)

    def append_klines(self, df: pd.DataFrame, timeframe: str):
        """
        Append kline DataFrame ke parquet file per timeframe.
        """
        if df.empty:
            return
        path = Path(self.raw_path) / "klines" / timeframe / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_df.sort_index(inplace=True)
                combined_df.to_parquet(path, compression=self.compression)
            except Exception as e:
                logger.error(f"Error appending klines for {timeframe}: {e}")
                df.to_parquet(path, compression=self.compression)
        else:
            df.to_parquet(path, compression=self.compression)

    def append_orderbook(self, snapshot: dict):
        """
        Append order book snapshot ke parquet file harian.
        """
        # snapshot: {"timestamp": int (ms), "bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        # Kita ratakan menjadi 1 baris DataFrame
        timestamp_dt = pd.to_datetime(snapshot["timestamp"], unit="ms", utc=True)
        row_data = {"timestamp": timestamp_dt}
        
        # Isi level 1-20
        for idx in range(20):
            col_idx = idx + 1
            bid_p = snapshot["bids"][idx][0] if idx < len(snapshot["bids"]) else np.nan
            bid_q = snapshot["bids"][idx][1] if idx < len(snapshot["bids"]) else np.nan
            ask_p = snapshot["asks"][idx][0] if idx < len(snapshot["asks"]) else np.nan
            ask_q = snapshot["asks"][idx][1] if idx < len(snapshot["asks"]) else np.nan
            
            row_data[f"bid_price_{col_idx}"] = float(bid_p)
            row_data[f"bid_qty_{col_idx}"] = float(bid_q)
            row_data[f"ask_price_{col_idx}"] = float(ask_p)
            row_data[f"ask_qty_{col_idx}"] = float(ask_q)
            
        df = pd.DataFrame([row_data]).set_index("timestamp")
        
        date_str = timestamp_dt.strftime("%Y-%m-%d")
        path = Path(self.raw_path) / "orderbook" / f"orderbook_{date_str}.parquet"
        
        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_df.sort_index(inplace=True)
                combined_df.to_parquet(path, compression=self.compression)
            except Exception as e:
                logger.error(f"Error appending orderbook snapshot: {e}")
                df.to_parquet(path, compression=self.compression)
        else:
            df.to_parquet(path, compression=self.compression)

    def append_trades(self, df: pd.DataFrame):
        """
        Append aggTrade DataFrame ke parquet file harian.
        """
        if df.empty:
            return
        
        date_str = df.index[0].strftime("%Y-%m-%d")
        path = Path(self.raw_path) / "trades" / f"trades_{date_str}.parquet"
        
        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                combined_df = pd.concat([existing_df, df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_df.sort_index(inplace=True)
                combined_df.to_parquet(path, compression=self.compression)
            except Exception as e:
                logger.error(f"Error appending trades: {e}")
                df.to_parquet(path, compression=self.compression)
        else:
            df.to_parquet(path, compression=self.compression)

    def load_klines(self, timeframe: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """
        Load klines dari storage dengan rentang waktu opsional.
        """
        path = Path(self.raw_path) / "klines" / timeframe / "data.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if start:
            start_tz = pd.to_datetime(start).tz_localize("UTC") if pd.to_datetime(start).tzinfo is None else pd.to_datetime(start)
            df = df[df.index >= start_tz]
        if end:
            end_tz = pd.to_datetime(end).tz_localize("UTC") if pd.to_datetime(end).tzinfo is None else pd.to_datetime(end)
            df = df[df.index <= end_tz]
        return df

    def get_latest_klines(self, timeframe: str, n: int = 100) -> pd.DataFrame:
        """
        Ambil N klines terakhir dari storage.
        """
        path = Path(self.raw_path) / "klines" / timeframe / "data.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        return df.tail(n)


class StreamManager:
    """
    Manager untuk semua WebSocket stream Binance.
    """
    def __init__(self, symbol: str, config: dict, storage: StorageManager = None, on_candle_closed=None):
        self.symbol = symbol.lower()
        self.config = config
        self.storage = storage
        self.on_candle_closed = on_candle_closed
        
        testnet = config["binance"].get("testnet", True)
        exchange = "binance.com-testnet" if testnet else "binance.com"
        
        # Override websocket_base_uri untuk memperbaiki bug hardcode testnet URL pada library
        ws_uri = "wss://stream.testnet.binance.vision/" if testnet else None
        
        self.ubwa = BinanceWebSocketApiManager(exchange=exchange, websocket_base_uri=ws_uri)
        self._running = False
        self._kline_buffer = []
        self._trade_buffer = []
        
        # Local orderbook cache
        self._local_orderbook = {"bids": {}, "asks": {}}
        self._last_depth_snapshot_time = 0.0
        self._last_depth_update_time = 0.0

    def start(self):
        """
        Mulai websocket streams di thread terpisah.
        """
        self._running = True
        
        # Subscribe Kline streams
        timeframes = [self.config["data"]["timeframes"]["primary"]] + self.config["data"]["timeframes"]["context"]
        kline_channels = [f"kline_{tf}" for tf in timeframes]
        
        self.ubwa.create_stream(
            channels=kline_channels,
            markets=[self.symbol],
            stream_label="klines"
        )
        
        # Subscribe Depth streams
        self.ubwa.create_stream(
            channels=["depth@100ms"],
            markets=[self.symbol],
            stream_label="depth"
        )
        
        # Subscribe aggTrade streams
        self.ubwa.create_stream(
            channels=["aggTrade"],
            markets=[self.symbol],
            stream_label="trades"
        )
        
        # Thread untuk memproses pesan masuk
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info(f"StreamManager started for {self.symbol} on {self.ubwa.exchange}")

    def _process_loop(self):
        while self._running:
            msg = self.ubwa.pop_stream_data_from_stream_buffer()
            if msg:
                self._dispatch(msg)
            else:
                time.sleep(0.001)

    def _dispatch(self, msg):
        if isinstance(msg, str):
            import json
            try:
                msg = json.loads(msg)
            except Exception as e:
                logger.error(f"Failed to parse websocket message: {e}")
                return
                
        # Unwrap data jika dibungkus oleh unicorn manager
        data = msg.get("data") if isinstance(msg, dict) and "data" in msg else msg
        if not isinstance(data, dict):
            return
            
        event_type = data.get("e", "")
        if event_type == "kline":
            self._handle_kline(data)
        elif event_type == "depthUpdate":
            self._handle_depth(data)
        elif event_type == "aggTrade":
            self._handle_trade(data)

    def _handle_kline(self, data: dict):
        k = data.get("k", {})
        if not k:
            return
            
        timeframe = k.get("i", "5m")
        is_closed = k.get("x", False)
        
        row = {
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
        
        df_row = pd.DataFrame([row]).set_index("open_time")
        
        if self.storage:
            self.storage.append_klines(df_row, timeframe)
            
        if is_closed and self.on_candle_closed:
            self.on_candle_closed(data, timeframe)

    def _handle_depth(self, data: dict):
        # Inisialisasi local orderbook jika kosong
        if not self._local_orderbook["bids"] and not self._local_orderbook["asks"]:
            try:
                from src.data_fetch import get_binance_client, fetch_depth_snapshot
                testnet = self.config["binance"].get("testnet", True)
                client = get_binance_client(testnet=testnet)
                snapshot = fetch_depth_snapshot(client, self.symbol.upper(), limit=100)
                self._local_orderbook["bids"] = {float(p): float(q) for p, q in snapshot["bids"]}
                self._local_orderbook["asks"] = {float(p): float(q) for p, q in snapshot["asks"]}
                self._last_depth_update_time = time.time()
                logger.info("Initialized local orderbook cache from REST snapshot")
            except Exception as e:
                logger.error(f"Failed to bootstrap orderbook snapshot: {e}")
                return
                
        # Terapkan diff update
        for p_str, q_str in data.get("b", []):
            p, q = float(p_str), float(q_str)
            if q == 0.0:
                self._local_orderbook["bids"].pop(p, None)
            else:
                self._local_orderbook["bids"][p] = q
                
        for p_str, q_str in data.get("a", []):
            p, q = float(p_str), float(q_str)
            if q == 0.0:
                self._local_orderbook["asks"].pop(p, None)
            else:
                self._local_orderbook["asks"][p] = q
                
        # Simpan snapshot berkala sesuai interval
        now = time.time()
        interval = self.config["data"]["orderbook"].get("snapshot_interval_sec", 30)
        if now - self._last_depth_snapshot_time >= interval:
            self._save_depth_snapshot()
            self._last_depth_snapshot_time = now

    def _save_depth_snapshot(self):
        sorted_bids = sorted(self._local_orderbook["bids"].items(), key=lambda x: x[0], reverse=True)[:20]
        sorted_asks = sorted(self._local_orderbook["asks"].items(), key=lambda x: x[0])[:20]
        
        snapshot = {
            "timestamp": int(time.time() * 1000),
            "bids": sorted_bids,
            "asks": sorted_asks
        }
        
        if self.storage:
            self.storage.append_orderbook(snapshot)

    def _handle_trade(self, data: dict):
        row = {
            "trade_time": pd.to_datetime(data["T"], unit="ms", utc=True),
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "is_buyer_maker": bool(data["m"])
        }
        df_row = pd.DataFrame([row]).set_index("trade_time")
        
        self._trade_buffer.append(df_row)
        
        # Flush buffer ke parquet jika sudah mencapai 100 trades
        if len(self._trade_buffer) >= 100:
            self._flush_trade_buffer()

    def _flush_trade_buffer(self):
        if not self._trade_buffer:
            return
        df = pd.concat(self._trade_buffer)
        df.sort_index(inplace=True)
        if self.storage:
            self.storage.append_trades(df)
        self._trade_buffer = []

    def stop(self):
        """
        Gracefully stop WebSocket streams.
        """
        self._running = False
        self._flush_trade_buffer()
        self.ubwa.stop_manager_with_all_streams()
        logger.info("StreamManager stopped")
