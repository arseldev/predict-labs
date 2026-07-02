# Phase 1 — Data Pipeline

> **Tujuan:** Kumpulkan data kline multi-timeframe, order book depth, dan aggTrade dari Binance — baik historis maupun live streaming. Ini adalah fondasi seluruh sistem; kualitas data menentukan kualitas model.

---

## 1.1 Overview Arsitektur Data Pipeline

```
┌─────────────────────────────────────────────────┐
│               Data Sources (Binance)            │
│  REST API          WebSocket        Bulk CSV    │
│  (historical)      (live)           (data.binance.vision) │
└────────┬───────────────┬───────────────┬────────┘
         │               │               │
         ▼               ▼               ▼
┌─────────────────────────────────────────────────┐
│              data_fetch.py / data_stream.py     │
│  get_klines()    handle_kline()   bulk_download()│
│  get_depth()     handle_depth()                 │
│  get_trades()    handle_trade()                 │
└────────┬───────────────┬────────────────────────┘
         │               │
         ▼               ▼
┌─────────────────────────────────────────────────┐
│              Storage Layer                      │
│  data/raw/klines/*.parquet                      │
│  data/raw/orderbook/*.parquet                   │
│  data/raw/trades/*.parquet                      │
└─────────────────────────────────────────────────┘
```

---

## 1.2 `src/data_fetch.py` — Historical Data Fetcher

### Fungsi-fungsi yang harus diimplementasikan:

#### `load_config(config_path: str) -> dict`
```python
"""
Load config.yaml dan merge dengan environment variables (.env).
Gunakan python-dotenv + pyyaml.
Contoh: ${BINANCE_API_KEY} di config.yaml diganti dengan nilai dari .env
"""
```

#### `get_binance_client(testnet: bool = True) -> Client`
```python
"""
Inisialisasi Binance client.
- Jika testnet=True, gunakan testnet_url
- Ambil API key dari environment, BUKAN hardcode
- Return: instance binance.Client
"""
```

#### `fetch_klines_rest(symbol, interval, start_str, end_str=None, save_path=None) -> pd.DataFrame`
```python
"""
Ambil data kline via REST API untuk periode PENDEK (< 90 hari).

Args:
    symbol: e.g., "BTCUSDT"
    interval: e.g., "5m", "1h"
    start_str: e.g., "90 days ago UTC"
    end_str: opsional, default sampai sekarang
    save_path: jika diisi, simpan ke parquet

Returns:
    DataFrame dengan kolom:
    [open_time, open, high, low, close, volume, close_time,
     quote_volume, trades, taker_buy_base, taker_buy_quote]

PENTING: open_time harus jadi index, semua kolom numerik harus float64.
Jangan simpan kolom 'ignore' dari Binance API.
"""
```

#### `download_bulk_klines(symbol, interval, year, month, save_path) -> str`
```python
"""
Download kline historis PANJANG dari data.binance.vision (bulk CSV).
Lebih efisien daripada looping REST API untuk data > 90 hari.

URL format: https://data.binance.vision/data/spot/monthly/klines/
            {symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip

Steps:
1. Download ZIP file
2. Extract CSV
3. Parse ke DataFrame (header sama dengan fetch_klines_rest)
4. Simpan ke parquet di save_path
5. Hapus file ZIP dan CSV temporary

Returns: path file parquet yang tersimpan

Error handling:
- Jika file bulan tidak tersedia, skip dengan warning
- Jika sudah ada parquet untuk bulan tersebut, skip (idempotent)
"""
```

#### `fetch_all_historical_klines(symbol, interval, days_back, save_dir) -> pd.DataFrame`
```python
"""
Orchestrator: ambil data kline historis sesuai days_back.
- Jika days_back > 90: gunakan bulk download + REST untuk bulan terakhir
- Jika days_back <= 90: gunakan REST API langsung
- Gabungkan semua data, dedup berdasarkan open_time, sort ascending
- Simpan gabungan ke processed parquet

Returns: DataFrame lengkap
"""
```

#### `fetch_depth_snapshot(symbol, limit=20) -> dict`
```python
"""
Ambil snapshot order book saat ini via REST.
Gunakan untuk bootstrapping sebelum WebSocket jalan.

Args:
    symbol: e.g., "BTCUSDT"
    limit: jumlah level (5, 10, 20, 50, 100, 500, 1000)

Returns:
    {
      "timestamp": int (ms),
      "bids": [[price, qty], ...],  # sorted descending by price
      "asks": [[price, qty], ...]   # sorted ascending by price
    }
"""
```

### Contoh Implementasi `fetch_klines_rest`:

```python
import pandas as pd
from binance.client import Client
from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]
NUMERIC_COLS = ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]

def fetch_klines_rest(
    client: Client,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_str: str = "90 days ago UTC",
    end_str: str = None,
    save_path: str = None
) -> pd.DataFrame:
    logger.info(f"Fetching {symbol} {interval} klines from {start_str}")

    raw = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    df.drop(columns=["ignore"], inplace=True)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in NUMERIC_COLS:
        df[col] = df[col].astype("float64")
    df["trades"] = df["trades"].astype("int64")

    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, compression="snappy")
        logger.info(f"Saved {len(df)} rows to {save_path}")

    return df
```

---

## 1.3 `src/data_stream.py` — WebSocket Live Stream Manager

### Fungsi-fungsi yang harus diimplementasikan:

#### `class StreamManager`
```python
"""
Manager untuk semua WebSocket stream Binance.
Gunakan unicorn-binance-websocket-api untuk auto-reconnect.

Attributes:
    symbol: trading pair
    callbacks: dict of {stream_type: callable}
    buffer: dict of {stream_type: list} — buffer data sebelum disimpan
    storage: instance StorageManager

Methods:
    start(): mulai semua stream
    stop(): hentikan semua stream dengan graceful shutdown
    _handle_kline(msg): callback untuk kline stream
    _handle_depth(msg): callback untuk depth stream
    _handle_trade(msg): callback untuk aggTrade stream
    _flush_buffer(): simpan buffer ke storage setiap N event
"""
```

#### `_handle_kline(msg: dict) -> None`
```python
"""
Callback untuk kline WebSocket event.

Logic:
1. Parse msg["k"] untuk data candle
2. Jika msg["k"]["x"] == True (candle CLOSED):
   a. Tambahkan ke buffer kline
   b. Update in-memory rolling DataFrame (untuk prediksi live)
   c. Trigger feature calculation (Fase 2)
3. Jika candle masih OPEN: update data sementara saja (jangan dipakai sebagai fitur!)

PENTING: Candle yang belum closed (x=False) TIDAK BOLEH dipakai sebagai input fitur.
Ini adalah aturan anti-lookahead bias #1.

Struktur msg["k"]:
    t: open time (ms)
    T: close time (ms)
    s: symbol
    i: interval
    f/L: first/last trade ID
    o/h/l/c: open/high/low/close
    v: volume
    n: number of trades
    x: is_closed (bool) ← cek ini!
    q: quote asset volume
    V: taker buy base volume
    Q: taker buy quote volume
"""
```

#### `_handle_depth(msg: dict) -> None`
```python
"""
Callback untuk order book depth stream.
Binance mengirim diff update, bukan snapshot penuh setiap kali.

Logic:
1. Jika ini update pertama, fetch snapshot via REST dulu (fetch_depth_snapshot)
2. Apply diff update ke snapshot lokal:
   - Untuk setiap bid/ask update: jika qty == 0, hapus level; else update level
3. Simpan snapshot terkini ke buffer setiap snapshot_interval_sec detik
4. Buffer di-flush ke parquet secara periodic

Catatan: Order book maintenance ini cukup kompleks.
Alternatif lebih simple: snapshot REST setiap 30 detik tanpa WebSocket diff —
akurasi lebih rendah tapi lebih mudah diimplementasikan awal-awal.
"""
```

#### `_handle_trade(msg: dict) -> None`
```python
"""
Callback untuk aggTrade WebSocket event.

Data yang perlu diambil dari msg:
    T: trade time (ms)
    p: price
    q: quantity
    m: is_buyer_maker (True = SELL aggressor, False = BUY aggressor)
    
Logic:
1. Kategorisasi: jika m=False → taker_buy; jika m=True → taker_sell
2. Akumulasikan dalam window waktu (e.g., 5 menit)
3. Hitung trade_flow_imbalance = taker_buy_vol / (taker_buy_vol + taker_sell_vol)
4. Simpan ke buffer, flush setiap N menit atau N event
"""
```

#### `class StorageManager`
```python
"""
Handle I/O ke storage (parquet files + SQLite).

Methods:
    append_klines(df, timeframe): append ke parquet kline per timeframe
    append_orderbook(snapshot): append snapshot order book ke parquet
    append_trades(df): append aggTrade data ke parquet
    load_klines(timeframe, start, end): load kline dari parquet untuk range waktu
    get_latest_klines(timeframe, n): ambil N candle terakhir
"""
```

### Contoh Skeleton `StreamManager`:

```python
from unicorn_binance_websocket_api import BinanceWebSocketApiManager
from loguru import logger
import threading
import time

class StreamManager:
    def __init__(self, symbol: str, config: dict, storage: StorageManager):
        self.symbol = symbol.lower()
        self.config = config
        self.storage = storage
        self.ubwa = BinanceWebSocketApiManager(
            exchange="binance.com"  # atau "binance.com-testnet"
        )
        self._running = False
        self._kline_buffer = []
        self._depth_buffer = []
        self._trade_buffer = []
        self._local_orderbook = {"bids": {}, "asks": {}}  # price → qty

    def start(self):
        self._running = True
        # Subscribe streams
        self.ubwa.create_stream(
            channels=["kline_5m", "kline_1m", "kline_15m", "kline_1h"],
            markets=[self.symbol],
            stream_label="klines"
        )
        self.ubwa.create_stream(
            channels=["depth@100ms"],
            markets=[self.symbol],
            stream_label="depth"
        )
        self.ubwa.create_stream(
            channels=["aggTrade"],
            markets=[self.symbol],
            stream_label="trades"
        )

        # Process loop di thread terpisah
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info(f"StreamManager started for {self.symbol}")

    def _process_loop(self):
        while self._running:
            msg = self.ubwa.pop_stream_data_from_stream_buffer()
            if msg:
                self._dispatch(msg)
            else:
                time.sleep(0.001)  # Avoid busy-wait

    def _dispatch(self, msg):
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        event_type = data.get("e", "")
        if event_type == "kline":
            self._handle_kline(data)
        elif event_type == "depthUpdate":
            self._handle_depth(data)
        elif event_type == "aggTrade":
            self._handle_trade(data)

    def stop(self):
        self._running = False
        self.ubwa.stop_manager_with_all_streams()
        logger.info("StreamManager stopped")
```

---

## 1.4 Data Schema

### Kline Parquet Schema

| Kolom | Tipe | Keterangan |
|---|---|---|
| open_time (index) | datetime64[ns, UTC] | Waktu buka candle |
| open | float64 | Harga buka |
| high | float64 | Harga tertinggi |
| low | float64 | Harga terendah |
| close | float64 | Harga tutup |
| volume | float64 | Volume dalam BTC |
| close_time | datetime64[ns, UTC] | Waktu tutup candle |
| quote_volume | float64 | Volume dalam USDT |
| trades | int64 | Jumlah trade |
| taker_buy_base | float64 | Volume taker buy dalam BTC |
| taker_buy_quote | float64 | Volume taker buy dalam USDT |

### Order Book Snapshot Parquet Schema

| Kolom | Tipe | Keterangan |
|---|---|---|
| timestamp (index) | datetime64[ns, UTC] | Waktu snapshot |
| bid_price_1..20 | float64 | Harga bid level 1-20 |
| bid_qty_1..20 | float64 | Volume bid level 1-20 |
| ask_price_1..20 | float64 | Harga ask level 1-20 |
| ask_qty_1..20 | float64 | Volume ask level 1-20 |

### AggTrade Parquet Schema

| Kolom | Tipe | Keterangan |
|---|---|---|
| trade_time (index) | datetime64[ns, UTC] | Waktu trade |
| price | float64 | Harga eksekusi |
| quantity | float64 | Volume BTC |
| is_buyer_maker | bool | True = sell aggressor |

---

## 1.5 Unit Tests (`tests/test_data_fetch.py`)

```python
"""
Test suite untuk data_fetch.py dan data_stream.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.data_fetch import fetch_klines_rest, fetch_depth_snapshot

class TestFetchKlines:
    def test_returns_dataframe(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert isinstance(df, pd.DataFrame)

    def test_index_is_datetime(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_no_nan_in_ohlcv(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert df[["open","high","low","close","volume"]].isna().sum().sum() == 0

    def test_sorted_ascending(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert df.index.is_monotonic_increasing

    def test_no_duplicate_timestamps(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert df.index.is_unique

    def test_high_gte_low(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert (df["high"] >= df["low"]).all()

    def test_volume_non_negative(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert (df["volume"] >= 0).all()

class TestDepthSnapshot:
    def test_returns_bids_asks(self, mock_client):
        snapshot = fetch_depth_snapshot(mock_client, "BTCUSDT", limit=20)
        assert "bids" in snapshot
        assert "asks" in snapshot
        assert "timestamp" in snapshot

    def test_bids_sorted_desc(self, mock_client):
        snapshot = fetch_depth_snapshot(mock_client, "BTCUSDT", limit=20)
        bid_prices = [float(b[0]) for b in snapshot["bids"]]
        assert bid_prices == sorted(bid_prices, reverse=True)

    def test_asks_sorted_asc(self, mock_client):
        snapshot = fetch_depth_snapshot(mock_client, "BTCUSDT", limit=20)
        ask_prices = [float(a[0]) for a in snapshot["asks"]]
        assert ask_prices == sorted(ask_prices)
```

---

## 1.6 Catatan Penting Phase 1

### ⚠️ Order Book: Mulai Kumpulkan SEKARANG
Histori order book gratis biasanya hanya tersedia beberapa hari ke belakang.
Kline bisa di-download bulk hingga bertahun-tahun, tapi order book tidak.
**Jalankan data_stream.py sesegera mungkin dan biarkan jalan terus.**

### ⚠️ Rate Limit Binance
- REST API: 1200 request weight per menit
- Bulk download dari data.binance.vision: tidak ada rate limit (HTTP biasa)
- Untuk kline > 90 hari: WAJIB pakai bulk download, bukan looping REST

### ⚠️ Reconnect & Reliability
- `unicorn-binance-websocket-api` sudah handle reconnect otomatis
- Tambahkan monitoring: log jumlah disconnect per jam
- Jika disconnect > 5 menit, kirim alert (email/Telegram/log kritis)

---

## 1.7 Kriteria Selesai Phase 1

- [ ] `data_fetch.py` bisa download minimal 90 hari kline 5m BTCUSDT
- [ ] `data_fetch.py` bisa download kline untuk semua timeframe: 1m, 15m, 1h
- [ ] `data_stream.py` berjalan tanpa disconnect > 5 menit dalam uji 24 jam
- [ ] Data order book tersimpan dalam parquet, schema sesuai 1.4
- [ ] Data aggTrade tersimpan dalam parquet, schema sesuai 1.4
- [ ] Semua unit test di `test_data_fetch.py` pass
- [ ] Tidak ada data duplikat atau gap besar di kline data
- [ ] Parquet files bisa dibaca ulang dengan `pd.read_parquet()` tanpa error

**→ Lanjut ke [Phase 2 — Feature Engineering](./phase-2-features.md)**
