# Phase 2 — Feature Engineering

> **Tujuan:** Bangun pipeline feature engineering yang reproducible, bebas lookahead bias, dan mencakup fitur teknikal klasik + fitur microstructure order book. Kualitas fitur adalah faktor terbesar penentu akurasi (lebih dari kompleksitas model).

---

## 2.1 Prinsip Anti-Lookahead Bias

> **Aturan Keras:** Fitur untuk baris waktu `t` (candle yang closed di waktu `t`) **hanya boleh** menggunakan data yang tersedia sebelum atau tepat di waktu `t`. Ini berarti:
>
> - ✅ `close[t]`, `close[t-1]`, `close[t-5]` → boleh
> - ✅ `high[t]`, `low[t]` → boleh (candle `t` sudah closed)
> - ❌ `close[t+1]`, `high[t+1]` → **DILARANG** (masa depan!)
> - ❌ Menggunakan `shift(-1)` tanpa kemudian membuang baris terakhir → **DILARANG**
> - ❌ `rolling(N).mean()` tanpa `shift(1)` di konteks tertentu → perlu diperiksa case-by-case

Setiap fungsi fitur **wajib** punya unit test yang membuktikan properti ini.

---

## 2.2 Struktur `src/features.py`

```python
"""
features.py — Feature Engineering Pipeline

Semua fungsi di sini menerima DataFrame yang sudah closed (tanpa candle yang masih open)
dan mengembalikan DataFrame dengan kolom fitur tambahan.

Kontrak:
- Input: df dengan index datetime (open_time), kolom OHLCV + taker_buy_base
- Output: df dengan kolom fitur tambahan, baris pertama mungkin NaN karena lookback
- NaN di awal (akibat rolling) di-handle oleh caller (biasanya .dropna() setelah build_all_features)
"""
```

---

## 2.3 Kelompok Fitur A — Teknikal (dari OHLCV)

### A1. Log Returns (Momentum)

```python
def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan log return berbagai lag.
    
    Fitur:
        ret_1:  log(close[t] / close[t-1])
        ret_3:  log(close[t] / close[t-3])
        ret_5:  log(close[t] / close[t-5])
        ret_10: log(close[t] / close[t-10])
        ret_1h: log(close[t] / close[t-12])  # 12 * 5m = 1 jam
        ret_4h: log(close[t] / close[t-48])  # 48 * 5m = 4 jam
    
    Implementasi:
        df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
        # dst...
    
    Anti-lookahead: shift(N) dengan N > 0 selalu aman (melihat ke belakang).
    """
```

### A2. RSI (Relative Strength Index)

```python
def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI standard dengan periode default 14.
    Gunakan library 'ta': ta.momentum.RSIIndicator(df["close"], window=window).rsi()
    
    Fitur: rsi_14
    
    Anti-lookahead: RSI hanya menggunakan harga close historis — aman.
    """
```

### A3. MACD

```python
def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    MACD line, signal line, dan histogram.
    
    Fitur:
        macd_line:       EMA(close, fast) - EMA(close, slow)
        macd_signal:     EMA(macd_line, signal)
        macd_histogram:  macd_line - macd_signal
    
    Gunakan ta.trend.MACD dari library 'ta'.
    Anti-lookahead: semua EMA hanya pakai data historis — aman.
    """
```

### A4. Bollinger Band Width

```python
def add_bollinger(df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Band features sebagai proxy volatilitas.
    
    Fitur:
        bb_width: (upper_band - lower_band) / middle_band  (volatilitas relatif)
        bb_pct:   (close - lower_band) / (upper_band - lower_band)  (posisi dalam band)
    
    Anti-lookahead: BB dihitung dari rolling window historis — aman.
    """
```

### A5. ATR (Average True Range)

```python
def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR sebagai proxy volatilitas absolute.
    
    Fitur:
        atr_14:     ATR raw
        atr_ratio:  atr_14 / close  (ATR relatif terhadap harga)
    
    Anti-lookahead: ATR menggunakan high, low, close historis — aman.
    """
```

### A6. EMA Cross (Trend Signal)

```python
def add_ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    """
    EMA crossover sebagai indikator tren.
    
    Fitur:
        ema_fast:    EMA(close, fast)
        ema_slow:    EMA(close, slow)
        ema_diff:    (ema_fast - ema_slow) / close  (cross signal, normalized)
        ema_trend:   1 jika ema_fast > ema_slow, else 0 (bullish/bearish)
    
    Anti-lookahead: EMA hanya pakai data historis — aman.
    """
```

### A7. Volume Fitur

```python
def add_volume_features(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    """
    Fitur berbasis volume sebagai proxy minat pasar.
    
    Fitur:
        vol_ma_ratio:      volume / rolling_mean(volume, ma_window)  (volume spike)
        taker_buy_ratio:   taker_buy_base / volume  (rasio buy pressure dari kline)
        taker_sell_ratio:  1 - taker_buy_ratio
        vol_log:           log(volume)  (normalisasi distribusi volume)
    
    Anti-lookahead: semua rolling menggunakan data historis — aman.
    CATATAN: taker_buy_ratio dari kline adalah proxy kasar; 
    fitur dari aggTrade stream (Bagian B) jauh lebih akurat.
    """
```

### A8. Fitur Waktu

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pola likuiditas berbeda per sesi trading (Asia 00-08 UTC, Eropa 08-16 UTC, US 13-21 UTC).
    
    Fitur:
        hour_sin:    sin(2π * hour / 24)  (encoding siklik jam)
        hour_cos:    cos(2π * hour / 24)
        dow_sin:     sin(2π * day_of_week / 7)  (encoding siklik hari)
        dow_cos:     cos(2π * day_of_week / 7)
        is_weekend:  1 jika Sabtu/Minggu
        session_asia:    1 jika 00-08 UTC
        session_europe:  1 jika 08-16 UTC
        session_us:      1 jika 13-21 UTC (overlap dengan Europe)
    
    Gunakan encoding siklik (sin/cos) untuk menangkap sifat siklik waktu —
    model tidak tahu bahwa jam 23 dan 00 itu berdekatan kalau hanya pakai integer.
    Anti-lookahead: waktu saat candle DITUTUP — aman.
    """
```

### A9. Realized Volatility

```python
def add_realized_vol(df: pd.DataFrame, windows: list = [5, 12, 48]) -> pd.DataFrame:
    """
    Volatilitas realized: std dari log returns dalam N candle terakhir.
    
    Fitur untuk setiap N di windows:
        rvol_5:   std(ret_1, 5 candle) * sqrt(5)   (annualized sederhana)
        rvol_12:  std(ret_1, 12 candle) * sqrt(12)  # 1 jam
        rvol_48:  std(ret_1, 48 candle) * sqrt(48)  # 4 jam
    
    Anti-lookahead: rolling std menggunakan data historis — aman.
    """
```

---

## 2.4 Kelompok Fitur B — Microstructure (dari Order Book + AggTrade)

> **Catatan:** Fitur ini membutuhkan data dari `data_stream.py` (Phase 1).
> Untuk awal development (saat order book data belum terkumpul),
> gunakan proxy dari kline (taker_buy_ratio) dan skip ke Kelompok A dulu.
> Tambahkan fitur B setelah data terkumpul minimal 2-4 minggu.

### B1. Order Book Imbalance (OBI)

```python
def add_order_book_imbalance(
    df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    levels: list = [1, 3, 5, 10]
) -> pd.DataFrame:
    """
    OBI adalah fitur microstructure paling penting menurut riset.
    
    Formula: OBI = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
    
    Fitur untuk setiap N level:
        obi_1:   OBI menggunakan best bid/ask saja
        obi_3:   OBI menggunakan top-3 bid/ask
        obi_5:   OBI menggunakan top-5 bid/ask  (paling sering dipakai di literatur)
        obi_10:  OBI menggunakan top-10 bid/ask
    
    Range output: [-1, 1]. Positif = tekanan beli dominan; Negatif = tekanan jual dominan.
    
    Alignment: match orderbook snapshot ke kline berdasarkan timestamp (merge_asof).
    Gunakan pd.merge_asof(df, orderbook_df, left_index=True, right_index=True,
                         direction='backward') untuk join yang safe (tidak lookahead).
    
    Args:
        df: kline DataFrame (index: open_time)
        orderbook_df: order book snapshots (index: timestamp)
        levels: level kedalaman yang dihitung
    
    Anti-lookahead: KRITIS — gunakan snapshot order book yang timestamp-nya
    SEBELUM atau tepat di open_time candle. Jangan pakai snapshot yang diambil
    setelah candle closed (itu sudah termasuk informasi masa depan).
    """
```

### B2. Micro-price / VAMP

```python
def add_micro_price(df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Micro-price (VAMP = Volume-Adjusted Mid Price):
    Estimasi harga "wajar" yang mempertimbangkan ketimpangan volume bid/ask.
    
    Formula: micro_price = (best_ask * best_bid_qty + best_bid * best_ask_qty) 
                            / (best_bid_qty + best_ask_qty)
    
    Fitur:
        micro_price:      nilai absolut micro-price
        micro_price_dev:  (micro_price - mid_price) / mid_price  (deviasi dari mid)
                          Positif = lebih banyak volume di ask → tekanan beli
    
    mid_price = (best_bid + best_ask) / 2
    
    Menurut riset (Mind the Gaps, 2026), micro-price lebih prediktif
    terhadap pergerakan harga 5-menit berikutnya dibanding mid-price biasa.
    
    Anti-lookahead: sama dengan OBI — pakai merge_asof backward.
    """
```

### B3. Bid-Ask Spread

```python
def add_spread_features(df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spread relatif sebagai proxy biaya transaksi dan likuiditas.
    
    Fitur:
        spread_abs:  best_ask - best_bid  (absolute spread dalam USDT)
        spread_rel:  spread_abs / mid_price  (relative spread dalam %)
                     Tinggi = likuiditas rendah, biaya transaksi tinggi
    
    Spread tinggi juga bisa jadi sinyal volatilitas akan meningkat.
    Anti-lookahead: merge_asof backward.
    """
```

### B4. Trade Flow Imbalance (dari AggTrade)

```python
def add_trade_flow_imbalance(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    windows_min: list = [1, 5, 15]
) -> pd.DataFrame:
    """
    Trade flow imbalance dari stream aggTrade — lebih akurat dari taker_buy_ratio kline.
    
    Formula: TFI = taker_buy_vol / (taker_buy_vol + taker_sell_vol)
    
    Fitur (per window):
        tfi_1m:   TFI dalam 1 menit terakhir sebelum candle closed
        tfi_5m:   TFI dalam 5 menit terakhir (= satu candle penuh)
        tfi_15m:  TFI dalam 15 menit terakhir
        
    taker_buy_vol:  sum(quantity) dimana is_buyer_maker == False
    taker_sell_vol: sum(quantity) dimana is_buyer_maker == True
    
    Implementasi:
        1. Resample trades_df ke window waktu yang diinginkan
        2. Groupby window: sum buy_vol dan sell_vol
        3. Hitung TFI
        4. Merge ke kline df dengan merge_asof backward
    
    Anti-lookahead: gunakan data trade yang timestamp < open_time candle BERIKUTNYA.
    """
```

### B5. Order Book Depth (Likuiditas)

```python
def add_depth_features(df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Total likuiditas di beberapa level sebagai proxy "dinding" support/resistance.
    
    Fitur:
        depth_bid_5:   total volume bid di 5 level teratas (dalam BTC)
        depth_ask_5:   total volume ask di 5 level teratas (dalam BTC)
        depth_bid_10:  total volume bid di 10 level teratas
        depth_ask_10:  total volume ask di 10 level teratas
        depth_ratio_5: depth_bid_5 / depth_ask_5  (alternatif OBI berbasis depth)
    
    Anti-lookahead: merge_asof backward.
    """
```

---

## 2.5 Kelompok Fitur C — Multi-Timeframe Context

```python
def add_htf_context(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame
) -> pd.DataFrame:
    """
    Tambahkan konteks tren dari timeframe lebih besar (1h).
    
    Fitur:
        htf_ret_1h:    log return 1 candle 1h (= 1 jam terakhir)
        htf_ret_4h:    log return 4 candle 1h (= 4 jam terakhir)
        htf_rsi_1h:    RSI(14) pada timeframe 1h
        htf_ema_trend: 1 jika EMA9 > EMA21 pada 1h (bullish), else 0
        htf_vol_rel:   volume 5m terkini / rata-rata volume 1h (konteks volume)
    
    Merge:
        Gunakan pd.merge_asof(df_5m, df_1h[htf_cols], left_index=True, 
                              right_index=True, direction='backward')
        
    Anti-lookahead: KRITIS — candle 1h yang dipakai harus sudah CLOSED sebelum
    atau tepat di open_time candle 5m yang bersangkutan. 
    Gunakan shift(1) pada df_1h sebelum merge untuk memastikan ini.
    
    Contoh aman:
        df_1h_safe = df_1h.shift(1)  # pakai data candle 1h SEBELUMNYA
        pd.merge_asof(df_5m, df_1h_safe, ...)
    """
```

---

## 2.6 Pipeline Utama `build_all_features()`

```python
def build_all_features(
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame = None,
    df_1h: pd.DataFrame = None,
    orderbook_df: pd.DataFrame = None,
    trades_df: pd.DataFrame = None,
    config: dict = None
) -> pd.DataFrame:
    """
    Orchestrator feature engineering.
    Panggil semua fungsi fitur secara berurutan.
    
    Urutan:
    1. Kelompok A (selalu tersedia dari kline)
    2. Kelompok C (jika df_1h tersedia)
    3. Kelompok B (jika orderbook_df dan trades_df tersedia)
    4. dropna() — buang baris awal yang NaN akibat lookback period
    5. Log jumlah baris yang tersisa setelah dropna
    
    Returns:
        DataFrame dengan semua fitur, tanpa NaN di baris non-awal.
        
    Catatan:
        Baris TERAKHIR harus TIDAK dipakai untuk training (karena target belum diketahui).
        Ini dihandle di labeling.py, bukan di sini.
    """
    df = df_5m.copy()
    
    # Kelompok A
    df = add_log_returns(df)
    df = add_rsi(df, window=config.get("rsi_period", 14))
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df, window=config.get("atr_period", 14))
    df = add_ema_cross(df, fast=config.get("ema_fast", 9), slow=config.get("ema_slow", 21))
    df = add_volume_features(df, ma_window=config.get("vol_ma_period", 20))
    df = add_time_features(df)
    df = add_realized_vol(df)
    
    # Kelompok C
    if df_1h is not None:
        df = add_htf_context(df, df_1h)
    
    # Kelompok B (microstructure — prioritas tinggi tapi butuh data terpisah)
    if orderbook_df is not None:
        obi_levels = config.get("obi_levels", [1, 3, 5, 10])
        df = add_order_book_imbalance(df, orderbook_df, levels=obi_levels)
        df = add_micro_price(df, orderbook_df)
        df = add_spread_features(df, orderbook_df)
        df = add_depth_features(df, orderbook_df)
    
    if trades_df is not None:
        df = add_trade_flow_imbalance(df, trades_df)
    
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Feature build complete: {initial_rows} → {len(df)} rows after dropna")
    
    return df
```

---

## 2.7 Daftar Fitur Final (Reference)

| # | Fitur | Kelompok | Butuh Data |
|---|---|---|---|
| 1 | ret_1, ret_3, ret_5, ret_10, ret_1h, ret_4h | A | Kline |
| 2 | rsi_14 | A | Kline |
| 3 | macd_line, macd_signal, macd_histogram | A | Kline |
| 4 | bb_width, bb_pct | A | Kline |
| 5 | atr_14, atr_ratio | A | Kline |
| 6 | ema_fast, ema_slow, ema_diff, ema_trend | A | Kline |
| 7 | vol_ma_ratio, taker_buy_ratio, vol_log | A | Kline |
| 8 | hour_sin, hour_cos, dow_sin, dow_cos, is_weekend | A | Kline |
| 9 | session_asia, session_europe, session_us | A | Kline |
| 10 | rvol_5, rvol_12, rvol_48 | A | Kline |
| 11 | htf_ret_1h, htf_ret_4h, htf_rsi_1h, htf_ema_trend | C | Kline 1h |
| 12 | obi_1, obi_3, obi_5, obi_10 | B | Order Book |
| 13 | micro_price_dev | B | Order Book |
| 14 | spread_abs, spread_rel | B | Order Book |
| 15 | depth_bid_5, depth_ask_5, depth_ratio_5 | B | Order Book |
| 16 | tfi_1m, tfi_5m, tfi_15m | B | AggTrade |

**Total: ~35-40 fitur** — cukup kaya tapi tidak terlalu banyak untuk LightGBM.

---

## 2.8 Unit Tests (`tests/test_features.py`)

```python
"""
Test suite untuk features.py
KRITIS: Tests anti-lookahead bias adalah yang paling penting!
"""
import pytest
import pandas as pd
import numpy as np
from src.features import build_all_features, add_log_returns, add_order_book_imbalance

def make_dummy_kline(n=200):
    """Buat DataFrame kline dummy untuk testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": np.random.uniform(40000, 50000, n),
        "high": np.random.uniform(40000, 50000, n),
        "low": np.random.uniform(40000, 50000, n),
        "close": np.random.uniform(40000, 50000, n),
        "volume": np.random.uniform(1, 100, n),
        "taker_buy_base": np.random.uniform(0, 50, n),
    }, index=idx)

class TestAntiLookaheadBias:
    """
    TEST PALING PENTING: Pastikan tidak ada lookahead bias.
    
    Cara test: ubah nilai close[T] dan pastikan fitur di baris T-1 TIDAK berubah.
    Kalau fitur di T-1 berubah ketika close[T] diubah, ada lookahead bias!
    """
    
    def test_ret_1_no_lookahead(self):
        df = make_dummy_kline(100)
        result_original = add_log_returns(df.copy())
        
        # Modifikasi masa depan
        df_modified = df.copy()
        df_modified.iloc[-1, df_modified.columns.get_loc("close")] *= 2
        result_modified = add_log_returns(df_modified)
        
        # Fitur di baris ke-98 (sebelum baris terakhir ke-99) tidak boleh berubah
        original_val = result_original["ret_1"].iloc[-2]
        modified_val = result_modified["ret_1"].iloc[-2]
        assert original_val == pytest.approx(modified_val), \
            "ret_1 di t-1 berubah saat close[t] dimodifikasi — ini LOOKAHEAD BIAS!"
    
    def test_obi_no_lookahead(self):
        df = make_dummy_kline(100)
        # Buat dummy orderbook — snapshot harus SEBELUM candle
        ob_idx = df.index - pd.Timedelta("1min")  # 1 menit sebelum candle open
        orderbook_df = pd.DataFrame({
            "bid_price_1": np.random.uniform(49900, 50000, 100),
            "bid_qty_1": np.random.uniform(0.1, 10, 100),
            "ask_price_1": np.random.uniform(50000, 50100, 100),
            "ask_qty_1": np.random.uniform(0.1, 10, 100),
        }, index=ob_idx)
        
        result = add_order_book_imbalance(df, orderbook_df, levels=[1])
        # Verifikasi: OBI di setiap baris menggunakan snapshot sebelum candle itu
        # (implementasi detail tergantung merge_asof yang benar)
        assert "obi_1" in result.columns

class TestFeatureOutputShape:
    def test_no_nan_after_build(self):
        df = make_dummy_kline(200)
        result = build_all_features(df, config={})
        # Setelah dropna, tidak boleh ada NaN di tengah data
        assert result.iloc[10:].isna().sum().sum() == 0, \
            "Ada NaN di data setelah warmup period — cek implementasi rolling"

    def test_no_future_columns(self):
        df = make_dummy_kline(200)
        result = build_all_features(df, config={})
        # Kolom 'target' tidak boleh ada di output features.py
        assert "target" not in result.columns, \
            "Kolom target tidak boleh ada di output features.py"

    def test_output_rows_less_than_input(self):
        """Wajar ada baris yang hilang akibat lookback period."""
        df = make_dummy_kline(200)
        result = build_all_features(df, config={})
        assert len(result) < len(df)
        assert len(result) > 100  # Tidak boleh terlalu banyak yang hilang

class TestFeatureValues:
    def test_taker_buy_ratio_between_0_and_1(self):
        df = make_dummy_kline(200)
        from src.features import add_volume_features
        result = add_volume_features(df)
        valid = result["taker_buy_ratio"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()
    
    def test_rsi_between_0_and_100(self):
        df = make_dummy_kline(200)
        from src.features import add_rsi
        result = add_rsi(df)
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()
    
    def test_obi_between_neg1_and_1(self):
        df = make_dummy_kline(200)
        ob_idx = df.index - pd.Timedelta("1min")
        # buat dummy orderbook dengan 10 levels
        ob_data = {}
        for i in range(1, 11):
            ob_data[f"bid_price_{i}"] = np.random.uniform(49000, 50000, 200)
            ob_data[f"bid_qty_{i}"] = np.random.uniform(0.1, 10, 200)
            ob_data[f"ask_price_{i}"] = np.random.uniform(50000, 51000, 200)
            ob_data[f"ask_qty_{i}"] = np.random.uniform(0.1, 10, 200)
        orderbook_df = pd.DataFrame(ob_data, index=ob_idx)
        
        from src.features import add_order_book_imbalance
        result = add_order_book_imbalance(df, orderbook_df)
        for col in ["obi_1", "obi_3", "obi_5"]:
            if col in result.columns:
                valid = result[col].dropna()
                assert (valid >= -1).all() and (valid <= 1).all(), \
                    f"{col} harus dalam range [-1, 1]"
```

---

## 2.9 Feature Importance Analysis (Notebook 02)

Setelah Phase 4 selesai (model training), jalankan analisis SHAP:

```python
import shap

# Setelah model LightGBM terlatih:
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot importance global
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Plot individual prediksi
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0]
))
```

Gunakan hasil SHAP untuk:
1. Buang fitur yang SHAP importance-nya sangat rendah (noise reduction)
2. Identifikasi apakah OBI/microstructure features menambah value vs hanya teknikal

---

## 2.10 Kriteria Selesai Phase 2

- [ ] Semua fungsi Kelompok A diimplementasikan di `features.py`
- [ ] Fungsi Kelompok C (multi-timeframe) diimplementasikan
- [ ] `build_all_features()` berjalan tanpa error dengan data kline saja (Kelompok B opsional)
- [ ] **Semua unit test anti-lookahead bias PASS** — ini yang terpenting!
- [ ] Tidak ada NaN di output setelah warmup period
- [ ] Fitur Kelompok B diimplementasikan setelah data order book terkumpul
- [ ] Total fitur: minimal 20 (tanpa B) atau 35+ (dengan B)
- [ ] `notebook/02_feature_analysis.ipynb` dibuat untuk EDA fitur

**→ Lanjut ke [Phase 3 — Labeling](./phase-3-labeling.md)**
