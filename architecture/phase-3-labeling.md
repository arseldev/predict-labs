# Phase 3 — Labeling

> **Tujuan:** Implementasikan dua metode labeling (fixed-horizon & triple-barrier), pahami trade-off masing-masing, dan siapkan dataset berlabel bersih untuk training model di Phase 4.

---

## 3.1 Mengapa Labeling Itu Krusial

Labeling menentukan **apa yang model coba pelajari**. Label yang salah = model belajar hal yang salah, meskipun model-nya kompleks.

Dua masalah utama dalam labeling data finansial:
1. **Lookahead bias dalam label:** label harus dibuat dari informasi yang benar-benar tersedia saat sinyal diprediksi
2. **Label overlap (concurrent labels):** untuk triple-barrier, satu candle bisa overlap secara waktu dengan banyak label — ini melanggar asumsi independensi ML standard

---

## 3.2 Metode 1 — Fixed-Horizon Labeling

### Definisi
```
Label[t] = 1  jika close[t+1] > close[t]  (harga naik pada candle berikutnya)
Label[t] = 0  jika close[t+1] <= close[t] (harga turun atau sama)
```

### Implementasi

```python
def label_fixed_horizon(df: pd.DataFrame, n_ahead: int = 1) -> pd.DataFrame:
    """
    Label sederhana berdasarkan pergerakan harga N candle ke depan.
    
    Args:
        df: DataFrame dengan kolom 'close'
        n_ahead: jumlah candle ke depan (default 1 = 5 menit berikutnya)
    
    Returns:
        df dengan kolom baru:
            'label_fh': int (0 atau 1)
            'future_ret': log return N candle ke depan (untuk analisis)
    
    PENTING: Baris N terakhir akan memiliki NaN di 'label_fh' karena
    harga N candle ke depan belum tersedia. HARUS di-drop sebelum training.
    
    Implementasi:
        df["future_close"] = df["close"].shift(-n_ahead)
        df["future_ret"] = np.log(df["future_close"] / df["close"])
        df["label_fh"] = (df["future_close"] > df["close"]).astype(int)
        # Hapus kolom helper, drop baris terakhir
        df.drop(columns=["future_close"], inplace=True)
        return df
    """
```

### Kelebihan dan Kekurangan Fixed-Horizon

| Kelebihan | Kekurangan |
|---|---|
| Simple & cepat diimplementasikan | Mengabaikan jalur harga di antaranya |
| Label bersih, tidak overlap | Tidak merepresentasikan trading nyata |
| Mudah diinterpretasikan | "Naik" bisa berarti naik lalu turun tajam |
| Cocok untuk baseline awal | Tidak mempertimbangkan risk/reward |

---

## 3.3 Metode 2 — Triple-Barrier Labeling

### Konsep (dari Marcos López de Prado)

Untuk setiap titik waktu `t`, tetapkan 3 barrier:
- **Barrier Atas (TP):** `close[t] * (1 + profit_pct)` — target profit
- **Barrier Bawah (SL):** `close[t] * (1 - loss_pct)` — stop loss
- **Barrier Vertikal:** batas waktu `t + max_candles` candle

Label ditentukan oleh barrier mana yang **tersentuh lebih dulu** oleh `high` atau `low` harga:
- **Label 1:** barrier atas tersentuh (profit target tercapai)
- **Label -1:** barrier bawah tersentuh (stop loss)
- **Label 0:** barrier vertikal tercapai (waktu habis, tidak ada sinyal kuat)

> **Catatan:** Beberapa implementasi menggunakan hanya 0/1 (binary) bukan -1/0/1.
> Untuk klasifikasi biner UP/DOWN, label 0 (timeout) bisa di-drop atau dijadikan "ambiguous signal".

### Implementasi

```python
def label_triple_barrier(
    df: pd.DataFrame,
    profit_pct: float = 0.0015,
    loss_pct: float = 0.0015,
    max_candles: int = 6,
    use_dynamic_barrier: bool = False,
    vol_col: str = "rvol_12"
) -> pd.DataFrame:
    """
    Triple-barrier labeling.
    
    Args:
        df: DataFrame dengan kolom 'open', 'high', 'low', 'close'
        profit_pct: % gain untuk trigger barrier atas (default 0.15%)
        loss_pct: % loss untuk trigger barrier bawah (default 0.15%)
        max_candles: batas waktu maksimum (default 6 = 30 menit)
        use_dynamic_barrier: jika True, barrier diset proporsional dengan realized volatility
        vol_col: kolom volatilitas untuk dynamic barrier (perlu add_realized_vol() dulu)
    
    Returns:
        df dengan kolom baru:
            'label_tb':      int (1, -1, 0) — barrier mana yang kena duluan
            'label_tb_bin':  int (1, 0) — versi binary: 1=naik, 0=turun/timeout
            'barrier_top':   float — harga barrier atas
            'barrier_bot':   float — harga barrier bawah
            'barrier_time':  datetime — waktu batas vertikal
            'hit_barrier':   str — "top" / "bottom" / "vertical" — debug info
    
    Dynamic barriers:
        Jika use_dynamic_barrier=True:
            profit_pct = df[vol_col] * multiplier  (vol_col dari add_realized_vol)
            loss_pct = profit_pct  (symmetric)
        Ini membuat barrier adaptif terhadap kondisi pasar.
    """
    results = []
    
    for i in range(len(df) - max_candles):
        entry_price = df["close"].iloc[i]
        
        if use_dynamic_barrier:
            vol = df[vol_col].iloc[i]
            profit_pct_i = vol * 1.0  # multiplier bisa diatur
            loss_pct_i = vol * 1.0
        else:
            profit_pct_i = profit_pct
            loss_pct_i = loss_pct
        
        barrier_top = entry_price * (1 + profit_pct_i)
        barrier_bot = entry_price * (1 - loss_pct_i)
        
        label = 0
        hit = "vertical"
        
        # Cek harga dalam window max_candles candle ke depan
        for j in range(1, max_candles + 1):
            if i + j >= len(df):
                break
            high_j = df["high"].iloc[i + j]
            low_j = df["low"].iloc[i + j]
            
            if high_j >= barrier_top and low_j <= barrier_bot:
                # Keduanya kena di candle yang sama — ambil yang lebih masuk akal
                # Konservatif: asumsikan SL kena duluan
                label = -1
                hit = "bottom_first"
                break
            elif high_j >= barrier_top:
                label = 1
                hit = "top"
                break
            elif low_j <= barrier_bot:
                label = -1
                hit = "bottom"
                break
        
        results.append({
            "label_tb": label,
            "label_tb_bin": 1 if label == 1 else 0,
            "barrier_top": barrier_top,
            "barrier_bot": barrier_bot,
            "hit_barrier": hit
        })
    
    # Tambahkan NaN untuk N baris terakhir
    for _ in range(max_candles):
        results.append({
            "label_tb": None,
            "label_tb_bin": None,
            "barrier_top": None,
            "barrier_bot": None,
            "hit_barrier": None
        })
    
    label_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, label_df], axis=1)
```

### Dynamic Barrier (Adaptive)

Barrier statis (+/-0.15%) mungkin terlalu kecil saat volatilitas tinggi (barrier langsung kena noise)
atau terlalu besar saat volatilitas rendah (barrier tidak pernah kena, label selalu "timeout").

Solusi: barrier proporsional dengan realized volatility:

```python
# Contoh: barrier = 1x realized volatility 12-candle terakhir
profit_pct_dynamic = df["rvol_12"] * 1.0
loss_pct_dynamic = df["rvol_12"] * 1.0

# Atau dengan minimum/maximum cap:
profit_pct_dynamic = df["rvol_12"].clip(0.0005, 0.005)
```

---

## 3.4 Sample Weighting (Untuk Triple-Barrier)

### Masalah: Label Overlap (Concurrent Labels)

Triple-barrier menghasilkan label yang **overlap secara waktu**. Contoh:
- Label untuk candle t=100 melihat harga sampai candle t=106
- Label untuk candle t=102 melihat harga sampai candle t=108
- Kedua label "melihat" candle t=102 hingga t=106 yang sama

Ini berarti banyak sampel dalam dataset yang **tidak independen** — melanggar asumsi ML.

### Solusi: Sample Uniqueness Weighting

```python
def compute_sample_weights(df: pd.DataFrame, max_candles: int = 6) -> pd.Series:
    """
    Hitung bobot sampel berdasarkan keunikan label (uniqueness).
    Sampel yang labelnya banyak overlap dengan sampel lain diberi bobot lebih kecil.
    
    Formula (simplified):
        Untuk setiap sampel i, hitung berapa banyak sampel lain yang labelnya
        "concurrent" (overlapping) dengan sampel i.
        weight_i = 1 / n_concurrent_i
    
    Returns:
        pd.Series dengan index sama seperti df, nilai bobot [0, 1]
    
    Catatan: Implementasi penuh ada di mlfinpy.
    Versi sederhana: weight = 1/n jika ada n sampel yang overlap.
    
    Kalau skip ini: TANDAI di kode sebagai "known limitation — label overlap exists"
    dan sadari bahwa in-sample accuracy akan terlihat lebih bagus dari realita.
    """
    n = len(df)
    weights = np.ones(n)
    
    for i in range(n):
        # Hitung berapa sampel lain yang labelnya cover candle i
        concurrent = 0
        for j in range(max(0, i - max_candles), min(n, i + max_candles)):
            if j != i:
                # Cek apakah label j overlap dengan label i
                # (simplified: semua yang dalam max_candles dianggap concurrent)
                concurrent += 1
        weights[i] = 1.0 / max(1, concurrent / 2)
    
    # Normalize ke [0, 1]
    weights = weights / weights.max()
    return pd.Series(weights, index=df.index)
```

> **Untuk MVP/awal:** Skip sample weighting dan **tandai di kode** sebagai known limitation.
> Cukup buang (drop) sampel dengan label "timeout" (label=0) untuk menyederhanakan jadi binary.

---

## 3.5 Perbandingan dan Analisis Label

```python
def analyze_labels(df_with_labels: pd.DataFrame) -> dict:
    """
    Analisis distribusi label untuk memastikan kualitas.
    
    Metrics:
        - Distribusi kelas (class balance): berapa % label 0 vs 1 vs -1
        - Distribusi waktu label kena barrier (untuk triple-barrier)
        - Korelasi antara label fixed-horizon dan triple-barrier
        - Distribusi future_ret per kelas label
    
    Target distribusi yang sehat:
        - Fixed-horizon: idealnya ~50/50 (karena market sering sideways)
        - Triple-barrier: biasanya banyak "timeout" (0); 
          idealnya 30-40% label 1, 30-40% label -1, 20-30% timeout
    
    Red flags:
        - Distribusi sangat tidak seimbang (>80% satu kelas) → cek implementasi
        - Hampir semua timeout → barrier terlalu besar
        - Hampir tidak ada timeout → barrier terlalu kecil
    """
    stats = {}
    
    if "label_fh" in df_with_labels.columns:
        fh_counts = df_with_labels["label_fh"].value_counts(normalize=True)
        stats["fixed_horizon"] = {
            "pct_up": fh_counts.get(1, 0),
            "pct_down": fh_counts.get(0, 0),
            "class_balance": fh_counts.get(1, 0) / max(fh_counts.get(0, 0.01), 0.01)
        }
    
    if "label_tb" in df_with_labels.columns:
        tb_counts = df_with_labels["label_tb"].value_counts(normalize=True)
        stats["triple_barrier"] = {
            "pct_up": tb_counts.get(1, 0),
            "pct_down": tb_counts.get(-1, 0),
            "pct_timeout": tb_counts.get(0, 0),
            "hit_distribution": df_with_labels["hit_barrier"].value_counts().to_dict()
        }
    
    return stats
```

---

## 3.6 Pipeline Utama `labeling.py`

```python
def build_labels(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Main labeling orchestrator.
    
    Steps:
    1. Jalankan fixed-horizon labeling
    2. Jalankan triple-barrier labeling
    3. Analisis distribusi label, log hasil
    4. Hitung sample weights (atau log sebagai known limitation)
    5. Return df dengan semua kolom label
    
    Output columns ditambahkan ke df:
        label_fh, future_ret  (fixed-horizon)
        label_tb, label_tb_bin, barrier_top, barrier_bot, hit_barrier  (triple-barrier)
        sample_weight  (opsional)
    """
    method = config.get("labeling", {}).get("method", "triple_barrier")
    
    fh_config = config.get("labeling", {}).get("fixed_horizon", {})
    tb_config = config.get("labeling", {}).get("triple_barrier", {})
    
    # Selalu buat keduanya untuk perbandingan
    df = label_fixed_horizon(df, n_ahead=fh_config.get("n_candles_ahead", 1))
    df = label_triple_barrier(
        df,
        profit_pct=tb_config.get("profit_pct", 0.0015),
        loss_pct=tb_config.get("loss_pct", 0.0015),
        max_candles=tb_config.get("max_candles", 6)
    )
    
    # Drop baris dengan NaN label (baris terakhir)
    df = df.dropna(subset=["label_fh", "label_tb"])
    
    # Analisis dan log
    stats = analyze_labels(df)
    logger.info(f"Label stats: {stats}")
    
    # Primary label berdasarkan config
    df["label"] = df["label_tb_bin"] if method == "triple_barrier" else df["label_fh"]
    
    return df
```

---

## 3.7 Unit Tests (`tests/test_labeling.py`)

```python
import pytest
import pandas as pd
import numpy as np
from src.labeling import label_fixed_horizon, label_triple_barrier

def make_controlled_kline():
    """Buat kline dengan harga yang bisa kita prediksi labelnya."""
    # Buat sequens harga: naik 0.5% lalu turun
    prices = [50000, 50250, 50100, 49950, 50300]  # 5 candle
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.001 for p in prices],
        "low":  [p * 0.999 for p in prices],
        "close": prices,
        "volume": [100] * 5,
        "taker_buy_base": [50] * 5,
    }, index=idx)

class TestFixedHorizon:
    def test_label_correct(self):
        """Label harus 1 jika close[t+1] > close[t]."""
        df = make_controlled_kline()
        result = label_fixed_horizon(df, n_ahead=1)
        # close[0]=50000, close[1]=50250 → naik → label=1
        assert result["label_fh"].iloc[0] == 1
        # close[1]=50250, close[2]=50100 → turun → label=0
        assert result["label_fh"].iloc[1] == 0
    
    def test_last_n_rows_are_nan(self):
        """N baris terakhir harus NaN (tidak ada future data)."""
        df = make_controlled_kline()
        result = label_fixed_horizon(df, n_ahead=1)
        assert pd.isna(result["label_fh"].iloc[-1])
    
    def test_no_lookahead_in_label(self):
        """Ubah candle ke depan, label candle sebelumnya tidak boleh berubah."""
        df = make_controlled_kline()
        result_original = label_fixed_horizon(df.copy(), n_ahead=1)
        
        df_modified = df.copy()
        df_modified["close"].iloc[-1] *= 10  # Ubah candle terakhir secara drastis
        result_modified = label_fixed_horizon(df_modified, n_ahead=1)
        
        # Label di baris ke-3 (bukan baris ke-4 yang terpengaruh perubahan close[4])
        # tidak boleh berubah
        assert result_original["label_fh"].iloc[-3] == result_modified["label_fh"].iloc[-3]

class TestTripleBarrier:
    def test_top_barrier_hit(self):
        """Jika high cukup tinggi, label harus 1."""
        idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
        prices = [50000] * 10
        highs = [50000] * 10
        highs[1] = 50200  # +0.4% — di atas barrier 0.15%
        df = pd.DataFrame({
            "open": prices, "high": highs, 
            "low": [p * 0.9995 for p in prices],
            "close": prices, "volume": [100] * 10,
            "taker_buy_base": [50] * 10,
        }, index=idx)
        result = label_triple_barrier(df, profit_pct=0.0015, loss_pct=0.0015, max_candles=6)
        assert result["label_tb"].iloc[0] == 1
    
    def test_bottom_barrier_hit(self):
        """Jika low cukup rendah, label harus -1."""
        idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
        prices = [50000] * 10
        lows = [p * 0.9995 for p in prices]
        lows[1] = 49800  # -0.4% — di bawah barrier -0.15%
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0005 for p in prices],
            "low": lows,
            "close": prices, "volume": [100] * 10,
            "taker_buy_base": [50] * 10,
        }, index=idx)
        result = label_triple_barrier(df, profit_pct=0.0015, loss_pct=0.0015, max_candles=6)
        assert result["label_tb"].iloc[0] == -1
    
    def test_vertical_barrier_hit(self):
        """Jika tidak ada barrier yang kena, label harus 0 (timeout)."""
        idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
        prices = [50000] * 10
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],  # hanya naik 0.01%, tidak sentuh barrier 0.15%
            "low":  [p * 0.9999 for p in prices],   # hanya turun 0.01%
            "close": prices, "volume": [100] * 10,
            "taker_buy_base": [50] * 10,
        }, index=idx)
        result = label_triple_barrier(df, profit_pct=0.0015, loss_pct=0.0015, max_candles=6)
        assert result["label_tb"].iloc[0] == 0
    
    def test_label_range(self):
        """Label triple-barrier hanya boleh -1, 0, atau 1."""
        idx = pd.date_range("2024-01-01", periods=50, freq="5min", tz="UTC")
        prices = 50000 + np.random.randn(50) * 100
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices,
            "volume": [100] * 50, "taker_buy_base": [50] * 50,
        }, index=idx)
        result = label_triple_barrier(df, max_candles=6)
        valid_labels = result["label_tb"].dropna()
        assert set(valid_labels.unique()).issubset({-1, 0, 1})
```

---

## 3.8 Checklist Selesai Phase 3

- [ ] `label_fixed_horizon()` diimplementasikan dan ditest
- [ ] `label_triple_barrier()` diimplementasikan dan ditest
- [ ] Semua unit test di `test_labeling.py` PASS
- [ ] `analyze_labels()` bisa menampilkan distribusi kelas
- [ ] Distribusi label divisualisasikan di `notebooks/01_data_exploration.ipynb`
- [ ] Label overlap (concurrent labels) sudah di-acknowledge dalam kode/README
- [ ] `build_labels()` menyimpan dataset berlabel ke `data/processed/labels/`
- [ ] Dataset final: kolom fitur (dari Phase 2) + kolom label (dari Phase 3)

**→ Lanjut ke [Phase 4 — Model & Validasi](./phase-4-model-validation.md)**
