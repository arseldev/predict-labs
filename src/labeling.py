"""
labeling.py — Fixed-horizon & Triple-barrier labeling

Fungsi utama:
- label_fixed_horizon(): Label sederhana berdasarkan pergerakan N candle ke depan
- label_triple_barrier(): Label triple-barrier (TP/SL/waktu)
- compute_sample_weights(): Bobot keunikan sampel untuk data overlap
- analyze_labels(): Distribusi kelas label
- build_labels(): Orchestrator pelabelan utama
"""

import pandas as pd
import numpy as np
from loguru import logger

def label_fixed_horizon(df: pd.DataFrame, n_ahead: int = 1) -> pd.DataFrame:
    """
    Label sederhana berdasarkan pergerakan harga N candle ke depan.
    """
    df = df.copy()
    
    # Ambil harga close N candle ke depan
    df["future_close"] = df["close"].shift(-n_ahead)
    df["future_ret"] = np.log(df["future_close"] / df["close"])
    df["label_fh"] = (df["future_close"] > df["close"]).astype(int)
    
    # Tandai N baris terakhir sebagai NaN pada label karena masa depannya belum ada
    df.iloc[-n_ahead:, df.columns.get_loc("label_fh")] = np.nan
    df.iloc[-n_ahead:, df.columns.get_loc("future_ret")] = np.nan
    
    df.drop(columns=["future_close"], inplace=True)
    return df

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
    - Barrier Atas (TP): Close * (1 + profit_pct)
    - Barrier Bawah (SL): Close * (1 - loss_pct)
    - Barrier Vertikal (Time): max_candles
    
    Label = 1 jika TP kena duluan
    Label = -1 jika SL kena duluan
    Label = 0 jika timeout (barrier vertikal kena duluan)
    """
    df = df.copy()
    results = []
    
    for i in range(len(df) - max_candles):
        entry_price = df["close"].iloc[i]
        
        if use_dynamic_barrier:
            # Pastikan kolom volatilitas ada
            if vol_col in df.columns:
                vol = df[vol_col].iloc[i]
                # Gunakan 1.0x realized volatility
                profit_pct_i = vol
                loss_pct_i = vol
            else:
                profit_pct_i = profit_pct
                loss_pct_i = loss_pct
        else:
            profit_pct_i = profit_pct
            loss_pct_i = loss_pct
            
        # Terapkan minimum/maximum cap agar barrier realistis
        profit_pct_i = np.clip(profit_pct_i, 0.0005, 0.005)
        loss_pct_i = np.clip(loss_pct_i, 0.0005, 0.005)
        
        barrier_top = entry_price * (1.0 + profit_pct_i)
        barrier_bot = entry_price * (1.0 - loss_pct_i)
        
        label = 0
        hit = "vertical"
        
        for j in range(1, max_candles + 1):
            if i + j >= len(df):
                break
            high_j = df["high"].iloc[i + j]
            low_j = df["low"].iloc[i + j]
            
            # Konservatif: Jika high dan low menyentuh kedua barrier pada candle yang sama, asumsikan SL kena duluan
            if high_j >= barrier_top and low_j <= barrier_bot:
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
        
    # Isi N baris terakhir dengan None/NaN karena time-horizon tidak mencukupi
    for _ in range(max_candles):
        results.append({
            "label_tb": np.nan,
            "label_tb_bin": np.nan,
            "barrier_top": np.nan,
            "barrier_bot": np.nan,
            "hit_barrier": None
        })
        
    label_df = pd.DataFrame(results, index=df.index)
    
    # Gabungkan kembali
    return pd.concat([df, label_df], axis=1)

def compute_sample_weights(df: pd.DataFrame, max_candles: int = 6) -> pd.Series:
    """
    Hitung bobot keunikan sampel untuk menangani concurrent label overlap.
    """
    n = len(df)
    weights = np.ones(n)
    
    # Estimasi kasar: jumlah overlap concurrency
    for i in range(n):
        concurrent = 0
        for j in range(max(0, i - max_candles), min(n, i + max_candles)):
            if j != i:
                concurrent += 1
        weights[i] = 1.0 / max(1.0, concurrent / 2.0)
        
    # Normalisasi bobot agar max = 1.0
    weights = weights / weights.max()
    return pd.Series(weights, index=df.index, name="sample_weight")

def analyze_labels(df_with_labels: pd.DataFrame) -> dict:
    """
    Analisis statistik distribusi kelas label.
    """
    stats = {}
    
    if "label_fh" in df_with_labels.columns:
        fh_counts = df_with_labels["label_fh"].dropna().value_counts(normalize=True)
        stats["fixed_horizon"] = {
            "pct_up": float(fh_counts.get(1, 0.0)),
            "pct_down": float(fh_counts.get(0, 0.0)),
            "class_balance": float(fh_counts.get(1, 0.0) / max(fh_counts.get(0, 0.01), 0.01))
        }
        
    if "label_tb" in df_with_labels.columns:
        tb_counts = df_with_labels["label_tb"].dropna().value_counts(normalize=True)
        hit_counts = df_with_labels["hit_barrier"].dropna().value_counts().to_dict()
        stats["triple_barrier"] = {
            "pct_up": float(tb_counts.get(1.0, 0.0)),
            "pct_down": float(tb_counts.get(-1.0, 0.0)),
            "pct_timeout": float(tb_counts.get(0.0, 0.0)),
            "hit_distribution": hit_counts
        }
        
    return stats

def build_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Orchestrator utama labeling.
    """
    method = config.get("labeling", {}).get("method", "triple_barrier")
    
    fh_config = config.get("labeling", {}).get("fixed_horizon", {})
    tb_config = config.get("labeling", {}).get("triple_barrier", {})
    
    # Hasilkan kedua versi label untuk analisis perbandingan
    df = label_fixed_horizon(df, n_ahead=fh_config.get("n_candles_ahead", 1))
    
    # Tambahkan realized volatility 12-candle untuk dynamic barrier jika tersedia
    if "rvol_12" not in df.columns:
        # Jika belum ada, panggil rvol secara lokal
        from src.features import add_realized_vol
        df = add_realized_vol(df, windows=[12])
        
    df = label_triple_barrier(
        df,
        profit_pct=tb_config.get("profit_pct", 0.0015),
        loss_pct=tb_config.get("loss_pct", 0.0015),
        max_candles=tb_config.get("max_candles", 6),
        use_dynamic_barrier=tb_config.get("use_dynamic_barrier", False),
        vol_col="rvol_12"
    )
    
    # Hitung sample weights
    df["sample_weight"] = compute_sample_weights(df, max_candles=tb_config.get("max_candles", 6))
    
    # Drop baris dengan NaN label pada target utama
    df_clean = df.dropna(subset=["label_fh", "label_tb"])
    
    stats = analyze_labels(df_clean)
    logger.info(f"Label analysis stats: {stats}")
    
    # Set primary label ke kolom 'label'
    if method == "triple_barrier":
        df_clean["label"] = df_clean["label_tb_bin"]
    else:
        df_clean["label"] = df_clean["label_fh"]
        
    return df_clean
