"""
validation.py — Time-Series Cross-Validation

Implements:
1. Walk-Forward Validation (expanding window)
2. Purged K-Fold CV (with embargo)
"""

from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd
import numpy as np
from loguru import logger

@dataclass
class WalkForwardConfig:
    n_splits: int = 10
    train_period_days: int = 45
    test_period_days: int = 7
    embargo_periods: int = 12  # Jumlah candle untuk dilewati (e.g. 12 × 5m = 1 jam)
    min_train_samples: int = 1000

def walk_forward_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Generator: menghasilkan (train_idx, test_idx) untuk setiap fold.
    Menerapkan expanding window strategy dengan embargo period setelah training set.
    """
    # Deteksi frekuensi data untuk konversi hari ke jumlah candle
    # Default ke 288 candle per hari untuk timeframe 5m jika tidak terdeteksi
    candles_per_day = 288
    
    # Coba deteksi timeframe dominan dari index
    if len(df) > 1:
        time_diff = df.index[1] - df.index[0]
        minutes = time_diff.total_seconds() / 60.0
        if minutes > 0:
            candles_per_day = int((24 * 60) // minutes)
            
    train_candles = config.train_period_days * candles_per_day
    test_candles = config.test_period_days * candles_per_day
    embargo = config.embargo_periods
    
    n = len(df)
    
    for i in range(config.n_splits):
        # Expanding window: train mulai dari awal data, test bergeser maju
        train_end_idx = train_candles + i * test_candles
        if train_end_idx >= n:
            logger.debug(f"Train end index {train_end_idx} exceeds data length {n} at fold {i+1}. Stopping split.")
            break
            
        test_start_idx = train_end_idx + embargo
        test_end_idx = min(test_start_idx + test_candles, n)
        
        if test_start_idx >= n or len(df.index[test_start_idx:test_end_idx]) == 0:
            logger.debug(f"Test start index {test_start_idx} exceeds data length {n} at fold {i+1}. Stopping split.")
            break
            
        train_idx = df.index[:train_end_idx]
        test_idx = df.index[test_start_idx:test_end_idx]
        
        if len(train_idx) < config.min_train_samples:
            logger.debug(f"Train samples count {len(train_idx)} is less than minimum {config.min_train_samples}. Skipping fold.")
            continue
            
        yield train_idx, test_idx

def purged_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    embargo_pct: float = 0.01
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Purged K-Fold CV: Membagi data menjadi K fold, membuang data training yang overlap
    secara temporal (label overlap) dengan data testing menggunakan embargo.
    """
    n = len(df)
    fold_size = n // n_folds
    embargo_size = int(n * embargo_pct)
    
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = test_start + fold_size
        
        # Purge & Embargo sebelum dan sesudah test set
        purge_start = max(0, test_start - embargo_size)
        purge_end = min(n, test_end + embargo_size)
        
        train_mask = np.ones(n, dtype=bool)
        train_mask[purge_start:purge_end] = False
        
        train_idx = df.index[train_mask]
        test_idx = df.index[test_start:test_end]
        
        yield train_idx, test_idx
