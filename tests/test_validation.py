"""
test_validation.py — Unit tests untuk validation.py
"""

import pytest
import pandas as pd
import numpy as np
from src.validation import walk_forward_splits, WalkForwardConfig

def make_dummy_dataset(n=1000):
    """Buat dummy dataset dengan DatetimeIndex."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "label": np.random.choice([0, 1], n)
    }, index=idx)

class TestWalkForward:
    def test_no_temporal_leakage(self):
        """Test bahwa test data tidak overlap dengan train data dan posisinya selalu setelah train data."""
        df = make_dummy_dataset(1000)
        config = WalkForwardConfig(n_splits=5, train_period_days=30, test_period_days=7)
        
        for train_idx, test_idx in walk_forward_splits(df, config):
            # Test tidak boleh ada overlap
            overlap = train_idx.intersection(test_idx)
            assert len(overlap) == 0, "Train dan test index overlap — temporal leakage!"
            
            # Test harus selalu setelah train
            assert train_idx[-1] < test_idx[0], "Test period terjadi sebelum train period!"
            
    def test_expanding_window(self):
        """Train window harus makin besar setiap fold (Expanding Window)."""
        df = make_dummy_dataset(2000)
        config = WalkForwardConfig(n_splits=5, train_period_days=30, test_period_days=7)
        
        train_sizes = []
        for train_idx, test_idx in walk_forward_splits(df, config):
            train_sizes.append(len(train_idx))
            
        # Setiap fold train size harus lebih besar dari fold sebelumnya
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1], "Train window tidak expanding!"
            
    def test_embargo_respected(self):
        """Pastikan embargo period dihormati dan terdapat gap antara train dan test."""
        df = make_dummy_dataset(1000)
        config = WalkForwardConfig(
            n_splits=3, train_period_days=20, test_period_days=5,
            embargo_periods=12
        )
        
        for train_idx, test_idx in walk_forward_splits(df, config):
            train_end_pos = df.index.get_loc(train_idx[-1])
            test_start_pos = df.index.get_loc(test_idx[0])
            gap = test_start_pos - train_end_pos
            assert gap >= config.embargo_periods, \
                f"Embargo tidak cukup dihormati: gap={gap}, required={config.embargo_periods}"
