"""
test_labeling.py — Unit tests untuk labeling.py
"""

import pytest
import pandas as pd
import numpy as np
from src.labeling import label_fixed_horizon, label_triple_barrier

def make_controlled_kline():
    """Buat kline dengan harga yang bisa kita prediksi labelnya."""
    # Naik 0.5% lalu turun
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
        
        # Label di baris ke-3 tidak boleh berubah
        assert result_original["label_fh"].iloc[-3] == result_modified["label_fh"].iloc[-3]

class TestTripleBarrier:
    def test_top_barrier_hit(self):
        """Jika high cukup tinggi, label harus 1."""
        idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
        prices = [50000] * 10
        highs = [50000] * 10
        highs[1] = 50200  # +0.4% — di atas barrier 0.15% (profit_pct)
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
        lows[1] = 49800  # -0.4% — di bawah barrier -0.15% (loss_pct)
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
            "high": [p * 1.0001 for p in prices],  # hanya naik 0.01%
            "low":  [p * 0.9999 for p in prices],  # hanya turun 0.01%
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
        assert set(valid_labels.unique()).issubset({-1.0, 0.0, 1.0})
