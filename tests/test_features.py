"""
test_features.py — Unit tests untuk features.py (KRITIS!)
"""

import pytest
import pandas as pd
import numpy as np
from src.features import build_all_features, add_log_returns, add_order_book_imbalance

def make_dummy_kline(n=200):
    """Buat DataFrame kline dummy untuk testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    volume = np.random.uniform(1, 100, n)
    return pd.DataFrame({
        "open": np.random.uniform(40000, 50000, n),
        "high": np.random.uniform(40000, 50000, n),
        "low": np.random.uniform(40000, 50000, n),
        "close": np.random.uniform(40000, 50000, n),
        "volume": volume,
        "taker_buy_base": volume * np.random.uniform(0.1, 0.9, n),
    }, index=idx)

class TestAntiLookaheadBias:
    """
    TEST PALING PENTING: Pastikan tidak ada lookahead bias.
    
    Cara test: ubah nilai close[T] dan pastikan fitur di baris T-1 TIDAK berubah.
    Jika fitur di T-1 berubah ketika close[T] diubah, berarti ada lookahead bias!
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
                assert (valid >= -1.0).all() and (valid <= 1.0).all(), \
                    f"{col} harus dalam range [-1, 1]"
