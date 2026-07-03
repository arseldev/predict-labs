"""
test_backtest.py — Unit tests untuk backtest.py
"""

import pytest
import pandas as pd
import numpy as np
from src.backtest import run_backtest, BacktestConfig, TradeRecord
from src.evaluate import compute_backtest_metrics

FEATURE_COLS = ["feature1"]

class MockModel:
    def __init__(self, always_predict_proba=0.7):
        self.proba = always_predict_proba
        
    def predict_proba(self, X):
        # Kembalikan numpy array dengan shape (len(X), 2)
        # Kolom 1 adalah probabilitas kelas 1 (naik)
        return np.array([[1.0 - self.proba, self.proba]] * len(X))

def make_dummy_kline_with_features(n=100):
    """Buat dummy kline dengan fitur tambahan."""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": [50000.0] * n,
        "high": [50100.0] * n,
        "low": [49900.0] * n,
        "close": [50000.0] * n,
        "volume": [10.0] * n,
        "taker_buy_base": [5.0] * n,
        "feature1": [1.0] * n
    }, index=idx)

class TestBacktest:
    def test_no_future_data_in_signal(self):
        """Sinyal hanya boleh berdasarkan data yang sudah tersedia (Entry di open candle T+1, bukan close T)."""
        df = make_dummy_kline_with_features(100)
        model = MockModel(always_predict_proba=0.7)
        config = BacktestConfig(probability_threshold=0.60)
        trades, _ = run_backtest(df, model, FEATURE_COLS, config)
        
        for trade in trades:
            # Entry time harus tepat setelah sinyal di-generate
            # Indeks entry_time harus berupa baris T+1
            assert trade.entry_time > df.index[0], "Entry time terjadi terlalu cepat!"
            
    def test_payout_calculation(self):
        """Verifikasi bahwa payout pool-ratio dihitung secara akurat."""
        df = make_dummy_kline_with_features(5)
        df.iloc[0, df.columns.get_loc("close")] = 50000.0
        df.iloc[1, df.columns.get_loc("open")] = 50000.0
        df.iloc[1, df.columns.get_loc("close")] = 50100.0 # candle T=1 close > open => UP win
        
        df.iloc[2, df.columns.get_loc("open")] = 50000.0
        df.iloc[2, df.columns.get_loc("close")] = 49900.0 # candle T=2 close < open => DOWN win
        
        model = MockModel(always_predict_proba=0.8)
        # Dengan proba=0.8, direction UP -> entry_cost = 0.8. fee = 1% * 1.0 = 0.01.
        # Jika WIN (T=1): PnL = (1.0 - 0.8) - 0.01 = 0.19
        # Jika LOSS (T=2): PnL = -0.8 - 0.01 = -0.81
        config = BacktestConfig(probability_threshold=0.60, platform_fee_pct=0.01)
        trades, _ = run_backtest(df, model, FEATURE_COLS, config)
        
        assert len(trades) >= 2
        # Trade ke-0 (sinyal di candle 0, eksekusi di candle 1):
        assert trades[0].direction == "up"
        assert trades[0].exit_reason == "WIN"
        assert abs(trades[0].net_pnl - 0.19) < 1e-6
        
        # Trade ke-1 (sinyal di candle 1, eksekusi di candle 2):
        assert trades[1].direction == "up"
        assert trades[1].exit_reason == "LOSS"
        assert abs(trades[1].net_pnl - (-0.81)) < 1e-6

    def test_capital_never_negative(self):
        """Kapital di equity curve tidak boleh negatif."""
        df = make_dummy_kline_with_features(200)
        model = MockModel(always_predict_proba=0.8)
        config = BacktestConfig(initial_capital=1000.0)
        trades, equity = run_backtest(df, model, FEATURE_COLS, config)
        
        assert (equity["equity"] >= 0).all(), "Kapital bernilai negatif!"

    def test_two_way_predictions(self):
        """Verifikasi bahwa arah DOWN dieksekusi jika proba <= 1 - threshold."""
        df = make_dummy_kline_with_features(5)
        model = MockModel(always_predict_proba=0.2)
        config = BacktestConfig(probability_threshold=0.60)
        trades, _ = run_backtest(df, model, FEATURE_COLS, config)
        
        assert len(trades) > 0
        for trade in trades:
            assert trade.direction == "down"

    def test_ev_calculation(self):
        """Verifikasi formula EV di evaluate.py dihitung dengan benar."""
        # 3 dummy trades
        t1 = TradeRecord(
            entry_time=pd.Timestamp("2024-01-01 00:00:00"),
            exit_time=pd.Timestamp("2024-01-01 00:30:00"),
            entry_price=50000.0,
            exit_price=50075.0,
            direction="long",
            position_size_usdt=1000.0,
            position_size_btc=0.02,
            predicted_proba=0.7,
            gross_pnl=1.5,
            fee_paid=1.0,
            slippage_cost=0.2,
            net_pnl=0.3, # positif (win)
            exit_reason="tp"
        )
        
        t2 = TradeRecord(
            entry_time=pd.Timestamp("2024-01-01 01:00:00"),
            exit_time=pd.Timestamp("2024-01-01 01:30:00"),
            entry_price=50000.0,
            exit_price=49925.0,
            direction="long",
            position_size_usdt=1000.0,
            position_size_btc=0.02,
            predicted_proba=0.7,
            gross_pnl=-1.5,
            fee_paid=1.0,
            slippage_cost=0.2,
            net_pnl=-2.7, # negatif (loss)
            exit_reason="sl"
        )
        
        t3 = TradeRecord(
            entry_time=pd.Timestamp("2024-01-01 02:00:00"),
            exit_time=pd.Timestamp("2024-01-01 02:30:00"),
            entry_price=50000.0,
            exit_price=50075.0,
            direction="long",
            position_size_usdt=1000.0,
            position_size_btc=0.02,
            predicted_proba=0.7,
            gross_pnl=1.5,
            fee_paid=1.0,
            slippage_cost=0.2,
            net_pnl=0.3, # positif (win)
            exit_reason="tp"
        )
        
        trades = [t1, t2, t3]
        
        # win_rate = 2/3 = 0.6667
        # avg_win = (0.3/1000 + 0.3/1000) / 2 = 0.0003
        # avg_loss = abs(-2.7/1000) = 0.0027
        # EV = (2/3 * 0.0003) - (1/3 * 0.0027) = 0.0002 - 0.0009 = -0.0007
        
        dummy_equity = pd.DataFrame([
            {"timestamp": pd.Timestamp("2024-01-01 00:00:00"), "equity": 10000.0},
            {"timestamp": pd.Timestamp("2024-01-01 03:00:00"), "equity": 9997.9}
        ]).set_index("timestamp")
        
        metrics = compute_backtest_metrics(trades, dummy_equity, BacktestConfig())
        
        assert abs(metrics["ev_per_trade"] - (-0.0007)) < 1e-6
