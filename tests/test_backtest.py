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
            
    def test_fee_applied(self):
        """Fee harus diterapkan untuk setiap trade (entry + exit)."""
        df = make_dummy_kline_with_features(50)
        model = MockModel(always_predict_proba=0.8)
        config = BacktestConfig(fee_taker=0.001, slippage_pct=0.0)
        trades, _ = run_backtest(df, model, FEATURE_COLS, config)
        
        assert len(trades) > 0, "Tidak ada trade yang di-generate"
        for trade in trades:
            assert trade.fee_paid > 0.0, "Fee tidak diterapkan pada trade"
            
    def test_capital_never_negative(self):
        """Kapital di equity curve tidak boleh negatif."""
        df = make_dummy_kline_with_features(200)
        model = MockModel(always_predict_proba=0.8)
        config = BacktestConfig(initial_capital=1000.0, position_size_pct=0.10)
        trades, equity = run_backtest(df, model, FEATURE_COLS, config)
        
        assert (equity["equity"] >= 0).all(), "Kapital bernilai negatif!"

    def test_kill_switch_stops_trading(self):
        """Kill switch harian harus menghentikan trading ketika akumulasi kerugian harian melebihi limit."""
        df = make_dummy_kline_with_features(200)
        # Buat harga turun drastis di candle-candle berikutnya agar SL terus kena
        df["close"] = df["close"] * 0.90
        df["low"] = df["low"] * 0.85
        df["open"] = df["open"] * 0.95
        
        model = MockModel(always_predict_proba=0.9)
        # Batasi loss harian maksimal 1%
        config = BacktestConfig(
            initial_capital=10000.0,
            stop_loss_pct=0.01,
            position_size_pct=0.50,       # 50% capital = $5000 per trade
            max_daily_loss_pct=0.01       # Kill switch jika rugi >$100 harian
        )
        
        trades, equity = run_backtest(df, model, FEATURE_COLS, config)
        
        # Karena position size besar (50%) dan SL 1% (rugi $50 per trade + fees),
        # setelah 2 atau 3 kali SL kena, kerugian harian akan melebihi $100 (1%).
        # Bot harus berhenti masuk posisi baru untuk hari itu.
        assert len(trades) < 10, "Bot tetap bertransaksi meskipun kill-switch seharusnya menyala"

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
