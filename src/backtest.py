"""
backtest.py — Backtesting Engine dengan Fee & Slippage Realistis

Kelas & Fungsi utama:
- BacktestConfig: Parameter simulasi
- TradeRecord: Pencatatan detail per trade
- run_backtest(): Loop backtest utama (candle-by-candle)
- _check_exit(): Helper pengecekan kondisi exit
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0     # USDT
    fee_taker: float = 0.001             # 0.10% taker fee
    fee_maker: float = 0.001             # 0.10% maker fee
    slippage_pct: float = 0.0002         # 0.02% slippage
    probability_threshold: float = 0.60  # Threshold minimal model proba untuk entry
    position_size_pct: float = 0.02      # 2% dari kapital per trade
    max_daily_loss_pct: float = 0.03     # Stop trading jika rugi >3% sehari
    max_weekly_loss_pct: float = 0.08    # Stop trading jika rugi >8% seminggu
    use_triple_barrier: bool = True
    profit_target_pct: float = 0.0015    # TP: +0.15%
    stop_loss_pct: float = 0.0015        # SL: -0.15%
    max_hold_candles: int = 6            # Maksimal hold posisi (30 menit)

@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    direction: str                  # 'long'
    position_size_usdt: float       # Ukuran posisi (USDT)
    position_size_btc: float        # Ukuran posisi (BTC)
    predicted_proba: float          # Probabilitas prediksi model
    
    # Hasil pnl & biaya
    gross_pnl: float = 0.0
    fee_paid: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0            # gross_pnl - fee_paid - slippage_cost
    exit_reason: str = ""           # "tp", "sl", "timeout", "eod"
    
    @property
    def net_pnl_pct(self) -> float:
        if self.position_size_usdt == 0:
            return 0.0
        return self.net_pnl / self.position_size_usdt

def run_backtest(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    config: BacktestConfig
) -> Tuple[List[TradeRecord], pd.DataFrame]:
    """
    Simulasi trading lilin-demi-lilin (candle-by-candle).
    
    Anti-lookahead:
    - Sinyal dihitung di akhir candle T (Close T).
    - Eksekusi order entry di open candle T+1 (Open T+1) dengan slippage.
    """
    trades = []
    capital = config.initial_capital
    equity_curve = []
    
    active_trade = None
    daily_pnl = {}
    
    n = len(df)
    
    for i in range(n - 1):
        timestamp = df.index[i]
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        
        # --- KILL SWITCH CHECK ---
        today = timestamp.date()
        # Jika kerugian hari ini melebihi limit harian, batalkan trading untuk sisa hari ini
        if daily_pnl.get(today, 0) < -config.initial_capital * config.max_daily_loss_pct:
            # Tetap log equity berjalan
            current_equity = capital
            if active_trade:
                current_equity += active_trade.position_size_btc * row["close"] - \
                                  active_trade.position_size_btc * active_trade.entry_price
            equity_curve.append({"timestamp": timestamp, "equity": current_equity})
            continue
            
        # --- EXIT CHECK (Jika sedang memegang posisi) ---
        if active_trade is not None:
            active_trade, capital = _check_exit(active_trade, row, df, i, capital, config)
            if active_trade.exit_time is not None:
                trades.append(active_trade)
                daily_pnl[today] = daily_pnl.get(today, 0) + active_trade.net_pnl
                active_trade = None
                
        # --- SIGNAL & ENTRY GENERATION ---
        if active_trade is None:
            # Hitung probabilitas naik menggunakan baris saat ini (T)
            features = df[feature_cols].iloc[i:i+1]
            
            # Mendukung model sklearn Pipeline, XGBoost, LightGBM
            try:
                proba = model.predict_proba(features)[0, 1]
            except Exception:
                # Fallback untuk mock model
                proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else 0.5
                
            if proba >= config.probability_threshold:
                # Masuk di open candle berikutnya (T+1) + slippage (taker buy)
                entry_price = next_row["open"] * (1.0 + config.slippage_pct)
                position_usdt = capital * config.position_size_pct
                position_btc = position_usdt / entry_price
                entry_fee = position_usdt * config.fee_taker
                
                active_trade = TradeRecord(
                    entry_time=df.index[i + 1],
                    exit_time=None,
                    entry_price=entry_price,
                    exit_price=None,
                    direction="long",
                    position_size_usdt=position_usdt,
                    position_size_btc=position_btc,
                    predicted_proba=proba,
                    fee_paid=entry_fee,
                    slippage_cost=position_usdt * config.slippage_pct
                )
                capital -= (position_usdt + entry_fee)
                
        # --- LOG EQUITY ---
        current_equity = capital
        if active_trade:
            # Mark-to-market berdasarkan close candle T
            current_equity += active_trade.position_size_btc * row["close"]
                              
        equity_curve.append({"timestamp": timestamp, "equity": current_equity})
        
    # Handle posisi yang masih terbuka di akhir data
    if active_trade is not None:
        last_row = df.iloc[-1]
        exit_price = last_row["close"] * (1.0 - config.slippage_pct)
        exit_fee = active_trade.position_size_btc * exit_price * config.fee_taker
        exit_slippage = active_trade.position_size_btc * exit_price * config.slippage_pct
        gross_pnl = active_trade.position_size_btc * (exit_price - active_trade.entry_price)
        
        net_pnl = gross_pnl - active_trade.fee_paid - exit_fee - active_trade.slippage_cost - exit_slippage
        
        active_trade.exit_time = df.index[-1]
        active_trade.exit_price = exit_price
        active_trade.gross_pnl = gross_pnl
        active_trade.fee_paid += exit_fee
        active_trade.slippage_cost += exit_slippage
        active_trade.net_pnl = net_pnl
        active_trade.exit_reason = "eod"
        
        trades.append(active_trade)
        capital += active_trade.position_size_usdt + net_pnl
        
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df.set_index("timestamp", inplace=True)
    return trades, equity_df

def _check_exit(
    trade: TradeRecord,
    current_row: pd.Series,
    df: pd.DataFrame,
    current_idx: int,
    capital: float,
    config: BacktestConfig
) -> Tuple[TradeRecord, float]:
    """
    Cek kondisi exit untuk posisi aktif berdasarkan Triple-Barrier.
    """
    tp_price = trade.entry_price * (1.0 + config.profit_target_pct)
    sl_price = trade.entry_price * (1.0 - config.stop_loss_pct)
    
    # Hitung berapa candle yang sudah dilewati sejak entry
    entry_pos = df.index.get_loc(trade.entry_time)
    candles_held = current_idx - entry_pos
    
    exit_price = None
    exit_reason = ""
    
    # Prioritas SL > TP > Timeout (Konservatif)
    if current_row["low"] <= sl_price:
        # Exit di harga SL (dikurangi slippage - taker sell)
        exit_price = sl_price * (1.0 - config.slippage_pct)
        exit_reason = "sl"
    elif current_row["high"] >= tp_price:
        exit_price = tp_price * (1.0 - config.slippage_pct)
        exit_reason = "tp"
    elif candles_held >= config.max_hold_candles:
        exit_price = current_row["close"] * (1.0 - config.slippage_pct)
        exit_reason = "timeout"
        
    if exit_price is not None:
        exit_fee = trade.position_size_btc * exit_price * config.fee_taker
        exit_slippage = trade.position_size_btc * exit_price * config.slippage_pct
        gross_pnl = trade.position_size_btc * (exit_price - trade.entry_price)
        
        net_pnl = gross_pnl - trade.fee_paid - exit_fee - trade.slippage_cost - exit_slippage
        
        trade.exit_time = df.index[current_idx]
        trade.exit_price = exit_price
        trade.gross_pnl = gross_pnl
        trade.fee_paid += exit_fee
        trade.slippage_cost += exit_slippage
        trade.net_pnl = net_pnl
        trade.exit_reason = exit_reason
        
        capital += trade.position_size_usdt + net_pnl
        
    return trade, capital
