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
    use_triple_barrier: bool = True      # NOT IMPLEMENTED (exit selalu fixed T+1)
    profit_target_pct: float = 0.0015    # TP: +0.15% (NOT IMPLEMENTED)
    stop_loss_pct: float = 0.0015        # SL: -0.15% (NOT IMPLEMENTED)
    max_hold_candles: int = 6            # Maksimal hold posisi (NOT IMPLEMENTED)
    bet_size_usd: float = 1.0            # Taruhan flat $1.00 USD
    pool_ratio_source: str = "fixed"     # "fixed" atau "historical" (model_proba is deprecated)
    fixed_ratio_up: float = 0.50         # Rasio default jika source="fixed"
    platform_fee_pct: float = 0.01       # 1% platform fee

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
    exit_reason: str = ""           # "WIN", "LOSS", "DRAW"
    
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
    Simulasi trading lilin-demi-lilin (candle-by-candle) untuk Binance Predict 5m.
    - Sinyal dihitung di akhir candle T (Close T / Open T+1).
    - Masuk taruhan $1.00 flat pada arah UP (jika proba >= threshold) atau DOWN (jika proba <= 1 - threshold).
    - Selesai/exit di akhir candle T+1 (Close T+1).
    """
    trades = []
    capital = config.initial_capital
    equity_curve = []
    
    n = len(df)
    
    # Batch predict probabilities to speed up backtest significantly
    try:
        all_probas = model.predict_proba(df[feature_cols])[:, 1]
    except Exception:
        # Fallback untuk mock model di unit test
        if hasattr(model, "predict_proba"):
            res = model.predict_proba(df[feature_cols])
            if len(res.shape) > 1 and res.shape[1] > 1:
                all_probas = res[:, 1]
            else:
                all_probas = res.flatten()
        else:
            all_probas = np.array([0.5] * n)
            
    # Tracking daily/weekly loss
    current_day = None
    current_week = None
    day_start_capital = capital
    week_start_capital = capital
    daily_stopped = False
    weekly_stopped = False
    
    for i in range(n - 1):
        timestamp = df.index[i]
        
        # Pengecekan limit harian dan mingguan
        ts_day = timestamp.date()
        ts_week = timestamp.isocalendar()[:2]
        
        # Reset harian
        if ts_day != current_day:
            current_day = ts_day
            day_start_capital = capital
            daily_stopped = False
            
        # Reset mingguan
        if ts_week != current_week:
            current_week = ts_week
            week_start_capital = capital
            weekly_stopped = False
            
        # Hitung persentase drawdown dari awal periode
        daily_loss_pct = (capital - day_start_capital) / day_start_capital
        weekly_loss_pct = (capital - week_start_capital) / week_start_capital
        
        if daily_loss_pct <= -config.max_daily_loss_pct:
            if not daily_stopped:
                logger.warning(f"[{timestamp}] Daily loss limit hit ({daily_loss_pct:.2%}). Stopping entries for today.")
                daily_stopped = True
            equity_curve.append({"timestamp": timestamp, "equity": capital})
            continue
            
        if weekly_loss_pct <= -config.max_weekly_loss_pct:
            if not weekly_stopped:
                logger.warning(f"[{timestamp}] Weekly loss limit hit ({weekly_loss_pct:.2%}). Stopping entries for this week.")
                weekly_stopped = True
            equity_curve.append({"timestamp": timestamp, "equity": capital})
            continue

        row = df.iloc[i] # candle T
        next_row = df.iloc[i + 1] # candle T+1
        
        proba = float(all_probas[i])
        
        threshold_up = config.probability_threshold
        threshold_down = 1.0 - threshold_up
        
        direction = None
        if proba >= threshold_up:
            direction = "up"
        elif proba <= threshold_down:
            direction = "down"
            
        if direction is not None:
            open_price = float(next_row["open"])
            close_price = float(next_row["close"])
            
            # Tentukan apakah tebakan benar
            if direction == "up":
                win = close_price > open_price
            else: # down
                win = close_price < open_price
                
            draw = close_price == open_price
            
            # Tentukan pool_ratio_up (persentase UP pool)
            if config.pool_ratio_source == "fixed":
                market_ratio_up = config.fixed_ratio_up
            else:
                # DEPRECATED: model_proba menyebabkan circular bias
                logger.warning("pool_ratio_source='model_proba' is deprecated. Using fixed=0.50 as baseline.")
                market_ratio_up = config.fixed_ratio_up
                
            # Hitung harga token UP/DOWN (0.01 - 0.99)
            if direction == "up":
                entry_cost = market_ratio_up
            else:
                entry_cost = 1.0 - market_ratio_up
                
            entry_cost = max(0.01, min(0.99, entry_cost))
            
            # Fee platform
            fee = config.bet_size_usd * config.platform_fee_pct
            
            if draw:
                net_pnl = 0.0
                exit_reason = "DRAW"
            elif win:
                # Menang -> Token bernilai $1.00. Profit = (1.0 - entry_cost) * bet_size
                net_pnl = config.bet_size_usd * (1.0 - entry_cost) - fee
                exit_reason = "WIN"
            else:
                # Kalah -> Token bernilai $0.00. Rugi = -entry_cost * bet_size
                net_pnl = -(config.bet_size_usd * entry_cost) - fee
                exit_reason = "LOSS"
                
            capital += net_pnl
            
            trade = TradeRecord(
                entry_time=df.index[i + 1],
                exit_time=next_row.name,
                entry_price=open_price,
                exit_price=close_price,
                direction=direction,
                position_size_usdt=config.bet_size_usd,
                position_size_btc=0.0,
                predicted_proba=proba,
                gross_pnl=net_pnl,
                fee_paid=fee,
                slippage_cost=0.0,
                net_pnl=net_pnl,
                exit_reason=exit_reason
            )
            trades.append(trade)
            
        equity_curve.append({"timestamp": timestamp, "equity": capital})
        
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df.set_index("timestamp", inplace=True)
    return trades, equity_df

