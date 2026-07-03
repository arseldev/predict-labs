"""
features.py — Feature Engineering Pipeline

Semua fungsi di sini menerima DataFrame yang sudah closed (tanpa candle yang masih open)
dan mengembalikan DataFrame dengan kolom fitur tambahan.
"""

import os
import numpy as np
import pandas as pd
from loguru import logger
import ta

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan log return berbagai lag.
    """
    df = df.copy()
    df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_3"] = np.log(df["close"] / df["close"].shift(3))
    df["ret_5"] = np.log(df["close"] / df["close"].shift(5))
    df["ret_10"] = np.log(df["close"] / df["close"].shift(10))
    df["ret_1h"] = np.log(df["close"] / df["close"].shift(12))   # 12 * 5m = 1 jam
    df["ret_4h"] = np.log(df["close"] / df["close"].shift(48))   # 48 * 5m = 4 jam
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI standard dengan periode default 14.
    """
    df = df.copy()
    df[f"rsi_{window}"] = ta.momentum.RSIIndicator(df["close"], window=window).rsi()
    return df

def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    MACD line, signal line, dan histogram.
    """
    df = df.copy()
    macd_ind = ta.trend.MACD(df["close"], window_fast=fast, window_slow=slow, window_sign=signal)
    df["macd_line"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_histogram"] = macd_ind.macd_diff()
    return df

def add_bollinger(df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Band features sebagai proxy volatilitas.
    """
    df = df.copy()
    bb_ind = ta.volatility.BollingerBands(df["close"], window=window, window_dev=std)
    upper = bb_ind.bollinger_hband()
    lower = bb_ind.bollinger_lband()
    middle = bb_ind.bollinger_mavg()
    df["bb_width"] = (upper - lower) / middle.replace(0, 1e-10)
    df["bb_pct"] = (df["close"] - lower) / (upper - lower).replace(0, 1e-10)
    return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR sebagai proxy volatilitas absolute.
    """
    df = df.copy()
    atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=window)
    df[f"atr_{window}"] = atr_ind.average_true_range()
    df["atr_ratio"] = df[f"atr_{window}"] / df["close"].replace(0, 1e-10)
    return df

def add_ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    """
    EMA crossover sebagai indikator tren.
    """
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=fast)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=slow)
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / df["close"].replace(0, 1e-10)
    df["ema_trend"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    return df

def add_volume_features(df: pd.DataFrame, ma_window: int = 20) -> pd.DataFrame:
    """
    Fitur berbasis volume sebagai proxy minat pasar.
    """
    df = df.copy()
    df["vol_ma_ratio"] = df["volume"] / df["volume"].rolling(ma_window).mean().replace(0, 1e-10)
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, 1e-10)
    df["taker_sell_ratio"] = 1.0 - df["taker_buy_ratio"]
    df["vol_log"] = np.log(df["volume"].replace(0, 1e-10) + 1e-5)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pola likuiditas berbeda per sesi trading.
    """
    df = df.copy()
    hours = df.index.hour
    dows = df.index.dayofweek
    
    df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dows / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dows / 7.0)
    df["is_weekend"] = (dows >= 5).astype(int)
    
    # Sesi trading UTC
    df["session_asia"] = ((hours >= 0) & (hours < 8)).astype(int)
    df["session_europe"] = ((hours >= 8) & (hours < 16)).astype(int)
    df["session_us"] = ((hours >= 13) & (hours < 21)).astype(int)
    return df

def add_realized_vol(df: pd.DataFrame, windows: list = [5, 12, 48]) -> pd.DataFrame:
    """
    Volatilitas realized: std dari log returns dalam N candle terakhir.
    """
    df = df.copy()
    # Pastikan ret_1 ada
    if "ret_1" not in df.columns:
        df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
        
    for w in windows:
        df[f"rvol_{w}"] = df["ret_1"].rolling(w).std() * np.sqrt(w)
    return df

def add_order_book_imbalance(
    df: pd.DataFrame,
    orderbook_df: pd.DataFrame,
    levels: list = [1, 3, 5, 10]
) -> pd.DataFrame:
    """
    OBI = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
    """
    df = df.copy()
    if orderbook_df is None or orderbook_df.empty:
        return df
        
    # Match orderbook snapshot ke kline berdasarkan timestamp (backward merge_asof)
    merged = pd.merge_asof(df, orderbook_df, left_index=True, right_index=True, direction="backward")
    
    for L in levels:
        bid_qty_cols = [f"bid_qty_{i}" for i in range(1, L + 1)]
        ask_qty_cols = [f"ask_qty_{i}" for i in range(1, L + 1)]
        
        # Check if columns exist in merged
        missing_cols = [c for c in bid_qty_cols + ask_qty_cols if c not in merged.columns]
        if missing_cols:
            logger.warning(f"Missing order book level columns: {missing_cols}")
            continue
            
        sum_bid = merged[bid_qty_cols].sum(axis=1)
        sum_ask = merged[ask_qty_cols].sum(axis=1)
        df[f"obi_{L}"] = (sum_bid - sum_ask) / (sum_bid + sum_ask).replace(0, 1e-10)
        
    return df

def add_micro_price(df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Micro-price / VAMP (volume-adjusted mid price)
    """
    df = df.copy()
    if orderbook_df is None or orderbook_df.empty:
        return df
        
    merged = pd.merge_asof(df, orderbook_df, left_index=True, right_index=True, direction="backward")
    
    if "bid_price_1" not in merged.columns or "bid_qty_1" not in merged.columns:
        return df
        
    best_bid = merged["bid_price_1"]
    best_bid_qty = merged["bid_qty_1"]
    best_ask = merged["ask_price_1"]
    best_ask_qty = merged["ask_qty_1"]
    
    mid_price = (best_bid + best_ask) / 2.0
    micro_price = (best_ask * best_bid_qty + best_bid * best_ask_qty) / (best_bid_qty + best_ask_qty).replace(0, 1e-10)
    
    df["micro_price"] = micro_price
    df["micro_price_dev"] = (micro_price - mid_price) / mid_price.replace(0, 1e-10)
    return df

def add_spread_features(df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spread relatif sebagai proxy biaya transaksi dan likuiditas.
    """
    df = df.copy()
    if orderbook_df is None or orderbook_df.empty:
        return df
        
    merged = pd.merge_asof(df, orderbook_df, left_index=True, right_index=True, direction="backward")
    
    if "bid_price_1" not in merged.columns:
        return df
        
    best_bid = merged["bid_price_1"]
    best_ask = merged["ask_price_1"]
    mid_price = (best_bid + best_ask) / 2.0
    
    df["spread_abs"] = best_ask - best_bid
    df["spread_rel"] = df["spread_abs"] / mid_price.replace(0, 1e-10)
    return df

def add_trade_flow_imbalance(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    windows_min: list = [1, 5, 15]
) -> pd.DataFrame:
    """
    Trade flow imbalance dari stream aggTrade.
    """
    df = df.copy()
    if trades_df is None or trades_df.empty:
        return df
        
    trades = trades_df.copy()
    trades["buy_vol"] = np.where(trades["is_buyer_maker"] == False, trades["quantity"], 0.0)
    trades["sell_vol"] = np.where(trades["is_buyer_maker"] == True, trades["quantity"], 0.0)
    
    resampled = trades[["buy_vol", "sell_vol"]].resample("1min").sum().fillna(0.0)
    
    for w in windows_min:
        rolling_buy = resampled["buy_vol"].rolling(f"{w}min", closed="right").sum()
        rolling_sell = resampled["sell_vol"].rolling(f"{w}min", closed="right").sum()
        tfi = rolling_buy / (rolling_buy + rolling_sell).replace(0, 1e-10)
        
        temp_df = pd.DataFrame({f"tfi_{w}m": tfi}, index=resampled.index)
        
        # Merge backward ke df
        df = pd.merge_asof(df, temp_df, left_index=True, right_index=True, direction="backward")
        
    return df

def add_depth_features(df: pd.DataFrame, orderbook_df: pd.DataFrame) -> pd.DataFrame:
    """
    Total likuiditas di beberapa level.
    """
    df = df.copy()
    if orderbook_df is None or orderbook_df.empty:
        return df
        
    merged = pd.merge_asof(df, orderbook_df, left_index=True, right_index=True, direction="backward")
    
    bid_cols_5 = [f"bid_qty_{i}" for i in range(1, 6)]
    ask_cols_5 = [f"ask_qty_{i}" for i in range(1, 6)]
    bid_cols_10 = [f"bid_qty_{i}" for i in range(1, 11)]
    ask_cols_10 = [f"ask_qty_{i}" for i in range(1, 11)]
    
    # Check if columns exist
    missing = [c for c in bid_cols_10 + ask_cols_10 if c not in merged.columns]
    if missing:
        return df
        
    df["depth_bid_5"] = merged[bid_cols_5].sum(axis=1)
    df["depth_ask_5"] = merged[ask_cols_5].sum(axis=1)
    df["depth_bid_10"] = merged[bid_cols_10].sum(axis=1)
    df["depth_ask_10"] = merged[ask_cols_10].sum(axis=1)
    df["depth_ratio_5"] = df["depth_bid_5"] / df["depth_ask_5"].replace(0, 1e-10)
    return df

def add_htf_context(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame
) -> pd.DataFrame:
    """
    Tambahkan konteks tren dari timeframe lebih besar (1h).
    """
    df_5m = df_5m.copy()
    if df_1h is None or df_1h.empty:
        return df_5m
        
    df_1h_features = pd.DataFrame(index=df_1h.index)
    df_1h_features["htf_ret_1h"] = np.log(df_1h["close"] / df_1h["close"].shift(1))
    df_1h_features["htf_ret_4h"] = np.log(df_1h["close"] / df_1h["close"].shift(4))
    df_1h_features["htf_rsi_1h"] = ta.momentum.RSIIndicator(df_1h["close"], window=14).rsi()
    
    ema9 = ta.trend.ema_indicator(df_1h["close"], window=9)
    ema21 = ta.trend.ema_indicator(df_1h["close"], window=21)
    df_1h_features["htf_ema_trend"] = (ema9 > ema21).astype(int)
    
    # Shift 1 candle demi keutuhan anti-lookahead
    df_1h_safe = df_1h_features.shift(1)
    
    merged = pd.merge_asof(df_5m, df_1h_safe, left_index=True, right_index=True, direction="backward")
    return merged

def build_all_features(
    df_5m: pd.DataFrame,
    df_1m: pd.DataFrame = None,
    df_1h: pd.DataFrame = None,
    orderbook_df: pd.DataFrame = None,
    trades_df: pd.DataFrame = None,
    config: dict = None
) -> pd.DataFrame:
    """
    Orchestrator pipeline feature engineering.
    """
    if config is None:
        config = {}
        
    df = df_5m.copy()
    
    # Kelompok A
    df = add_log_returns(df)
    df = add_rsi(df, window=config.get("rsi_period", 14))
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df, window=config.get("atr_period", 14))
    df = add_ema_cross(df, fast=config.get("ema_fast", 9), slow=config.get("ema_slow", 21))
    df = add_volume_features(df, ma_window=config.get("vol_ma_period", 20))
    df = add_time_features(df)
    df = add_realized_vol(df)
    
    # Kelompok C
    if df_1h is not None:
        df = add_htf_context(df, df_1h)
        
    # Kelompok B
    if orderbook_df is not None:
        obi_levels = config.get("obi_levels", [1, 3, 5, 10])
        df = add_order_book_imbalance(df, orderbook_df, levels=obi_levels)
        df = add_micro_price(df, orderbook_df)
        df = add_spread_features(df, orderbook_df)
        df = add_depth_features(df, orderbook_df)
        
    if trades_df is not None:
        df = add_trade_flow_imbalance(df, trades_df)
        
    initial_rows = len(df)
    
    # Log persentase NaN per kolom sebelum dropna
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        nan_pct = (nan_counts / initial_rows * 100).round(2)
        cols_with_nan = nan_pct[nan_pct > 0].sort_values(ascending=False)
        logger.warning(f"Detected NaN in features before dropna (out of {initial_rows} rows):\n{cols_with_nan.to_string()}")
        
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    logger.info(f"Feature build complete: {initial_rows} → {len(df)} rows after dropna (dropped {dropped_rows} rows, {dropped_rows/initial_rows*100:.2f}%)")
    
    return df
