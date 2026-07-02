# Phase 5 — Backtest dengan Biaya Realistis

> **Tujuan:** Simulasikan performa trading sistem secara realistis — bukan hanya akurasi klasifikasi, tapi **Expected Value (EV) setelah fee dan slippage**. Ini adalah gatekeeper sebelum paper trading.

---

## 5.1 Mengapa Akurasi Saja Tidak Cukup

Contoh yang membukakan pikiran:

| Skenario | Win Rate | Avg Profit | Avg Loss | Fee (RT) | Slippage | EV per trade |
|---|---|---|---|---|---|---|
| A: "Akurasi Tinggi" | 58% | 0.08% | 0.12% | 0.10% | 0.02% | **-0.024%** ❌ |
| B: "Akurasi Sedang" | 54% | 0.20% | 0.10% | 0.10% | 0.02% | **+0.022%** ✅ |
| C: "Akurasi Rendah" | 51% | 0.30% | 0.15% | 0.10% | 0.02% | **+0.005%** ✅ |

**Skenario A rugi meskipun akurasi 58%** karena profit tiap trade lebih kecil dari biaya transaksi.
EV adalah metrik yang benar.

---

## 5.2 Arsitektur Backtesting Engine

```
┌─────────────────────────────────────────────────────────────┐
│                   backtest.py                               │
│                                                             │
│  HistoricalData → SignalGenerator → TradeSimulator → Report │
│                                                             │
│  SignalGenerator:                                           │
│    - Load model (dari Phase 4)                              │
│    - Loop candle by candle (tidak boleh lookahead!)         │
│    - predict_proba → cek threshold → generate signal        │
│                                                             │
│  TradeSimulator:                                            │
│    - Apply fee taker/maker                                  │
│    - Apply slippage (estimasi dari spread historis)         │
│    - Position sizing (fixed fraction atau Kelly)            │
│    - Stop loss & take profit                                │
│    - Max daily/weekly drawdown check                        │
│                                                             │
│  Report:                                                    │
│    - EV, Sharpe, max drawdown, win rate, RR ratio           │
│    - Equity curve plot                                      │
│    - Per-regime breakdown                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5.3 `src/backtest.py` — Implementasi

### Konfigurasi Backtest

```python
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0     # USDT
    fee_taker: float = 0.001             # 0.10%
    fee_maker: float = 0.001             # 0.10% (gunakan limit order untuk 0.08%)
    slippage_pct: float = 0.0002         # 0.02% estimasi slippage
    probability_threshold: float = 0.60  # Min P untuk eksekusi
    position_size_pct: float = 0.02      # 2% dari kapital per trade
    max_daily_loss_pct: float = 0.03     # Stop jika rugi >3% dalam sehari
    max_weekly_loss_pct: float = 0.08    # Stop jika rugi >8% dalam seminggu
    use_triple_barrier: bool = True      # Gunakan TP/SL dari triple-barrier config
    profit_target_pct: float = 0.0015   # TP: +0.15%
    stop_loss_pct: float = 0.0015       # SL: -0.15%
    max_hold_candles: int = 6           # Maksimum hold 6 candle = 30 menit
```

### Trade Record

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    direction: str                  # 'long' atau 'short'
    position_size_usdt: float       # Ukuran posisi dalam USDT
    position_size_btc: float        # Ukuran posisi dalam BTC
    predicted_proba: float          # Probabilitas dari model
    
    # Hasil
    gross_pnl: float = 0.0         # PnL sebelum fee
    fee_paid: float = 0.0          # Total fee (entry + exit)
    slippage_cost: float = 0.0     # Slippage cost
    net_pnl: float = 0.0          # PnL bersih = gross_pnl - fee - slippage
    exit_reason: str = ""          # "tp", "sl", "timeout", "eod"
    
    @property
    def net_pnl_pct(self) -> float:
        if self.position_size_usdt == 0:
            return 0.0
        return self.net_pnl / self.position_size_usdt
```

### Engine Utama

```python
def run_backtest(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    config: BacktestConfig
) -> tuple[list[TradeRecord], pd.DataFrame]:
    """
    Loop backtest candle-by-candle.
    
    PENTING — Anti-lookahead dalam backtest:
    - Prediksi dibuat setelah candle T CLOSED
    - Entry dilakukan pada OPEN candle T+1 (bukan close T)
    - Karena di live trading, kita tidak bisa masuk tepat di harga close candle
    
    Flow per candle:
    1. Candle T closed → hitung fitur → predict_proba
    2. Jika proba > threshold → queue order untuk masuk di open candle T+1
    3. Candle T+1 buka → eksekusi entry di harga open (dengan slippage)
    4. Dalam candle T+1 hingga T+6:
       a. Cek apakah high >= TP barrier → exit dengan untung
       b. Cek apakah low <= SL barrier → exit dengan rugi
       c. Jika candle T+max_hold → exit di close (timeout)
    5. Hitung PnL bersih (gross - fee - slippage)
    
    Args:
        df: DataFrame dengan fitur + OHLCV (DatetimeIndex, sorted ascending)
        model: trained model dengan predict_proba()
        feature_cols: nama kolom fitur
        config: BacktestConfig
    
    Returns:
        (trade_records, equity_curve_df)
    """
    trades = []
    capital = config.initial_capital
    equity_curve = []
    
    active_trade = None
    daily_pnl = {}
    weekly_pnl = {}
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        timestamp = df.index[i]
        
        # --- KILL SWITCH CHECK ---
        today = timestamp.date()
        if daily_pnl.get(today, 0) < -config.initial_capital * config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit hit on {today}. Skipping rest of day.")
            continue
        
        # --- EXIT CHECK (jika ada posisi aktif) ---
        if active_trade is not None:
            trade, capital = _check_exit(active_trade, row, df, i, capital, config)
            if trade.exit_time is not None:
                trades.append(trade)
                daily_pnl[today] = daily_pnl.get(today, 0) + trade.net_pnl
                active_trade = None
        
        # --- SIGNAL GENERATION ---
        if active_trade is None:  # Tidak trading jika sudah ada posisi
            features = df[feature_cols].iloc[i:i+1]
            proba = model.predict_proba(features)[0, 1]  # P(naik)
            
            if proba > config.probability_threshold:
                # Entry di OPEN candle berikutnya + slippage
                entry_price = next_row["open"] * (1 + config.slippage_pct)
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
                capital -= entry_fee
        
        # Log equity
        current_equity = capital
        if active_trade:
            # Mark-to-market: nilai posisi terkini
            current_equity += active_trade.position_size_btc * row["close"] - \
                              active_trade.position_size_btc * active_trade.entry_price
        equity_curve.append({"timestamp": timestamp, "equity": current_equity})
    
    equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
    return trades, equity_df


def _check_exit(
    trade: TradeRecord,
    current_row: pd.Series,
    df: pd.DataFrame,
    current_idx: int,
    capital: float,
    config: BacktestConfig
) -> tuple[TradeRecord, float]:
    """
    Cek kondisi exit untuk posisi aktif.
    
    Urutan prioritas:
    1. Stop loss (low >= barrier_bot)
    2. Take profit (high >= barrier_top)
    3. Timeout (sudah hold max_hold_candles)
    4. End of data
    
    CATATAN: Jika high dan low sama-sama melebihi barrier dalam satu candle,
    ini "ambiguous" — asumsikan SL kena duluan (konservatif).
    """
    tp_price = trade.entry_price * (1 + config.profit_target_pct)
    sl_price = trade.entry_price * (1 - config.stop_loss_pct)
    
    candles_held = current_idx - df.index.get_loc(trade.entry_time)
    
    exit_price = None
    exit_reason = None
    
    # Cek SL dulu (konservatif)
    if current_row["low"] <= sl_price:
        exit_price = sl_price * (1 - config.slippage_pct)  # slippage saat exit
        exit_reason = "sl"
    elif current_row["high"] >= tp_price:
        exit_price = tp_price * (1 - config.slippage_pct)
        exit_reason = "tp"
    elif candles_held >= config.max_hold_candles:
        exit_price = current_row["close"] * (1 - config.slippage_pct)
        exit_reason = "timeout"
    
    if exit_price is not None:
        exit_fee = trade.position_size_btc * exit_price * config.fee_taker
        gross_pnl = trade.position_size_btc * (exit_price - trade.entry_price)
        net_pnl = gross_pnl - trade.fee_paid - exit_fee - trade.slippage_cost
        
        trade.exit_time = df.index[current_idx]
        trade.exit_price = exit_price
        trade.gross_pnl = gross_pnl
        trade.fee_paid += exit_fee
        trade.slippage_cost += trade.position_size_btc * exit_price * config.slippage_pct
        trade.net_pnl = net_pnl
        trade.exit_reason = exit_reason
        
        capital += trade.position_size_usdt + net_pnl
    
    return trade, capital
```

---

## 5.4 `src/evaluate.py` — Metrik Backtest

```python
def compute_backtest_metrics(
    trades: list[TradeRecord],
    equity_curve: pd.DataFrame,
    config: BacktestConfig
) -> dict:
    """
    Hitung semua metrik backtest yang relevan.
    
    Metrik wajib:
    1. Expected Value (EV) per trade
    2. Win rate
    3. Average profit vs average loss (Risk-Reward Ratio)
    4. Sharpe Ratio (annualized)
    5. Maximum Drawdown (%)
    6. Total return (%)
    7. Number of trades
    8. Average holding period
    9. Fee & slippage impact (berapa % return yang dimakan biaya)
    
    Breakdowns:
    - Per exit reason (tp vs sl vs timeout)
    - Per market regime
    - Per probability bucket (0.60-0.65, 0.65-0.70, 0.70+)
      → Apakah prob lebih tinggi = profit lebih tinggi? (kalibrasi)
    """
    if not trades:
        return {"error": "No trades generated"}
    
    net_pnls = [t.net_pnl_pct for t in trades]
    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl <= 0]
    
    win_rate = len(winning_trades) / len(trades)
    avg_win = np.mean([t.net_pnl_pct for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t.net_pnl_pct for t in losing_trades])) if losing_trades else 0
    
    # Expected Value
    ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Fee impact
    total_fee = sum(t.fee_paid for t in trades)
    total_slippage = sum(t.slippage_cost for t in trades)
    total_gross_pnl = sum(t.gross_pnl for t in trades)
    
    # Sharpe Ratio (dari equity curve)
    equity_returns = equity_curve["equity"].pct_change().dropna()
    if equity_returns.std() > 0:
        # Annualize: 288 candle 5m per hari × 365 hari
        sharpe = (equity_returns.mean() / equity_returns.std()) * np.sqrt(288 * 365)
    else:
        sharpe = 0.0
    
    # Maximum Drawdown
    rolling_max = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Total return
    total_return = (equity_curve["equity"].iloc[-1] - equity_curve["equity"].iloc[0]) / \
                   equity_curve["equity"].iloc[0]
    
    # Risk-Reward Ratio
    rr_ratio = avg_win / max(avg_loss, 0.0001)
    
    return {
        # Core metrics
        "ev_per_trade": ev,
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "risk_reward_ratio": rr_ratio,
        
        # Performance
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown,
        
        # Trade statistics
        "n_trades": len(trades),
        "n_winning": len(winning_trades),
        "n_losing": len(losing_trades),
        "avg_holding_candles": np.mean([
            (t.exit_time - t.entry_time).total_seconds() / 300
            for t in trades if t.exit_time
        ]),
        
        # Cost analysis
        "total_fee_paid": total_fee,
        "total_slippage": total_slippage,
        "fee_as_pct_gross_pnl": total_fee / max(abs(total_gross_pnl), 1e-10),
        
        # Exit breakdown
        "exit_tp_pct": len([t for t in trades if t.exit_reason == "tp"]) / len(trades),
        "exit_sl_pct": len([t for t in trades if t.exit_reason == "sl"]) / len(trades),
        "exit_timeout_pct": len([t for t in trades if t.exit_reason == "timeout"]) / len(trades),
        
        # Probability calibration (apakah prob lebih tinggi = hasil lebih baik?)
        "prob_buckets": _analyze_prob_buckets(trades)
    }

def _analyze_prob_buckets(trades: list[TradeRecord]) -> dict:
    """
    Analisis: apakah probabilitas model terkalibrasi dengan baik?
    Model yang baik: prob tinggi = win rate tinggi.
    """
    buckets = {"0.60-0.65": [], "0.65-0.70": [], "0.70-0.75": [], "0.75+": []}
    
    for t in trades:
        p = t.predicted_proba
        bucket = ("0.60-0.65" if p < 0.65 else
                  "0.65-0.70" if p < 0.70 else
                  "0.70-0.75" if p < 0.75 else "0.75+")
        buckets[bucket].append(t.net_pnl > 0)
    
    return {
        bucket: {
            "n_trades": len(results),
            "win_rate": np.mean(results) if results else None
        }
        for bucket, results in buckets.items()
    }
```

---

## 5.5 Format Laporan Backtest

```
============================================================
BACKTEST REPORT
Period: 2024-01-01 → 2024-06-30 | Model: LightGBM
Fee: 0.10% RT | Slippage: 0.02% | Threshold: P > 0.60
============================================================

CORE METRICS:
  Expected Value per trade:   +0.023%  ✅ (positif = lanjut ke testnet)
  Win Rate:                    54.2%
  Avg Win:                     +0.187%
  Avg Loss:                    -0.132%
  Risk-Reward Ratio:           1.42x

PERFORMANCE:
  Total Return:               +18.4%
  Sharpe Ratio:                1.23
  Max Drawdown:               -8.7%
  Total Trades:               1,247

COST ANALYSIS:
  Total Fee Paid:             $234.50
  Total Slippage:             $46.90
  Fee as % of Gross PnL:      31.2%  ← Penting! Fee makan 31% gross profit

EXIT BREAKDOWN:
  Take Profit:                 38.2%
  Stop Loss:                   41.6%
  Timeout:                     20.2%

PROBABILITY CALIBRATION:
  P 0.60-0.65:  n=523, win_rate=52.1%
  P 0.65-0.70:  n=389, win_rate=55.3%
  P 0.70-0.75:  n=221, win_rate=58.9%  ✅ Higher prob → Higher win rate
  P 0.75+:      n=114, win_rate=61.4%  ✅ Model well-calibrated

KESIMPULAN: EV > 0 → LOLOS ke Phase 6 (Paper Trading)
============================================================
```

---

## 5.6 Sensitivity Analysis

Setelah backtest dasar selesai, jalankan sensitivity analysis:

```python
def sensitivity_analysis(df, model, feature_cols):
    """
    Test sensitivitas terhadap asumsi biaya.
    Jika EV negatif saat fee dinaikkan sedikit, sistem tidak robust.
    """
    results = []
    
    # Variasikan fee dan threshold
    for fee in [0.0005, 0.001, 0.0015, 0.002]:  # 0.05% - 0.20%
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            config = BacktestConfig(fee_taker=fee, probability_threshold=threshold)
            trades, equity = run_backtest(df, model, feature_cols, config)
            metrics = compute_backtest_metrics(trades, equity, config)
            results.append({
                "fee": fee,
                "threshold": threshold,
                "ev": metrics["ev_per_trade"],
                "sharpe": metrics["sharpe_ratio"],
                "n_trades": metrics["n_trades"]
            })
    
    return pd.DataFrame(results)
```

Buat heatmap EV vs fee vs threshold untuk melihat "zona aman" sistem.

---

## 5.7 Kriteria Lolos ke Phase 6 (Paper Trading)

```
MUST PASS (wajib semua):
  ✅ EV > 0 secara konsisten di semua fold walk-forward backtest
  ✅ Sharpe Ratio >= 0.5
  ✅ Max Drawdown <= 15%
  ✅ Win rate > 50%
  ✅ EV masih positif bahkan saat fee dinaikkan 50% (robustness)
  ✅ Probabilitas lebih tinggi → win rate lebih tinggi (model terkalibrasi)

NICE TO HAVE:
  ⭕ Sharpe >= 1.0
  ⭕ EV > 0.02% per trade
  ⭕ Max drawdown <= 10%
  ⭕ Konsisten di semua market regime (bullish, bearish, sideways)

RED FLAGS — STOP, jangan ke Phase 6:
  ❌ EV negatif di mayoritas fold
  ❌ Sharpe < 0
  ❌ Drawdown > 25%
  ❌ Hasil sangat berbeda per market regime (model fragile)
  ❌ Fee makan > 70% gross profit (terlalu sensitif terhadap biaya)
```

---

## 5.8 Unit Tests (`tests/test_backtest.py`)

```python
class TestBacktest:
    def test_no_future_data_in_signal(self):
        """Sinyal hanya boleh berdasarkan data yang sudah tersedia."""
        # Entry di open candle T+1, bukan close candle T
        df = make_dummy_kline_with_features(100)
        model = MockModel(always_predict_proba=0.7)
        config = BacktestConfig(probability_threshold=0.60)
        trades, _ = run_backtest(df, model, FEATURE_COLS, config)
        
        for trade in trades:
            # Entry time harus SETELAH sinyal di-generate
            signal_bar_idx = df.index.get_loc(trade.entry_time) - 1
            assert signal_bar_idx >= 0, "Entry sebelum sinyal di-generate!"
    
    def test_fee_applied(self):
        """Fee harus selalu dikurangi dari PnL."""
        df = make_dummy_kline_with_features(50)
        model = MockModel(always_predict_proba=0.9)
        config = BacktestConfig(fee_taker=0.001, slippage_pct=0.0)
        trades, _ = run_backtest(df, model, FEATURE_COLS, config)
        
        for trade in trades:
            # Fee harus positif untuk setiap trade
            assert trade.fee_paid > 0, f"Fee tidak diterapkan pada trade {trade}"
    
    def test_capital_never_negative(self):
        """Kapital tidak boleh negatif."""
        df = make_dummy_kline_with_features(200)
        model = MockModel(always_predict_proba=0.9)
        config = BacktestConfig(initial_capital=1000.0, position_size_pct=0.10)
        _, equity = run_backtest(df, model, FEATURE_COLS, config)
        
        assert (equity["equity"] >= 0).all(), "Kapital negatif — ada bug di PnL calculation"
    
    def test_kill_switch_stops_trading(self):
        """Kill switch harus menghentikan trading saat rugi terlalu besar."""
        # Buat mock model yang selalu salah prediksi
        df = make_dummy_kline_with_features(500)
        model = MockModel(always_predict_proba=0.9)
        # Set SL sangat ketat sehingga selalu loss
        config = BacktestConfig(
            stop_loss_pct=0.0001,  # SL 0.01% — akan sering kena
            max_daily_loss_pct=0.01  # Kill switch 1%
        )
        trades, equity = run_backtest(df, model, FEATURE_COLS, config)
        
        # Tidak boleh ada banyak trade setelah kill switch
        # (tergantung implementasi, cek bahwa trading berhenti setelah loss limit)
        assert len(trades) < 500  # Harus berhenti, tidak trading 500 candle penuh
    
    def test_ev_calculation(self):
        """Verifikasi formula EV benar."""
        # Buat set trade dummy dengan win/loss yang kita tahu
        trades = [
            TradeRecord(net_pnl=10, net_pnl_pct=0.01, ...),  # win
            TradeRecord(net_pnl=-5, net_pnl_pct=-0.005, ...),  # loss
            TradeRecord(net_pnl=10, net_pnl_pct=0.01, ...),  # win
        ]
        # win_rate = 2/3, avg_win = 0.01, avg_loss = 0.005
        # EV = (2/3 * 0.01) - (1/3 * 0.005) = 0.0067 - 0.0017 = 0.0050
        metrics = compute_backtest_metrics(trades, dummy_equity, BacktestConfig())
        assert abs(metrics["ev_per_trade"] - 0.0050) < 0.0001
```

---

## 5.9 Checklist Selesai Phase 5

- [ ] `backtest.py` dengan `run_backtest()` diimplementasikan
- [ ] Entry dilakukan di OPEN candle berikutnya (bukan close candle prediksi)
- [ ] Fee taker, slippage diterapkan untuk setiap trade (entry + exit)
- [ ] Kill-switch (daily loss limit) diimplementasikan
- [ ] `evaluate.py` menghasilkan semua metrik wajib (EV, Sharpe, drawdown)
- [ ] Laporan backtest dihasilkan dalam format yang terstruktur
- [ ] Sensitivity analysis (fee vs threshold) dijalankan
- [ ] Semua unit test di `test_backtest.py` PASS
- [ ] **Kriteria minimum:** EV > 0 di semua fold walk-forward

**→ Lanjut ke [Phase 6 — Paper Trading](./phase-6-paper-trading.md)**
