# Phase 0 — Project Setup & Skeleton

> **Tujuan:** Siapkan seluruh struktur project, environment, konfigurasi, dan kerangka file kosong (dengan docstring) sebelum menulis logic apapun.

---

## 0.1 Prasyarat Sistem

| Kebutuhan | Versi Minimum | Catatan |
|---|---|---|
| Python | 3.10+ | Gunakan `pyenv` atau `conda` untuk isolasi versi |
| pip / uv | terbaru | `uv` direkomendasikan — jauh lebih cepat dari pip |
| Git | 2.x | Version control wajib |
| SQLite | bawaan Python | Untuk storage lokal development |
| RAM | 8 GB+ | Untuk proses data parquet + model training |
| Storage | 10 GB+ | Raw data historis bisa besar |

---

## 0.2 Struktur Folder Lengkap

Buat semua folder dan file berikut (file boleh kosong dulu, isi sesuai fase):

```
btc-5m-prediction/
├── config/
│   └── config.yaml                  # Semua konfigurasi terpusat
├── data/
│   ├── raw/
│   │   ├── klines/                  # Kline data per timeframe
│   │   │   ├── 1m/
│   │   │   ├── 5m/
│   │   │   ├── 15m/
│   │   │   └── 1h/
│   │   ├── orderbook/               # Snapshot depth per waktu
│   │   └── trades/                  # AggTrade data
│   └── processed/
│       ├── features/                # Dataset fitur siap latih (.parquet)
│       └── labels/                  # Label hasil fixed-horizon & triple-barrier
├── src/
│   ├── __init__.py
│   ├── data_fetch.py                # Historical data fetcher
│   ├── data_stream.py               # WebSocket live stream manager
│   ├── features.py                  # Feature engineering pipeline
│   ├── labeling.py                  # Fixed-horizon & triple-barrier labeling
│   ├── validation.py                # Walk-forward & purged K-fold CV
│   ├── models.py                    # Model training (LGB/XGB/LSTM)
│   ├── backtest.py                  # Backtesting engine dengan fee & slippage
│   ├── evaluate.py                  # Metrik: EV, Sharpe, drawdown, precision/recall
│   ├── live_predict.py              # Real-time prediction loop
│   └── executor.py                  # Order execution (testnet & live)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA dan sanity check data
│   ├── 02_feature_analysis.ipynb    # Analisis feature importance & SHAP
│   ├── 03_model_comparison.ipynb    # Perbandingan model walk-forward
│   └── 04_backtest_analysis.ipynb   # Analisis hasil backtest lengkap
├── tests/
│   ├── __init__.py
│   ├── test_data_fetch.py
│   ├── test_features.py             # KRITIS: test anti-lookahead bias
│   ├── test_labeling.py
│   ├── test_validation.py
│   └── test_backtest.py
├── logs/
│   ├── predictions.db               # SQLite: histori prediksi vs aktual
│   └── system.log                   # Application log
├── architecture/                    # Sudah ada
├── .env.example                     # Template API key (jangan commit .env asli!)
├── .gitignore
├── requirements.txt
├── requirements-dev.txt             # Pytest, jupyter, dll
└── README.md
```

---

## 0.3 Langkah Setup Environment

### Step 1: Buat Virtual Environment

```bash
# Gunakan venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### Step 2: Buat `requirements.txt`

```txt
# Data fetching
python-binance==1.0.19
unicorn-binance-websocket-api>=2.6.0
requests>=2.31.0

# Data processing
pandas>=2.1.0
numpy>=1.26.0
pyarrow>=14.0.0
fastparquet>=2023.10.0

# Technical indicators
ta>=0.11.0
pandas-ta>=0.3.14b

# Machine Learning
scikit-learn>=1.3.0
lightgbm>=4.1.0
xgboost>=2.0.0
shap>=0.43.0

# Deep Learning (opsional, fase lanjut)
torch>=2.1.0

# Backtesting
vectorbt>=0.25.0

# Database & Config
sqlalchemy>=2.0.0
pyyaml>=6.0.1
python-dotenv>=1.0.0

# Monitoring & Utilities
loguru>=0.7.0
tqdm>=4.66.0
schedule>=1.2.0
scipy>=1.11.0
```

```txt
# requirements-dev.txt
pytest>=7.4.0
pytest-cov>=4.1.0
jupyter>=1.0.0
ipykernel>=6.26.0
black>=23.10.0
isort>=5.12.0
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 0.4 Konfigurasi (`config/config.yaml`)

```yaml
# ============================================
# BTC 5-Minute Prediction System Configuration
# ============================================

binance:
  symbol: "BTCUSDT"
  testnet: true                     # WAJIB true sampai Fase 6 selesai
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}

data:
  timeframes:
    primary: "5m"
    context: ["1m", "15m", "1h"]
  raw_path: "data/raw"
  processed_path: "data/processed"
  parquet_compression: "snappy"
  orderbook:
    depth_levels: 20
    snapshot_interval_sec: 30
  history:
    days_back: 90

features:
  lookback_candles: 100
  rsi_period: 14
  atr_period: 14
  ema_fast: 9
  ema_slow: 21
  vol_ma_period: 20
  obi_levels: [1, 3, 5, 10]

labeling:
  method: "triple_barrier"
  fixed_horizon:
    n_candles_ahead: 1
  triple_barrier:
    profit_pct: 0.0015
    loss_pct: 0.0015
    max_candles: 6

model:
  primary: "lightgbm"
  lightgbm:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    num_leaves: 63
    min_child_samples: 50
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42

validation:
  method: "walk_forward"
  walk_forward:
    n_splits: 10
    train_period_days: 45
    test_period_days: 7
    embargo_periods: 12

trading:
  fee_taker: 0.001
  fee_maker: 0.001
  slippage_pct: 0.0002
  probability_threshold: 0.60
  position_size_pct: 0.02
  max_daily_loss_pct: 0.03
  max_weekly_loss_pct: 0.08

logging:
  db_path: "logs/predictions.db"
  log_path: "logs/system.log"
  log_level: "INFO"
```

---

## 0.5 File `.env.example`

```bash
# Salin ke .env dan isi dengan API key asli
# JANGAN commit file .env ke Git!

BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here
```

---

## 0.6 Checklist Selesai Phase 0

- [ ] Folder structure dibuat sesuai 0.2
- [ ] Virtual environment aktif dan dependencies terinstall
- [ ] `config/config.yaml` terisi dengan nilai default
- [ ] `.env.example` dibuat, `.env` sudah diisi API key (tidak di-commit)
- [ ] `.gitignore` diperbarui
- [ ] `README.md` dibuat
- [ ] Semua file `src/*.py` ada (meski kosong dengan docstring)
- [ ] Semua file `tests/*.py` ada
- [ ] Git repository diinisialisasi, initial commit dilakukan

**→ Lanjut ke [Phase 1 — Data Pipeline](./phase-1-data-pipeline.md)**
