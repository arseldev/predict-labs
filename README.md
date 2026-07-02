# BTC 5-Minute Direction Predictor

Sistem prediksi arah harga Bitcoin (Up/Down) pada timeframe 5 menit menggunakan ML (LightGBM/XGBoost + fitur microstructure order book).

> ⚠️ **Peringatan:** Dokumen ini adalah panduan teknis, bukan rekomendasi investasi. Uji menyeluruh di testnet sebelum menggunakan dana riil.

---

## Struktur Fase

| Fase | File Implementasi | Status |
|---|---|---|
| **Phase 0** — Setup & Skeleton | [phase-0-setup.md](architecture/phase-0-setup.md) | 📋 Ready |
| **Phase 1** — Data Pipeline | [phase-1-data-pipeline.md](architecture/phase-1-data-pipeline.md) | 📋 Ready |
| **Phase 2** — Feature Engineering | [phase-2-features.md](architecture/phase-2-features.md) | 📋 Ready |
| **Phase 3** — Labeling | [phase-3-labeling.md](architecture/phase-3-labeling.md) | 📋 Ready |
| **Phase 4** — Model & Validasi | [phase-4-model-validation.md](architecture/phase-4-model-validation.md) | 📋 Ready |
| **Phase 5** — Backtest | [phase-5-backtest.md](architecture/phase-5-backtest.md) | 📋 Ready |
| **Phase 6** — Paper Trading | [phase-6-paper-trading.md](architecture/phase-6-paper-trading.md) | 📋 Ready |
| **Phase 7** — Live Trading | [phase-7-live.md](architecture/phase-7-live.md) | 📋 Ready |

---

## Quick Start

```bash
# 1. Clone & setup environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Konfigurasi API key
cp .env.example .env
# Edit .env dengan API key Binance kamu

# 4. Mulai dari Phase 0 — baca architecture/phase-0-setup.md
```

---

## Anti-Pattern Checklist (Tempel di Monitor)

- [ ] **Tidak ada lookahead bias** — fitur di t hanya pakai data sebelum/di t
- [ ] **Validasi walk-forward** — BUKAN random split atau k-fold biasa
- [ ] **Fee + slippage selalu dihitung** — akurasi tanpa EV tidak berarti
- [ ] **Triple-barrier label di-purge** — atau dicatat sebagai known limitation
- [ ] **Model diretrain berkala** — data finansial non-stasioner
- [ ] **Akurasi ≠ profitabilitas** — selalu cek Expected Value

---

## Target Realistis

Berdasarkan literatur akademik:
- **Akurasi directional:** 52–60% (bukan 90%+, itu overfitting)
- **EV per trade:** > 0 setelah fee 0.1% dan slippage 0.02%
- **Sharpe Ratio:** > 0.5 (target > 1.0)

---

## Struktur Project

```
btc-5m-prediction/
├── config/config.yaml       # Konfigurasi terpusat
├── data/
│   ├── raw/                 # Data mentah (kline, orderbook, trades)
│   └── processed/           # Fitur + label siap training
├── src/
│   ├── data_fetch.py        # Fase 1: Historical data
│   ├── data_stream.py       # Fase 1: Live WebSocket stream
│   ├── features.py          # Fase 2: Feature engineering
│   ├── labeling.py          # Fase 3: Triple-barrier labeling
│   ├── validation.py        # Fase 4: Walk-forward validation
│   ├── models.py            # Fase 4: LightGBM/XGBoost/LSTM
│   ├── backtest.py          # Fase 5: Backtesting engine
│   ├── evaluate.py          # Fase 5: Metrik EV, Sharpe, drawdown
│   ├── live_predict.py      # Fase 6: Real-time prediction loop
│   └── executor.py          # Fase 6: Order execution (testnet/live)
├── notebooks/               # EDA dan analisis
├── tests/                   # Unit tests per modul
├── logs/                    # predictions.db + system.log
└── architecture/            # Dokumen implementasi per fase
```
