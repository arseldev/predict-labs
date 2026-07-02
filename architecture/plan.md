# Build Instructions — BTC 5-Menit Direction Predictor (Binance)

> Dokumen ini adalah **spesifikasi teknis** untuk di-feed ke coding assistant (Claude Code, Cursor, dll) supaya proses "vibecoding" tetap terstruktur dan tidak melewatkan bagian kritis (labeling, validasi, biaya transaksi). Ikuti urutan fase — jangan loncat ke Fase 5 (live) sebelum Fase 1–4 selesai dan lolos kriteria.

---

## 0. Prinsip Non-Negotiable

Sebelum coding assistant menulis kode apapun, dia harus paham aturan ini:

1. **Tidak ada lookahead bias.** Fitur di waktu `t` hanya boleh pakai data yang closed sebelum/di `t`. Candle yang belum closed = haram dipakai.
2. **Validasi wajib walk-forward**, bukan random train/test split atau `sklearn` k-fold biasa.
3. **Setiap angka akurasi harus disandingkan dengan Expected Value setelah fee+slippage.** Akurasi tanpa EV tidak berarti apa-apa.
4. **Tidak ada langkah ke live trading** sebelum lolos paper trading di testnet minimal beberapa minggu.
5. Simpan **semua prediksi vs hasil aktual** ke database — ini yang akan membuktikan apakah sistem beneran bekerja atau cuma bagus di backtest.

---

## 1. Struktur Proyek

```
btc-5m-predictor/
├── config/
│   └── config.yaml            # symbol, interval, fee tier, paths
├── data/
│   ├── raw/                   # kline, depth, trade mentah
│   └── processed/             # dataset fitur siap latih
├── src/
│   ├── data_fetch.py          # historical + bulk data.binance.vision
│   ├── data_stream.py         # websocket kline/depth/trade live
│   ├── features.py            # feature engineering (teknikal + microstructure)
│   ├── labeling.py            # fixed-horizon & triple-barrier
│   ├── validation.py          # walk-forward, purged K-fold
│   ├── models.py               # training LightGBM/XGBoost/LSTM
│   ├── backtest.py            # simulasi dengan fee & slippage
│   ├── evaluate.py            # EV, sharpe, drawdown, precision/recall
│   ├── live_predict.py        # loop realtime -> sinyal
│   └── executor.py            # eksekusi order (testnet dulu!)
├── notebooks/                 # eksplorasi & sanity check manual
├── logs/
├── tests/                     # unit test tiap modul
└── requirements.txt
```

Minta coding assistant generate skeleton ini dulu (folder + file kosong dengan docstring tujuan tiap file) sebelum isi logic.

---

## 2. Fase 1 — Data Pipeline

**Tujuan:** kumpulkan kline 5m (+ 1m, 15m, 1h untuk konteks multi-timeframe), order book depth, dan aggTrade.

### Tugas untuk coding assistant:
- [ ] `data_fetch.py`: fungsi ambil historical kline via `python-binance` (`get_historical_klines`), simpan ke `data/raw/` sebagai parquet (bukan CSV, lebih cepat & hemat storage).
- [ ] Untuk histori panjang (>90 hari), gunakan bulk download dari `data.binance.vision` alih-alih REST API looping (hindari rate limit).
- [ ] `data_stream.py`: WebSocket manager (`ThreadedWebsocketManager` atau `unicorn-binance-websocket-api` untuk auto-reconnect) yang subscribe ke:
  - `kline_5m` (dan `kline_1m`, `kline_15m`, `kline_1h` untuk multi-timeframe)
  - `depth` (top 10–20 level)
  - `aggTrade`
- [ ] Semua data live di-append ke storage lokal (SQLite/Postgres/parquet) secara real-time — ini jadi bahan retraining nanti.
- [ ] Order book depth **wajib mulai dikumpulkan dari sekarang** — histori order book gratis biasanya tidak tersedia jauh ke belakang.

### Kriteria selesai:
Minimal 60–90 hari data kline tersimpan bersih, dan stream live berjalan tanpa disconnect >5 menit dalam uji 24 jam.

---

## 3. Fase 2 — Feature Engineering

**File:** `features.py`

### Fitur wajib (dari riset):

**Teknikal (dari OHLCV):**
- `ret_1`, `ret_5`, `ret_10` (log return berbagai lag)
- RSI(14), MACD, Bollinger Band width, ATR
- EMA cross (cepat vs lambat)
- `vol_ma_ratio` = volume / rolling mean volume
- Fitur dari timeframe lebih besar (1h) sebagai konteks tren

**Microstructure (dari depth + aggTrade — ini pembeda akurasi utama menurut riset):**
- Order Book Imbalance: `(vol_bid - vol_ask) / (vol_bid + vol_ask)` di beberapa level
- Micro-price / VAMP (volume-adjusted mid price)
- Spread bid-ask relatif terhadap harga
- Trade flow imbalance: rasio taker buy vs taker sell volume dalam window
- Depth di beberapa level sebagai proxy likuiditas

**Waktu/statistik:**
- Jam UTC, hari dalam minggu (sesi Asia/Eropa/AS punya pola likuiditas beda)
- Realized volatility N-candle terakhir

### Aturan ketat untuk coding assistant:
> Setiap fungsi fitur HARUS diberi unit test yang membuktikan fitur di baris `t` tidak menggunakan `high`/`low`/`close` dari candle yang closed setelah `t`. Kalau tidak bisa dibuktikan, fitur itu dibuang.

---

## 4. Fase 3 — Labeling

**File:** `labeling.py`

Implementasikan **dua** metode, bandingkan hasilnya:

1. **Fixed-horizon** (baseline cepat): `label = 1 if close[t+1] > close[t] else 0`
2. **Triple-barrier** (prioritas — lebih realistis):
   - Barrier atas: target profit (mis. +0.15%, sesuaikan dengan realized volatility)
   - Barrier bawah: stop loss (mis. -0.15%)
   - Barrier vertikal: batas waktu (5 menit / N candle)
   - Label = barrier mana yang tersentuh duluan

> **Catatan untuk coding assistant:** triple-barrier menghasilkan label yang overlap waktu (concurrent). Implementasikan sample weighting berdasarkan uniqueness, atau minimal tandai jelas di kode bahwa ini exist sebagai known limitation kalau di-skip.

---

## 5. Fase 4 — Model & Validasi

**File:** `models.py`, `validation.py`

### Model (urutan prioritas sesuai riset):
1. Logistic Regression — baseline wajib, jangan skip meski "boring"
2. **LightGBM / XGBoost — model utama**, mulai dari sini
3. LSTM/GRU — opsional, setelah baseline tree-based solid

### Validasi — WAJIB salah satu dari ini, TIDAK BOLEH random split:
- **Walk-forward validation**: train di window waktu N, test di window N+1, geser terus.
- **Purged K-Fold CV**: buang sampel training yang overlap waktu dengan test set, tambah embargo period.

### Kriteria selesai fase ini:
- Akurasi walk-forward rata-rata dilaporkan per fold (bukan angka tunggal yang bisa lucky).
- Precision/recall dilaporkan per kelas (naik/turun), bukan cuma akurasi keseluruhan.
- Uji performa terpisah di kondisi bullish/bearish/sideways — kalau model cuma bagus di satu rezim, catat sebagai risiko.

---

## 6. Fase 5 — Backtest dengan Biaya Realistis

**File:** `backtest.py`, `evaluate.py`

Simulasi HARUS memasukkan:
- Fee taker Binance sesuai tier akun (default ~0.1%, bisa lebih rendah dengan BNB)
- Slippage estimasi dari spread order book historis (bukan asumsi fill sempurna di harga close)
- Threshold probabilitas minimum sebelum eksekusi (mis. hanya trade kalau `P > 0.6`)

### Metrik wajib dilaporkan (bukan cuma akurasi):
```
EV = (win_rate × avg_profit_pct) - ((1 - win_rate) × avg_loss_pct) - fee - slippage
```
- Sharpe ratio
- Max drawdown
- Win rate vs risk/reward ratio (akurasi 52% dengan RR bagus bisa tetap profit; akurasi 55% dengan RR jelek bisa tetap rugi)

### Kriteria lolos ke fase berikutnya:
EV > 0 secara konsisten di walk-forward, bukan cuma di satu periode backtest.

---

## 7. Fase 6 — Paper Trading (Testnet)

**File:** `live_predict.py`, `executor.py`

- Jalankan sistem full end-to-end di **Binance Testnet**, uang virtual.
- Loop: candle closed → hitung fitur → predict_proba → cek threshold → kirim order testnet → log hasil.
- Simpan setiap prediksi + hasil aktual ke database untuk dibandingkan dengan backtest.
- Minimal jalan **beberapa minggu** sebelum dianggap valid.
- Bangun **kill-switch**: auto-stop kalau drawdown melebihi batas harian/mingguan, atau kalau koneksi/API bermasalah.

### Kriteria lolos ke live:
Hasil paper trading (EV, win rate, drawdown) konsisten dengan angka backtest — kalau meleset jauh, ada bug atau overfitting yang belum ketahuan.

---

## 8. Fase 7 — Live (Modal Kecil)

- Mulai dengan modal minimum yang sanggup hilang total.
- Monitoring ketat: dashboard/log realtime, alert kalau ada anomali.
- Scale up bertahap HANYA kalau performa live konsisten selama periode yang cukup panjang (bukan setelah 2-3 hari untung).
- Retraining berkala — data finansial non-stasioner, model butuh update rutin.

---

## 9. Checklist Anti-Kesalahan Umum

Tempel ini di README proyek sebagai reminder:

- [ ] Tidak ada lookahead bias di fitur manapun
- [ ] Validasi pakai walk-forward, bukan random split
- [ ] Fee & slippage masuk ke semua perhitungan profitabilitas
- [ ] Triple-barrier label sudah di-purge dari overlap (atau di-note sebagai limitation)
- [ ] Model di-retrain berkala, tidak dipakai statis selamanya
- [ ] Akurasi klasifikasi tidak disamakan dengan profitabilitas — selalu cek EV

---

## 10. Library yang Dipakai

```
python-binance          # REST + testnet
unicorn-binance-websocket-api  # websocket robust, auto-reconnect
pandas, numpy
ta / pandas-ta           # indikator teknikal
scikit-learn
lightgbm / xgboost
torch atau tensorflow    # kalau lanjut ke LSTM/GRU
mlfinpy                  # triple-barrier, purged CV (atau implementasi manual)
backtesting.py / vectorbt
sqlite3 / psycopg2        # simpan histori prediksi vs aktual
```

---

## Catatan Penting

Dokumen ini adalah blueprint teknis untuk membangun dan memvalidasi sistem — bukan jaminan profitabilitas. Berdasarkan riset akademik, akurasi directional 5 menit yang realistis ada di kisaran 55–67%, dan edge sekecil itu baru berarti setelah dikurangi fee dan slippage. Uji menyeluruh di testnet sebelum pakai dana riil, dan jangan alokasikan dana yang tidak sanggup Anda rugikan. Saya bukan penasihat keuangan — dokumen ini murni panduan implementasi teknis.