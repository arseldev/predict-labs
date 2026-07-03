# Evaluasi Sistem Prediksi BTC 5m — Yang Perlu Diperbaiki & Dimaksimalkan

Status: review teknis berdasarkan `features.py`, `backtest.py`, dan dokumen arsitektur.

---

## 🔴 Prioritas 1 — Wajib Diperbaiki (Bug Serius)

### 1. Bug sirkular: `market_ratio_up = proba`
**File:** `backtest.py`, fungsi `run_backtest`

```python
else:
    market_ratio_up = proba  # entry_cost dihitung dari proba model sendiri
```

Ini membuat `entry_cost` (harga token) dihitung dari keyakinan model itu sendiri, bukan dari harga pasar riil (order book Binance Predict/Polymarket). Akibatnya:
- Semua hasil backtest yang pernah kamu lihat kemungkinan **terlalu optimis**.
- Backtest ini sebenarnya mengukur "seberapa konsisten model dengan dirinya sendiri", bukan edge riil terhadap market.

**Perbaikan:**
- Idealnya: ambil `entry_cost` dari data historis harga pool riil (kalau tersedia).
- Kalau belum ada data itu: pakai `pool_ratio_source = "fixed"` dengan `fixed_ratio_up = 0.50` sebagai baseline realistis, lalu bandingkan performa model vs baseline ini.

### 2. Risk management di config tidak pernah dieksekusi
`max_daily_loss_pct` dan `max_weekly_loss_pct` didefinisikan di `BacktestConfig` tapi **tidak pernah dicek** di loop `run_backtest`. Kalau kamu berasumsi sistem "auto-stop" saat rugi >3%/hari, itu **tidak benar** — baik di backtest maupun kemungkinan juga di live jika logikanya sama.

**Perbaikan:** tambahkan pengecekan cumulative PnL harian/mingguan di dalam loop, dan hentikan entry baru jika limit tercapai.

### 3. Parameter dead code yang membingungkan
`use_triple_barrier`, `profit_target_pct`, `stop_loss_pct`, `max_hold_candles` — ada di config tapi tidak dipakai (exit selalu fix di T+1). Ini bikin config *terlihat* lebih canggih dari implementasi aslinya, rawan salah asumsi saat debugging nanti.

**Perbaikan:** hapus kalau memang tidak dipakai, atau implementasikan kalau memang berencana pakai exit dinamis.

---

## 🟡 Prioritas 2 — Validasi yang Belum Ada

### 4. Tidak ada walk-forward validation
Belum terlihat proses training/testing yang menghormati urutan waktu (time series). Random train/test split akan bocor informasi masa depan.

**Perbaikan:** pakai walk-forward (expanding/rolling window) — train di periode N, test di periode N+1, geser terus.

### 5. Belum ada expected value (EV) setelah fee riil
Akurasi klasifikasi arah (%UP benar) tidak sama dengan profitability. EV harus dihitung: `EV = P(menang) × payout_menang − P(kalah) × payout_kalah − fee`, dengan payout berbasis harga pool riil, bukan proba model.

### 6. Belum ada uji multi-rezim pasar
Model perlu diuji di kondisi pasar berbeda: trending naik, trending turun, sideways/choppy, volatilitas tinggi & rendah. Performa yang bagus di satu rezim saja rawan overfit.

---

## 🟢 Prioritas 3 — Feature Engineering (Sudah Cukup Baik, Ada Catatan Kecil)

- ✅ `add_htf_context` sudah benar pakai `.shift(1)` sebelum merge — mencegah lookahead dari candle 1h yang belum closed.
- ⚠️ `dropna()` di akhir pipeline bisa membuang banyak baris diam-diam kalau data order book/trades ada gap. Sebaiknya log persentase NaN per kolom sebelum drop, supaya tahu kualitas data sebenarnya.
- ⚠️ `rvol_48` & `ret_4h` butuh warmup ±4 jam data valid. Pastikan proses warmup di `live_predict.py` benar-benar mengisi ini sebelum candle live pertama diproses — kalau tidak, sinyal pertama bisa gagal diam-diam (NaN → skip tanpa notifikasi jelas).
- 🔎 Fitur mikro (OBI, trade flow imbalance) punya edge paling nyata untuk 5m karena capture info jangka sangat pendek, tapi juga paling cepat basi — perlu re-training/re-kalibrasi lebih sering dibanding fitur teknikal biasa (RSI/MACD/EMA yang sudah sangat umum dan kemungkinan sudah priced-in market lain).

---

## 🟡 Prioritas 2b — Timing Entry (Kapan Order Dieksekusi)

Saat ini sistem tampaknya langsung entry begitu window prediksi baru dibuka (detik ke-0 setelah candle T closed). Ini berisiko dan sebaiknya diperbaiki:

### Masalah entry instan
1. **Likuiditas pool belum stabil di awal window** — begitu window baru dibuka, peserta pool masih sedikit, sehingga `entry_cost` (harga token) bisa bergerak liar dalam beberapa detik pertama dan jauh dari fair value.
2. **Lag konfirmasi candle closed** — kline WebSocket butuh waktu (network delay, parsing) untuk benar-benar final/tidak direvisi. Entry yang terlalu cepat berisiko memakai data candle yang belum sepenuhnya settle.
3. **Order flow di detik-detik awal window sering noise**, bukan tren riil — bisa jadi reaksi sisa dari closing candle sebelumnya (mis. stop hunt kecil), bukan representasi kondisi candle baru yang sedang berjalan.

### Rekomendasi
- **Tambahkan jeda kecil (± 3–10 detik) setelah candle closed** sebelum eksekusi order, untuk memastikan data kline final dan harga pool sudah mulai settle. Jeda ini tidak signifikan mengurangi durasi efektif candle 5 menit, tapi cukup untuk menghindari entry di harga yang belum wajar.
- **Tambahkan sanity-check pada `entry_cost` sebelum submit order**: jika harga pool masih bergerak tajam (>threshold tertentu) dalam window singkat setelah open, lebih aman skip trade tersebut daripada memaksa entry.
- **Jangan menunggu terlalu lama** (misal beberapa menit) — itu malah membuang sebagian besar durasi candle T+1 yang jadi basis exit, sehingga edge sinyal ikut berkurang. Yang dibutuhkan adalah **detik**, bukan menit.
- Pertimbangkan menambahkan mekanisme retry/re-check: ambil beberapa snapshot `entry_cost` berturut-turut dalam window delay tersebut, baru submit order berdasarkan harga yang sudah lebih stabil (mis. rata-rata 2–3 snapshot terakhir).

---

## 📁 Perlu Hapus & Ulangi Data Simulasi?

**Ya, sebaiknya dihapus dan disimulasikan ulang** — dengan alasan spesifik berikut, bukan sekadar kehati-hatian umum:

1. Bug `market_ratio_up = proba` memengaruhi **perhitungan PnL setiap trade** (via `entry_cost`), bukan cuma statistik ringan. Setiap angka net_pnl, win rate, dan equity curve yang sudah dihasilkan **tidak valid untuk dijadikan dasar keputusan** karena baseline biaya masuknya salah secara sistematis.
2. Karena bug ini bias-nya konsisten ke arah menguntungkan (entry_cost otomatis "selaras" dengan keyakinan model), hasil lama akan selalu terlihat **lebih baik dari kenyataan** — bukan sekadar noise acak yang bisa diabaikan.
3. Tidak perlu hapus *data harga OHLCV mentah* atau *fitur* yang sudah dihitung (itu masih valid, tidak terkait bug) — cukup **hapus/jalankan ulang hasil backtest & equity curve** setelah fix diterapkan. Data fitur (features.py output) bisa dipakai lagi tanpa masalah.

**Rekomendasi urutan kerja:**
1. Perbaiki `market_ratio_up` (pakai `fixed_ratio_up=0.50` dulu sebagai baseline, atau data pool historis riil kalau ada).
2. Jalankan ulang backtest dari data fitur yang sudah ada (tidak perlu rebuild dari nol).
3. Implementasikan pengecekan daily/weekly loss limit.
4. Baru setelah itu percaya angka win-rate/EV yang keluar untuk keputusan lanjut ke paper trading.

---

## Catatan Penutup

Bukan saran finansial — ini murni review teknis kode dan metodologi. Sebelum masuk modal riil (bahkan mode `paper`), pastikan backtest dengan entry_cost yang sudah diperbaiki tetap menunjukkan EV positif secara konsisten di berbagai rezim pasar dan periode out-of-sample yang model belum pernah lihat.