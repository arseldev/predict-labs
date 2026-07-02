# Riset & Analisa: Sistem Prediksi Arah Harga Bitcoin 5 Menit (Up/Down) di Binance

**Tanggal riset:** 2 Juli 2026
**Tujuan:** Merangkum metode yang terbukti efektif secara akademik/praktik untuk memprediksi arah pergerakan harga BTC pada timeframe 5 menit, sekaligus blueprint implementasi menggunakan Python.

---

## 1. Ringkasan Eksekutif

Prediksi arah harga (klasifikasi biner: naik/turun) pada timeframe 5 menit **bisa dilakukan dengan akurasi di atas 50%**, tapi jangan berharap angka fantastis. Beberapa studi akademik terkontrol pada interval 5 menit melaporkan akurasi sekitar **59–67%** menggunakan model seperti XGBoost, Random Forest, dan LSTM/GRU. Ini terdengar kecil, tapi dalam trading frekuensi tinggi, edge sekecil itu *bisa* profitable — dengan syarat: biaya transaksi (fee + spread + slippage) diperhitungkan, manajemen risiko ketat, dan sistem divalidasi dengan metode yang tidak bocor (leak) informasi masa depan.

Poin paling penting yang sering diabaikan pemula: **akurasi klasifikasi bukan satu-satunya metrik yang penting**. Model dengan akurasi 55% bisa saja rugi terus kalau posisi yang salah menghasilkan kerugian lebih besar dari posisi yang benar (atau kalau fee memakan seluruh edge). Jadi fokus risetnya bukan cuma "model apa yang akurat", tapi "sistem apa yang *expected value*-nya positif setelah biaya riil".

Python **bisa mencakup seluruh pipeline** — dari ambil data live/historis dari Binance, feature engineering, training model, backtesting, sampai eksekusi order otomatis. Tidak perlu bahasa lain, kecuali nanti butuh performa ultra-low-latency (itu di luar cakupan 5 menit, biasanya baru relevan untuk HFT sub-detik).

---

## 2. Realita yang Harus Dipahami Dulu

1. **Pasar crypto cukup efisien di timeframe pendek.** Order book dan harga sudah mencerminkan hampir semua informasi publik. Edge yang bisa ditemukan biasanya kecil dan bersifat statistik (bukan sinyal pasti), berbasis pada microstructure pasar (ketimpangan order book, arus transaksi), bukan "pola candlestick ajaib".
2. **Biaya transaksi adalah musuh utama.** Fee taker Binance spot sekitar 0,1% (bisa lebih rendah dengan BNB/VIP tier), plus spread dan slippage. Untuk sinyal 5 menit, pergerakan harga rata-rata sering **lebih kecil dari biaya bolak-balik (buy+sell)**. Riset tentang order book imbalance pada crypto secara eksplisit menemukan bahwa return rata-rata di jendela waktu pendek berada di bawah biaya transaksi umum di bursa crypto (~10 bps), sehingga sinyal microstructure murni sering *tidak* otomatis profitable tanpa mempertimbangkan spread dan fee.
3. **Akurasi 55–60% sudah dianggap bagus** untuk klasifikasi arah harga jangka pendek di pasar finansial manapun (bukan cuma crypto). Siapa pun yang menjanjikan akurasi 90%+ pada arah harga 5 menit murni dari OHLCV, patut dicurigai overfitting.
4. **Model harus divalidasi dengan cara yang meniru kondisi real trading** (walk-forward, bukan random train-test split), atau angka akurasi akan menyesatkan.

---

## 3. Bukti dari Riset Akademik (Ringkasan)

| Studi / Pendekatan | Model | Timeframe | Akurasi Directional |
|---|---|---|---|
| Ranjan et al. (2022) | XGBoost | 5 menit | ~59,4% |
| Ranjan et al. (2022) | Logistic Regression | Harian | ~64,8% |
| Studi "sample dimension engineering" | RF, XGBoost, QDA, SVM, LSTM | 5 menit | hingga ~67,2% |
| Jaquart, Dann & Weinhardt | GRU vs LSTM vs GBC vs RF | 1, 5, 15, 60 menit | GRU & LSTM setara di 1–5 menit; GBC unggul tipis di 5 menit |
| CryptOL (proyek open-source) | Linear Regression | 15 menit | klaim hingga ~72% (skala kecil, hati-hati overfitting) |

**Insight kunci dari literatur ini:**
- Untuk timeframe **sangat pendek (5 menit)**, model berbasis pohon (Random Forest, XGBoost, Gradient Boosting) dan model deep learning sekuensial (LSTM/GRU) secara konsisten mengalahkan model statistik sederhana (Logistic Regression), berbeda dengan prediksi harian di mana model statistik sederhana justru sering menang.
- Feature selection dan kualitas input data punya pengaruh lebih besar terhadap akurasi dibanding kompleksitas arsitektur model. Riset terbaru tentang limit order book crypto secara eksplisit menyimpulkan bahwa memperbaiki kualitas fitur input jauh lebih berdampak daripada menambah layer di neural network.
- Fitur berbasis **order book microstructure** (order flow imbalance, micro-price, spread) terbukti punya hubungan yang kuat dan hampir linear dengan pergerakan harga jangka sangat pendek (level detik hingga menit), dan pola ini konsisten lintas berbagai koin crypto (BTC, LTC, ETC, ENJ, ROSE), tidak hanya spesifik Bitcoin.

---

## 4. Kerangka Sistem End-to-End

### 4.1 Data yang Dibutuhkan

| Sumber Data | Kegunaan | Endpoint Binance |
|---|---|---|
| Kline/candlestick 5m (dan multi-timeframe: 1m, 15m, 1h) | Fitur teknikal, label | REST `klines` / WebSocket `kline` stream |
| Order book depth (top 5–20 level) | Fitur microstructure (imbalance, spread) | REST `depth` / WebSocket `depth` stream |
| Trade/aggTrade stream | Order flow, volume delta (buy vs sell aggresif) | WebSocket `aggTrade` stream |
| Funding rate & open interest (jika pakai futures) | Sentimen leverage pasar | Binance Futures API |
| Data eksternal opsional | Sentimen (Fear & Greed Index), dominasi BTC, korelasi ETH | API pihak ketiga |

Untuk histori jangka panjang dalam jumlah besar, Binance menyediakan **bulk data (CSV/ZIP)** di `data.binance.vision` — lebih cepat dan tidak membebani rate limit dibanding menarik ribuan candle lewat REST API satu per satu.

### 4.2 Feature Engineering

**a. Fitur teknikal klasik (dari OHLCV)**
- Return log pada berbagai lag (1, 3, 5, 10 candle)
- RSI, MACD, Bollinger Band width, ATR (volatilitas)
- Moving average cross (EMA cepat vs lambat)
- Volume relatif terhadap rata-rata bergerak
- Fitur dari timeframe lebih besar (1h, 4h) sebagai konteks tren (multi-timeframe)

**b. Fitur microstructure (order book & trade flow) — ini yang sering jadi pembeda akurasi**
- **Order Book Imbalance (OBI):** `(volume_bid - volume_ask) / (volume_bid + volume_ask)` pada beberapa level kedalaman
- **Micro-price / VAMP (Volume-Adjusted Mid Price):** estimasi "harga wajar" yang lebih akurat dari mid-price biasa, terbukti pada riset BTC menjadi prediktor jangka pendek yang lebih baik dibanding mid-price polos
- **Spread bid-ask** relatif terhadap harga
- **Trade flow imbalance:** rasio volume taker buy vs taker sell dalam window waktu tertentu
- **Kedalaman order book (depth)** pada beberapa level sebagai indikator likuiditas

**c. Fitur statistik/waktu**
- Jam dalam hari, hari dalam minggu (crypto punya pola likuiditas berdasarkan sesi pasar Asia/Eropa/AS)
- Volatilitas realized dalam N candle terakhir

**Penting:** semua fitur harus dihitung hanya dari data yang **sudah tersedia pada saat prediksi dibuat** (tidak boleh "mengintip" candle yang belum closed — lookahead bias adalah penyebab #1 sistem terlihat bagus di backtest tapi gagal live).

### 4.3 Cara Melabeli Data (Labeling)

Ini bagian yang sering diremehkan padahal krusial untuk akurasi riil.

**Metode 1 — Fixed Time Horizon (paling umum & paling sederhana):**
Label = 1 jika `close[t+5m] > close[t]`, else 0. Mudah, tapi kelemahannya: mengabaikan jalur harga di antaranya (bisa saja harga naik dulu lalu turun tajam, padahal ditandai "naik" karena titik akhir kebetulan lebih tinggi).

**Metode 2 — Triple-Barrier Method (lebih realistis, dari Marcos López de Prado):**
Untuk tiap titik waktu, tetapkan 3 barrier:
- Barrier atas = target profit (mis. +0,15%)
- Barrier bawah = stop loss (mis. -0,15%)
- Barrier vertikal = batas waktu (mis. 5 menit / N candle)

Label ditentukan oleh barrier mana yang tersentuh lebih dulu. Ini jauh lebih dekat dengan kondisi trading nyata (karena trading sungguhan punya take-profit/stop-loss), dan mengurangi risiko model belajar pola yang tidak actionable.

**Catatan teknis penting:** Triple-barrier menghasilkan label yang **overlap secara waktu** (concurrent), sehingga melanggar asumsi data "independen dan identik" yang dipakai kebanyakan algoritma ML. Solusinya: gunakan *sample weighting* berdasarkan uniqueness label, atau minimal sadari bahwa akurasi in-sample akan terlihat lebih baik dari performa riil kalau ini diabaikan.

### 4.4 Model — Mana yang Efektif?

| Model | Kapan Cocok | Catatan |
|---|---|---|
| **Logistic Regression** | Baseline wajib | Cepat, interpretable, sering jadi patokan minimum |
| **Random Forest** | Data tabular dengan banyak fitur & interaksi nonlinear | Robust terhadap noise, relatif tahan overfitting |
| **XGBoost / LightGBM** | Rekomendasi utama untuk 5 menit berdasarkan literatur | Konsisten unggul untuk directional prediction jangka pendek; cepat dilatih, mudah tuning, mendukung feature importance & SHAP |
| **LSTM / GRU** | Kalau punya cukup data & ingin menangkap pola sekuensial | Performanya sebanding dengan tree-based model di 5 menit menurut riset; butuh data lebih banyak dan lebih rawan overfitting |
| **Ensemble / Stacking** | Setelah punya beberapa model solid | Kombinasi tree-based + sequential sering menambah stabilitas |

**Rekomendasi praktis:** mulai dari **LightGBM/XGBoost** sebagai model utama (cepat diiterasi, importance fitur mudah dianalisis), baru eksplorasi LSTM/GRU kalau sudah punya fondasi fitur & data yang kuat.

**Teknik lanjutan — Meta-Labeling:** alih-alih model langsung memutuskan arah (naik/turun), pisahkan jadi dua tahap: (1) model/aturan sederhana menentukan sisi (long/short), (2) model ML kedua hanya memutuskan apakah sinyal itu layak dieksekusi (ya/tidak) dan seberapa besar ukuran posisinya. Pendekatan ini terbukti mengurangi overfitting karena ML tidak menebak arah dari nol, hanya menyaring kualitas sinyal.

### 4.5 Validasi — Cara yang Benar (Ini Penentu Utama Berhasil/Gagalnya Sistem)

Kesalahan paling umum sistem trading berbasis ML: memakai **random train-test split atau k-fold cross-validation biasa**. Ini menghasilkan akurasi yang terlihat tinggi di eksperimen, tapi menyesatkan karena data finansial berurutan waktu (ada autokorelasi), sehingga model "mengintip" masa depan secara tidak sengaja.

Metode yang benar:
1. **Walk-Forward Validation:** latih model pada periode waktu tertentu, uji pada periode berikutnya yang belum pernah dilihat, lalu geser jendela waktu ke depan dan ulangi. Ini paling mendekati simulasi kondisi trading riil.
2. **Purged K-Fold Cross-Validation:** kalau tetap ingin pakai cross-validation, buang (purge) sampel training yang labelnya overlap secara waktu dengan sampel testing, dan tambahkan "embargo" periode setelah test set sebelum fold training berikutnya dimulai.
3. **Walk-Forward + berbagai kondisi pasar:** uji performa di periode bullish, bearish, dan sideways secara terpisah — model yang cuma bagus di satu kondisi pasar biasanya rapuh saat rezim pasar berubah.

### 4.6 Metrik Evaluasi — Jangan Cuma Lihat Akurasi

- **Precision & recall per kelas** (naik vs turun) — model bisa akurat tinggi hanya karena selalu prediksi kelas mayoritas
- **Log-loss / Brier score** — mengukur kalibrasi probabilitas, penting kalau nanti probabilitas dipakai untuk position sizing
- **Expected Value setelah biaya:** `EV = (P_menang × rata-rata_profit) - (P_kalah × rata-rata_rugi) - fee - slippage`. Ini metrik paling penting sebelum sistem dianggap layak jalan
- **Sharpe ratio & max drawdown** dari hasil backtest strategi (bukan cuma akurasi klasifikasi mentah)

### 4.7 Dari Sinyal ke Keputusan Trading

- **Threshold probabilitas:** jangan trading di setiap prediksi. Hanya eksekusi kalau probabilitas model melebihi ambang tertentu (mis. >60% naik) — ini menyaring sinyal lemah yang tidak menutup biaya transaksi
- **Position sizing:** gunakan ukuran posisi proporsional terhadap keyakinan model (atau fixed-fractional/Kelly yang dikonservatifkan)
- **Risk management wajib:** stop-loss, take-profit, dan batas maksimum kerugian harian/mingguan — independen dari seberapa "yakin" model
- **Biaya riil:** hitung fee maker/taker Binance sesuai tier VIP akun, plus slippage estimasi dari spread order book saat backtest

---

## 5. Implementasi dengan Python — Bisa Cover Semua?

**Ya, Python bisa mencakup seluruh pipeline ini end-to-end:**

| Tahap | Library Python |
|---|---|
| Ambil data historis & live dari Binance | `python-binance`, `ccxt` (multi-exchange), atau `binance-connector-python` resmi; untuk histori besar pakai bulk data dari `data.binance.vision` |
| WebSocket real-time (kline, depth, trade) | `python-binance` (`ThreadedWebsocketManager`), atau `unicorn-binance-websocket-api` (lebih robust untuk reconnect otomatis & production) |
| Manipulasi data & fitur | `pandas`, `numpy`, `ta` / `pandas-ta` (indikator teknikal siap pakai) |
| Model ML klasik | `scikit-learn` (Logistic Regression, RF), `xgboost`, `lightgbm` |
| Model deep learning (LSTM/GRU) | `pytorch` atau `tensorflow/keras` |
| Validasi finansial (purged CV, triple-barrier) | `mlfinlab`/`mlfinpy` (implementasi teknik dari buku *Advances in Financial Machine Learning*), atau implementasi manual (tidak sulit, ~50 baris kode) |
| Backtesting strategi | `backtesting.py`, `vectorbt`, atau backtest custom dengan pandas |
| Eksekusi order otomatis (live) | `python-binance` / `ccxt` untuk kirim order ke Binance API |
| Monitoring & logging | `logging` bawaan Python, database `sqlite`/`PostgreSQL` untuk simpan histori prediksi vs hasil aktual |

Python cukup cepat untuk timeframe 5 menit (bukan HFT sub-detik), jadi tidak ada kebutuhan bahasa lain seperti C++/Rust kecuali nanti sistem berkembang ke strategi market-making berbasis order book di level milidetik.

### 5.1 Kerangka Kode (Skeleton Pipeline)

Kode di bawah ini adalah **kerangka**, bukan sistem siap pakai — tujuannya menunjukkan struktur pipeline. Perlu disesuaikan, diuji, dan divalidasi sebelum dipakai dengan uang sungguhan.

```python
# ==== 1. AMBIL DATA HISTORIS ====
from binance.client import Client
import pandas as pd

client = Client(api_key, api_secret)

def get_klines(symbol="BTCUSDT", interval="5m", limit=1000):
    klines = client.get_historical_klines(symbol, interval, "90 days ago UTC")
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume","taker_buy_base"]:
        df[c] = df[c].astype(float)
    return df.set_index("open_time")

df = get_klines()

# ==== 2. FEATURE ENGINEERING ====
import numpy as np
import ta  # pip install ta

def build_features(df):
    df = df.copy()
    df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_5"] = np.log(df["close"] / df["close"].shift(5))
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["vol_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    # proxy order-flow dari kline: rasio taker buy volume terhadap total volume
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"]
    return df.dropna()

df = build_features(df)

# ==== 3. LABELING (fixed horizon sederhana, 1 candle ke depan = 5 menit) ====
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna()

# ==== 4. WALK-FORWARD VALIDATION + MODEL ====
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

features = ["ret_1","ret_5","rsi","atr","vol_ma_ratio","taker_buy_ratio"]
X, y = df[features], df["target"]

def walk_forward_eval(X, y, n_splits=5, test_size=500):
    n = len(X)
    fold_size = n // (n_splits + 1)
    results = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + test_size, n)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        if len(X_test) == 0:
            continue
        model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append(acc)
        print(f"Fold {i+1}: akurasi = {acc:.4f}")
    print(f"Rata-rata akurasi walk-forward: {np.mean(results):.4f}")
    return results

walk_forward_eval(X, y)

# ==== 5. EVALUASI EXPECTED VALUE SETELAH FEE (contoh sederhana) ====
FEE_ROUND_TRIP = 0.0008  # ~0.08% (buy+sell), sesuaikan tier VIP & pemakaian BNB
def expected_value(win_rate, avg_win_pct, avg_loss_pct, fee=FEE_ROUND_TRIP):
    return (win_rate * avg_win_pct) - ((1 - win_rate) * avg_loss_pct) - fee
```

```python
# ==== 6. LIVE STREAMING UNTUK PREDIKSI REAL-TIME ====
from binance import ThreadedWebsocketManager

def handle_kline(msg):
    k = msg["k"]
    if k["x"]:  # candle sudah closed
        # ambil data terbaru, hitung ulang fitur, panggil model.predict_proba()
        print(f"Candle 5m closed: close={k['c']}, volume={k['v']}")
        # -> update dataframe, hitung fitur, prediksi, cek threshold, eksekusi/skip

twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
twm.start()
twm.start_kline_socket(callback=handle_kline, symbol="BTCUSDT", interval="5m")
twm.join()
```

### 5.2 Yang Perlu Ditambahkan Sebelum Ini Layak Dipakai Riil

1. Ganti label fixed-horizon dengan **triple-barrier** agar lebih realistis terhadap risk/reward
2. Tambahkan **fitur order book** (butuh depth stream, bukan cuma kline)
3. Tambahkan **purging & embargo** di walk-forward agar tidak ada leakage antar fold
4. Backtest lengkap dengan **fee, slippage, dan latency eksekusi** yang realistis (order tidak selalu terisi tepat di harga close candle)
5. Paper trading dulu di **Binance Testnet** sebelum live dengan uang sungguhan
6. Sistem **kill-switch**: hentikan otomatis kalau drawdown melebihi batas, atau kalau model API/koneksi bermasalah

---

## 6. Kesalahan Umum yang Membuat Sistem Gagal di Live Trading

1. **Lookahead bias** — fitur dihitung memakai data yang seharusnya belum diketahui saat prediksi dibuat (mis. memakai high/low candle yang belum closed)
2. **Overfitting ke periode backtest tertentu** — model "hafal" pola satu rezim pasar (mis. bull run 2024–2025) dan gagal total saat rezim berubah
3. **Mengabaikan biaya transaksi & slippage** saat menghitung profitabilitas backtest
4. **Label leakage dari triple-barrier tanpa purging** — akurasi in-sample tinggi tapi tidak representatif
5. **Data non-stasioner** — hubungan statistik antar fitur dan harga berubah seiring waktu (perlu retraining berkala, bukan model sekali latih dipakai selamanya)
6. **Menyamakan akurasi klasifikasi dengan profitabilitas** — akurasi 55% dengan risk/reward buruk bisa tetap rugi; akurasi 52% dengan risk/reward bagus bisa tetap profit

---

## 7. Roadmap Bertahap yang Disarankan

1. **Kumpulkan & simpan data** kline + order book + trade selama minimal beberapa bulan (mulai sekarang, karena order book snapshot historis panjang sulit didapat gratis)
2. **Bangun baseline sederhana** (Logistic Regression + fitur teknikal dasar) untuk punya patokan
3. **Tambahkan fitur microstructure** dan bandingkan peningkatan performa vs baseline
4. **Uji beberapa model** (LightGBM/XGBoost sebagai prioritas, lalu LSTM/GRU) dengan walk-forward validation
5. **Backtest dengan biaya realistis**, evaluasi expected value, Sharpe ratio, dan max drawdown — bukan cuma akurasi
6. **Paper trading** di testnet minimal beberapa minggu untuk validasi live tanpa risiko dana
7. **Live dengan modal kecil** dan monitoring ketat, baru scale up kalau performa live konsisten dengan backtest

---

## 8. Catatan Risiko

Trading crypto berisiko tinggi dan bisa menyebabkan kerugian besar, termasuk kerugian modal secara keseluruhan. Riset dan backtest yang bagus **tidak menjamin** profitabilitas di masa depan — kondisi pasar berubah, dan model statistik bisa berhenti bekerja kapan saja (regime change). Dokumen ini bersifat edukatif/riset teknis, bukan rekomendasi investasi atau jaminan keuntungan. Selalu uji sistem secara menyeluruh di testnet/paper trading sebelum menggunakan dana riil, dan jangan mengalokasikan dana yang tidak sanggup Anda rugikan.

---

## 9. Referensi

- Ranjan, S. et al. (2022) — perbandingan model ML untuk prediksi harga BTC harian & 5 menit
- Studi "Bitcoin price prediction using machine learning: sample dimension engineering" — ScienceDirect
- Jaquart, P., Dann, D., & Weinhardt, C. — "Short-term bitcoin market prediction via machine learning" — Journal of Finance and Data Science / ScienceDirect
- "Mind the Gaps: Short-Term Crypto Price Prediction" (SSRN) — micro-price & VAMP untuk BTC
- "Explainable Patterns in Cryptocurrency Microstructure" (arXiv, 2026) — order flow imbalance lintas aset crypto
- "Price Impact of Order Book Imbalance in Cryptocurrency Markets" — Towards Data Science
- Marcos López de Prado — *Advances in Financial Machine Learning* (Triple-Barrier Method, Purged K-Fold CV, Meta-Labeling)
- Dokumentasi resmi `python-binance` dan `unicorn-binance-websocket-api`
- Binance bulk historical data: data.binance.vision