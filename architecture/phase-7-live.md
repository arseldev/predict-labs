# Phase 7 — Live Trading (Modal Kecil)

> **Tujuan:** Transisi dari paper trading ke live trading dengan modal minimum, monitoring ketat, dan scale-up yang bertahap berdasarkan bukti performa nyata.

> ⚠️ **Peringatan:** Phase ini melibatkan uang riil. Baca seluruh dokumen ini sebelum memulai. Jangan alokasikan dana yang tidak sanggup Anda rugikan.

---

## 7.1 Pra-Kondisi Wajib

Sebelum mengaktifkan live mode, konfirmasi **semua** poin ini:

```
CHECKLIST FINAL SEBELUM LIVE:
  ✅ Phase 1-6 selesai sepenuhnya
  ✅ Paper trading minimal 4 minggu, EV positif
  ✅ Paper vs backtest selisih < 5% win rate
  ✅ Kill-switch ditest dan terbukti bekerja
  ✅ Semua unit test pass
  ✅ Sistem berjalan stabil 24/7 tanpa restart lebih dari 1x/minggu
  ✅ Modal yang dialokasikan = dana yang SANGGUP HILANG TOTAL
  ✅ Sudah punya Binance API key live (bukan testnet)
  ✅ API key sudah dibatasi: hanya Spot Trading, tidak bisa withdraw
  ✅ IP whitelist aktif di API key settings Binance
```

---

## 7.2 Keamanan API Key

```bash
# Di Binance API Management:
# 1. Buat API key baru khusus untuk bot ini
# 2. Aktifkan: "Enable Spot & Margin Trading"
# 3. NONAKTIFKAN: "Enable Withdrawals" — bot tidak perlu ini!
# 4. IP Restriction: whitelist hanya IP server bot
# 5. Jangan pernah simpan API key di code atau git

# Di server:
export BINANCE_API_KEY="xxxx"
export BINANCE_API_SECRET="xxxx"

# Atau gunakan secret manager (rekomendasi untuk produksi):
# AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault
```

---

## 7.3 Perubahan dari Testnet ke Live

Ubah hanya satu hal di `config.yaml`:

```yaml
binance:
  testnet: false   # ← Ini satu-satunya perubahan
```

Dan pastikan API key di `.env` diganti ke live key (bukan testnet key).

Semua logic lain IDENTIK dengan paper trading. Kalau perlu mengubah logic untuk live mode, itu tanda ada bug dalam paper trading — perbaiki dulu.

---

## 7.4 Capital Management (Position Sizing)

### Recommended Starting Capital

| Level | Modal | Risk per Trade | Max Daily Loss | Catatan |
|---|---|---|---|---|
| Minimum | $100 | 1% ($1) | $3 | Sangat kecil, hampir tidak profitable setelah fee |
| Starter | $500 | 2% ($10) | $15 | Cukup untuk test sistem riil |
| Comfortable | $1,000-$2,000 | 2% ($20-$40) | $60-$120 | Range yang lebih reasonable |

**Aturan emas:** Mulai dengan modal terkecil yang masih meaningful untuk Anda, bukan terbesar yang "mungkin saja OK".

### Position Sizing yang Benar

```python
# Fixed Fractional (dipakai di config default)
position_size = capital * config["trading"]["position_size_pct"]  # 2%

# Kelly Criterion (lebih agresif, gunakan half-Kelly untuk safety)
# Kelly = (win_rate - loss_rate / risk_reward) / 1
kelly_fraction = (win_rate - (1 - win_rate) / rr_ratio)
half_kelly = kelly_fraction / 2  # Selalu gunakan half-Kelly minimum
position_size = capital * min(half_kelly, 0.05)  # Cap di 5%

# Rekomendasi: Fixed 2% per trade untuk awal
```

---

## 7.5 Monitoring Dashboard

Bangun monitoring sederhana yang bisa dipantau setiap hari:

```python
"""
monitoring/dashboard.py — CLI dashboard untuk monitoring live performance
Jalankan: python -m monitoring.dashboard
"""

def print_live_dashboard(pred_logger: PredictionLogger):
    """
    Dashboard terminal yang update setiap menit.
    
    Tampilkan:
    ┌─────────────────────────────────────────┐
    │   BTC 5M PREDICTION SYSTEM MONITOR     │
    │   Status: 🟢 RUNNING | Mode: LIVE      │
    ├─────────────────────────────────────────┤
    │ TODAY (2024-06-01):                     │
    │   Signals:   12                         │
    │   Trades:    8 (4 TP | 3 SL | 1 TO)    │
    │   Win Rate:  50.0%                      │
    │   PnL:       +$2.34 (net after fee)     │
    │   Fee paid:  $1.82                      │
    ├─────────────────────────────────────────┤
    │ LAST 7 DAYS:                            │
    │   Win Rate:  53.2%                      │
    │   Total PnL: +$18.50                    │
    │   Sharpe:    0.89                       │
    │   Max DD:    -2.1%                      │
    ├─────────────────────────────────────────┤
    │ SYSTEM:                                 │
    │   WebSocket: ✅ Connected               │
    │   Last ping: 3s ago                     │
    │   API calls: 234/1200 weight            │
    │   Kill-switch: OFF (loss: $1.20/$30.0)  │
    └─────────────────────────────────────────┘
    """
    pass  # Implementasi dengan rich library atau simple print
```

---

## 7.6 Scale-Up Protocol

Jangan scale-up secara emosional ("sudah profit 3 hari berturut-turut!").
Gunakan kriteria berbasis data:

```
PROTOCOL SCALE-UP MODAL:

Syarat untuk 2x modal:
  ✅ Live trading berjalan minimal 4 minggu
  ✅ Win rate live konsisten dengan paper (selisih < 5%)
  ✅ EV positif selama periode tersebut
  ✅ Max drawdown tidak melebihi 2x backtest prediction
  ✅ Tidak ada bug kritikal dalam 2 minggu terakhir

Cara scale-up yang aman:
  - Naikkan position_size_pct perlahan (dari 2% ke 2.5%, bukan langsung 5%)
  - Atau: deposit modal tambahan dengan size yang sama
  - Review kembali setelah 2 minggu di level baru sebelum scale lagi

RED FLAGS — turunkan modal atau stop:
  ❌ Win rate live < paper trading - 10%
  ❌ EV negatif 3 hari berturut-turut
  ❌ Drawdown mencapai 50% dari max_drawdown config
  ❌ Sistem crash lebih dari 1x dalam seminggu
```

---

## 7.7 Retraining Protocol untuk Live

```python
"""
Retraining schedule yang mature untuk sistem live.
"""

class LiveRetrainingScheduler:
    """
    Automated retraining dengan safeguards.
    
    Trigger retraining:
    1. Weekly scheduled retrain (Minggu tengah malam)
    2. Performance degradation: win rate 7-day < (backtest win rate - 8%)
    3. Feature drift: distribusi input fitur berubah signifikan (KL divergence)
    
    Safeguards:
    1. Model baru TIDAK langsung deploy ke produksi
    2. Jalankan backtest model baru pada 30 hari terakhir
    3. Compare model baru vs model lama menggunakan periode yang sama
    4. Deploy hanya jika model baru >= model lama - 2% accuracy
    5. Kalau tidak, pertahankan model lama dan kirim alert
    """
    
    def run_retrain_and_evaluate(self, new_data: pd.DataFrame) -> bool:
        """
        Returns True jika model baru di-deploy, False jika model lama dipertahankan.
        """
        # 1. Train model baru dengan data terbaru
        new_model = self._train_new_model(new_data)
        
        # 2. Backtest keduanya pada periode yang SAMA (recent holdout)
        holdout_period = new_data.tail(2000)  # ~1 minggu 5m candle
        old_results = self._backtest_model(self.current_model, holdout_period)
        new_results = self._backtest_model(new_model, holdout_period)
        
        # 3. Deploy hanya jika model baru tidak signifikan lebih buruk
        if new_results["accuracy"] >= old_results["accuracy"] - 0.02:
            self._deploy_model(new_model)
            logger.info(f"New model deployed: {new_results['accuracy']:.4f} "
                       f"vs old {old_results['accuracy']:.4f}")
            return True
        else:
            logger.warning(f"New model NOT deployed: too much degradation "
                          f"({new_results['accuracy']:.4f} < "
                          f"{old_results['accuracy']:.4f} - 0.02)")
            return False
```

---

## 7.8 Incident Response

Buat runbook untuk skenario darurat:

### Skenario 1: Sistem Crash saat Ada Posisi Terbuka
```bash
# 1. Check posisi terbuka di Binance
python -c "from src.executor import *; check_open_positions()"

# 2. Jika perlu, close manual via Binance app/web
# 3. Kill semua proses bot
pkill -f "main.py"

# 4. Investigasi log
tail -n 200 logs/system.log

# 5. Fix bug, test di testnet, deploy ulang
```

### Skenario 2: Win Rate Tiba-tiba Drop
```
Langkah investigasi:
1. Cek apakah ada regime change (volatilitas, trend BTC berubah drastis)
2. Bandingkan distribusi fitur hari ini vs baseline (feature drift)
3. Cek apakah API/data masih benar (tidak ada data corrupt)
4. Cek waktu prediksi vs waktu eksekusi (latency meningkat?)
5. Jika tidak ditemukan penyebab jelas: turunkan threshold ke 0.65+ 
   atau pause trading sampai investigasi selesai
```

### Skenario 3: API Rate Limit Tercapai
```python
# Tambahkan exponential backoff di semua API calls
import time

def api_call_with_retry(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except BinanceAPIException as e:
            if e.code == -1003:  # TOO_MANY_REQUESTS
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## 7.9 Legal & Tax Considerations

> **Catatan:** Ini bukan saran hukum/pajak. Konsultasikan dengan profesional.

- **Rekam semua transaksi** dari database untuk keperluan laporan pajak
- Di Indonesia: crypto dianggap aset, keuntungan bisa kena PPh
- `PredictionLogger` sudah menyimpan semua trade — gunakan ini untuk ekspor laporan
- Binance juga menyediakan export histori trading (Tax Report)

---

## 7.10 Kapan Harus STOP Selamanya

Ada kondisi di mana sistem sebaiknya dihentikan dan dievaluasi ulang dari awal:

1. **Performa live konsisten jauh di bawah backtest** setelah 3 bulan → model overfitting parah atau ada bug fundamental
2. **Market regime berubah drastis** (exchange baru dominan, regulasi, dll) → model butuh data baru dan retraining besar
3. **Total loss melebihi 20% modal yang dialokasikan** → stop, evaluasi, jangan average down
4. **Anda tidak paham kenapa model membuat keputusan** → tanda sistem terlalu kompleks tanpa fondasi yang kuat

Menghentikan sistem pada waktu yang tepat adalah **keputusan terbaik** yang bisa Anda buat.

---

## 7.11 Checklist Selesai Phase 7

- [ ] API key live sudah dikonfigurasi dengan benar (tanpa withdraw permission)
- [ ] IP whitelist aktif
- [ ] Modal live sudah ditentukan (amount yang sanggup hilang total)
- [ ] Sistem live berjalan minimal 2 minggu sebelum evaluasi
- [ ] Daily monitoring report aktif dan dikirim
- [ ] Kill-switch ditest sekali di live mode (dengan modal kecil)
- [ ] Scale-up protocol didokumentasikan dan dipatuhi
- [ ] Retraining scheduled aktif
- [ ] Incident response runbook tersedia
- [ ] Backup strategi: cara manual close posisi jika sistem crash

---

## 7.12 Summary Keseluruhan Sistem

```
┌────────────────────────────────────────────────────────────────┐
│              BTC 5M PREDICTION SYSTEM — ARCHITECTURE           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 0: Setup    →  Folder, config, dependencies              │
│  Phase 1: Data     →  Kline + OrderBook + AggTrade (live)       │
│  Phase 2: Features →  ~35 fitur (teknikal + microstructure)     │
│  Phase 3: Labels   →  Triple-barrier (realistic TP/SL)          │
│  Phase 4: Model    →  LightGBM + Walk-Forward validation        │
│  Phase 5: Backtest →  EV + Sharpe + Drawdown dengan fee riil    │
│  Phase 6: Paper    →  Testnet 4+ minggu, compare vs backtest    │
│  Phase 7: Live     →  Modal kecil, scale perlahan               │
│                                                                 │
│  Target akurasi realistis: 52-60% directional accuracy          │
│  Target EV: > 0 per trade setelah fee & slippage                │
│                                                                 │
│  ⚠️ Dokumen ini adalah panduan teknis, bukan jaminan profit.    │
│     Selalu test di testnet sebelum modal riil.                   │
└────────────────────────────────────────────────────────────────┘
```
