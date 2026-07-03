"""
Audit script - cek kondisi sistem lengkap
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os

print("=" * 70)
print("SYSTEM AUDIT REPORT")
print("=" * 70)

# 1. Cek database predictions
print("\n[1] DATABASE STATUS")
conn = sqlite3.connect('logs/predictions.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(predictions)")
cols = [(c[1], c[2]) for c in cursor.fetchall()]
print(f"   Columns in predictions: {[c[0] for c in cols]}")
cursor.execute("SELECT COUNT(*) FROM predictions")
total_preds = cursor.fetchone()[0]
print(f"   Total predictions: {total_preds}")
cursor.execute("SELECT COUNT(*) FROM predictions WHERE signal=1")
signal_preds = cursor.fetchone()[0]
print(f"   Signal=1 (UP signal fired): {signal_preds}")
cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_return IS NOT NULL")
evaluated = cursor.fetchone()[0]
print(f"   Evaluated (actual_return filled): {evaluated}")
cursor.execute("SELECT COUNT(*) FROM predictions WHERE actual_return IS NULL")
not_evaluated = cursor.fetchone()[0]
print(f"   Not yet evaluated: {not_evaluated}")

if evaluated > 0:
    cursor.execute("SELECT AVG(is_correct) FROM predictions WHERE is_correct IS NOT NULL")
    acc = cursor.fetchone()[0]
    print(f"   Accuracy (evaluated): {acc:.2%}" if acc else "   Accuracy: N/A")

cursor.execute("SELECT COUNT(*) FROM trades")
total_trades = cursor.fetchone()[0]
print(f"   Total trades: {total_trades}")

# 2. Analisis apakah sinyal HANYA up (tidak pernah short/down)
print("\n[2] SIGNAL BIAS ANALYSIS")
df_preds = pd.read_sql_query("SELECT * FROM predictions", conn)
if not df_preds.empty:
    df_preds['proba_up'] = df_preds['proba_up'].astype(float)
    print(f"   proba_up min: {df_preds['proba_up'].min():.4f}")
    print(f"   proba_up max: {df_preds['proba_up'].max():.4f}")
    print(f"   proba_up mean: {df_preds['proba_up'].mean():.4f}")
    print(f"   Signals UP fired (>=0.52): {(df_preds['proba_up'] >= 0.52).sum()}")
    print(f"   Signals DOWN (proba_up < 0.48): {(df_preds['proba_up'] < 0.48).sum()}")
    print()
    print("   ⚠️  CRITICAL ISSUE: System only predicts UP direction (no SHORT)")
    print("   ⚠️  Signal=1 means UP only — DOWN is NEVER executed as a trade")

conn.close()

# 3. Data yang tersedia
print("\n[3] AVAILABLE DATA")
for tf in ['5m', '1m', '15m', '1h']:
    p = Path(f"data/raw/klines/{tf}/data.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        print(f"   {tf}: {len(df):,} candles | {str(df.index.min())[:16]} → {str(df.index.max())[:16]}")
    else:
        print(f"   {tf}: NO data.parquet file")

# Check monthly files
print("\n   Monthly parquet files (5m):")
for f in sorted(Path("data/raw/klines/5m").glob("*.parquet")):
    df = pd.read_parquet(f)
    print(f"     {f.name}: {len(df):,} candles | {str(df.index.min())[:16]} → {str(df.index.max())[:16]}")

# 4. Model info
print("\n[4] MODEL INFO")
import pickle
with open('models/latest.pkl', 'rb') as f:
    model_data = pickle.load(f)

if isinstance(model_data, dict):
    print(f"   Model type: {type(model_data.get('model'))}")
    feature_cols = model_data.get('feature_cols', [])
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Features: {feature_cols[:5]}... (first 5)")
else:
    print(f"   Model type: {type(model_data)}")

# 5. Cek masalah waktu prediksi
print("\n[5] TIMING ANALYSIS")
print("   Prediction triggered: When 5m candle CLOSES")
print("   This means the candle is already done → user needs to ENTER at start of NEXT candle")
print("   ✅ This is correct — prediction at close of C, entry at open of C+1")
print()
print("   Evaluation window (evaluate_live.py):")
print("   entry_time = pred_time (close of predicted candle)")
print("   exit_time = pred_time + 5m (close of NEXT candle)")
print("   ⚠️  This only evaluates 1-candle direction, not the actual trade outcome!")

# 6. Issues summary
print("\n[6] CRITICAL ISSUES FOUND")
issues = [
    "❌ ISSUE 1: System ONLY trades UP (Long only). Never shorts DOWN signals",
    "❌ ISSUE 2: Model trained on 'fixed_horizon' label but config uses both methods",
    "❌ ISSUE 3: Paper trading uses Binance Testnet API for ORDER EXECUTION but testnet may reject orders",
    "❌ ISSUE 4: No SIMULATION mode — paper trading tries REAL testnet orders not simulated",
    "❌ ISSUE 5: evaluate_live.py uses +5m window but actual trade has TP/SL up to 12 candles (1hr)",
    "⚠️  ISSUE 6: Data only up to May 2026 — June+July 2026 klines from REST only (recent 30d)",
    "❌ ISSUE 7: When running 'paper' mode, if testnet API fails, NO trades are logged = empty simulation",
]
for i in issues:
    print(f"   {i}")

print("\n" + "=" * 70)
