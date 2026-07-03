import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from loguru import logger
import requests
from dotenv import load_dotenv

from src.data_fetch import load_config, get_binance_client

def init_db_columns(db_path: str):
    """Pastikan kolom actual_return dan is_correct ada di tabel predictions."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "actual_return" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN actual_return REAL")
        logger.info("Added actual_return column to predictions table.")
    if "is_correct" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN is_correct INTEGER")
        logger.info("Added is_correct column to predictions table.")
        
    conn.commit()
    conn.close()

def send_telegram_summary(token: str, chat_id: str, message: str):
    """Kirim ringkasan laporan ke Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        logger.info("Report successfully sent to Telegram!")
    except Exception as e:
        logger.error(f"Failed to send Telegram report: {e}")

def evaluate_predictions(config_path: str = "config/config.yaml", send_tele: bool = True):
    from datetime import datetime, timezone, timedelta
    load_dotenv()
    config = load_config(config_path)
    db_path = config["logging"]["db_path"]
    symbol = config["binance"]["symbol"]
    threshold = config["trading"]["probability_threshold"]
    
    init_db_columns(db_path)
    
    # 1. Reset premature evaluations (where timestamp is within the last 10 minutes)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE predictions SET actual_return = NULL, is_correct = NULL WHERE created_at >= datetime('now', '-10 minutes')"
    )
    conn.commit()
    
    # Load predictions yang belum dievaluasi
    df_preds = pd.read_sql_query(
        "SELECT id, timestamp, proba_up, signal, direction, actual_return FROM predictions WHERE actual_return IS NULL",
        conn
    )
    conn.close()
    
    if df_preds.empty:
        logger.info("All predictions are already evaluated.")
        # Hitung metrics dari seluruh data di DB
        return show_summary_report(db_path, threshold, config, send_tele)
        
    logger.info(f"Found {len(df_preds)} predictions waiting for evaluation.")
    
    # 2. Ambil client Binance untuk fetch harga aktual
    client = get_binance_client(testnet=config["binance"].get("testnet", True))
    
    # Parse timestamps
    df_preds["dt"] = pd.to_datetime(df_preds["timestamp"], utc=True)
    min_dt = df_preds["dt"].min() - timedelta(minutes=15)
    max_dt = df_preds["dt"].max() + timedelta(minutes=15)
    
    # Fetch klines aktual untuk rentang tersebut
    logger.info(f"Fetching actual klines from {min_dt} to {max_dt}")
    try:
        # Konversi ke timestamp ms untuk REST API
        start_ms = int(min_dt.timestamp() * 1000)
        end_ms = int(max_dt.timestamp() * 1000)
        raw_klines = client.get_historical_klines(symbol, "5m", start_ms, end_ms)
    except Exception as e:
        logger.error(f"Failed to fetch klines from Binance: {e}")
        return
        
    if not raw_klines:
        logger.warning("No actual kline data returned from Binance REST API.")
        return
        
    # Build actual prices DataFrame
    # 0: Open time, 4: Close price
    actual_data = []
    for k in raw_klines:
        actual_data.append({
            "open_time": pd.to_datetime(k[0], unit="ms", utc=True),
            "close": float(k[4])
        })
    df_actual = pd.DataFrame(actual_data).set_index("open_time").sort_index()
    
    # 3. Hitung outcome untuk masing-masing prediction
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    updated_count = 0
    now_utc = datetime.now(timezone.utc)
    for idx, row in df_preds.iterrows():
        pred_id = int(row["id"])
        pred_time = row["dt"]
        direction = row["direction"] or "NEUTRAL"
        
        # Entry price adalah harga close lilin pred_time (atau open lilin berikutnya)
        # Target penutupan prediksi adalah close lilin berikutnya (pred_time + 5m)
        entry_time = pred_time
        exit_time = pred_time + timedelta(minutes=5)
        
        # Hanya evaluasi jika lilin exit sudah fully closed (sekarang >= exit_time + 5m)
        if now_utc < exit_time + timedelta(minutes=5):
            continue
            
        if entry_time in df_actual.index and exit_time in df_actual.index:
            entry_price = df_actual.loc[entry_time, "close"]
            exit_price = df_actual.loc[exit_time, "close"]
            
            actual_ret = (exit_price - entry_price) / entry_price
            
            # Evaluasi is_correct berdasarkan direction
            if direction == "UP":
                is_correct = int(actual_ret > 0)
            elif direction == "DOWN":
                is_correct = int(actual_ret < 0)
            else: # NEUTRAL
                is_correct = 0
            
            cursor.execute(
                "UPDATE predictions SET actual_return = ?, is_correct = ? WHERE id = ?",
                (float(actual_ret), int(is_correct), pred_id)
            )
            updated_count += 1
            
    conn.commit()
    conn.close()
    
    logger.info(f"Successfully evaluated and updated {updated_count} predictions in database.")
    
    # 4. Tampilkan dan kirim report
    return show_summary_report(db_path, threshold, config, send_tele)

def show_summary_report(db_path: str, threshold: float, config: dict, send_tele: bool = True):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM predictions WHERE actual_return IS NOT NULL", conn)
    df_trades = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    
    if df.empty:
        logger.info("No evaluated predictions found in database to summarize.")
        return
        
    df["dt"] = pd.to_datetime(df["timestamp"])
    total_preds = len(df)
    
    # Perhitungan akurasi terarah
    # Sinyal aktif (UP atau DOWN)
    active_signals = df[df["signal"] == 1]
    
    total_active = len(active_signals)
    correct_active = active_signals["is_correct"].sum()
    sig_accuracy = correct_active / total_active if total_active > 0 else 0.0
    
    up_signals = active_signals[active_signals["direction"] == "UP"]
    down_signals = active_signals[active_signals["direction"] == "DOWN"]
    
    up_accuracy = up_signals["is_correct"].mean() if not up_signals.empty else 0.0
    down_accuracy = down_signals["is_correct"].mean() if not down_signals.empty else 0.0
    
    # Breakdown per Confidence Bin
    bins = [0.0, 0.4, 0.48, 0.52, 0.6, 0.7, 0.8, 1.0]
    df["conf_bin"] = pd.cut(df["proba_up"], bins=bins)
    bin_report = []
    for grp, sub_df in df.groupby("conf_bin", observed=False):
        if not sub_df.empty:
            # Hitung akurasi sinyal terarah di bin ini
            # Di bawah 0.48 adalah sinyal DOWN, di atas 0.52 adalah sinyal UP
            active_in_bin = sub_df[sub_df["signal"] == 1]
            bin_acc = active_in_bin["is_correct"].mean() if not active_in_bin.empty else 0.0
            bin_report.append(f"  • {grp}: {len(sub_df)} sampel | Akurasi Sinyal: {bin_acc:.1%} ({len(active_in_bin)} sinyal)")
            
    bin_str = "\n".join(bin_report)
    
    # Trade Stats
    total_trades = len(df_trades)
    win_rate = (df_trades["net_pnl"] > 0).mean() if total_trades > 0 else 0.0
    net_pnl = df_trades["net_pnl"].sum() if total_trades > 0 else 0.0
    
    report_msg = (
        f"📊 <b>LAPORAN PERFORMA BOT SIMULASI</b>\n"
        f"Symbol: {config['binance']['symbol']} | Threshold: {threshold:.2f}\n"
        f"Periode: {df['dt'].min().strftime('%Y-%m-%d %H:%M')} s/d {df['dt'].max().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"<b>1. Prediksi Arah (Kline 5m)</b>\n"
        f"• Total Prediksi Terbentuk: {total_preds}\n"
        f"• Total Sinyal Aktif: {total_active}\n"
        f"• Akurasi Sinyal Aktif: {sig_accuracy:.2%}\n"
        f"  - Akurasi UP (Long): {up_accuracy:.2%} ({len(up_signals)} sinyal)\n"
        f"  - Akurasi DOWN (Short): {down_accuracy:.2%} ({len(down_signals)} sinyal)\n\n"
        f"<b>2. Distribusi Keyakinan (Confidence Bin)</b>\n"
        f"{bin_str}\n\n"
        f"<b>3. Performa Eksekusi Trading Simulasi</b>\n"
        f"• Total Trades: {total_trades}\n"
        f"• Win Rate: {win_rate:.2%}\n"
        f"• Net PnL: ${net_pnl:,.2f}\n"
    )
    
    print("\n" + "="*50)
    print("LIVE PREDICTIONS REPORT")
    print("="*50)
    print(report_msg.replace("<b>", "").replace("</b>", ""))
    print("="*50)
    
    bot_token = os.getenv("BOT_TOKEN")
    bot_target_id = os.getenv("BOT_TARGET_ID")
    if send_tele and bot_token and bot_target_id:
        send_telegram_summary(bot_token, str(bot_target_id), report_msg)
        
    return df

if __name__ == "__main__":
    evaluate_predictions()
