"""
collect_data.py — Entry Point Pengumpulan Data BTC 5m

Usage:
    # Download 90 hari data historis kline semua timeframe
    python collect_data.py --mode historical --days 90

    # Jalankan live streaming (kline + orderbook + aggTrade) - biarkan jalan terus!
    python collect_data.py --mode stream

    # Download historical LALU langsung mulai streaming (recommended)
    python collect_data.py --mode all --days 90

⚠️  Tidak butuh API Key untuk historical bulk download dan public stream.
     API Key hanya untuk mode 'paper' dan 'live' trading.
"""

import argparse
import sys
import time
import signal
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import threading
import pandas as pd

load_dotenv()

# Setup logger format yang bersih
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/collector.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
    rotation="100 MB",
    retention="30 days"
)

from src.data_fetch import load_config, get_binance_client, fetch_all_historical_klines
from src.data_stream import StorageManager, StreamManager


def run_historical(config: dict, days_back: int):
    """
    Download data kline historis untuk semua timeframe dari data.binance.vision (bulk download).
    
    Menggunakan bulk download dari data.binance.vision — tidak butuh API key,
    tidak butuh koneksi ke api.binance.com (bypass SSL/block issue).
    Data bulan berjalan diambil via REST jika tersedia, atau dilewati.
    """
    from datetime import datetime, timezone, timedelta
    from src.data_fetch import download_bulk_klines

    symbol = config["binance"]["symbol"]
    timeframes = [config["data"]["timeframes"]["primary"]] + config["data"]["timeframes"]["context"]
    save_dir_base = config["data"].get("raw_path", "data/raw")

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days_back)

    logger.info(f"=" * 60)
    logger.info(f"HISTORICAL DATA DOWNLOAD (via data.binance.vision)")
    logger.info(f"Symbol    : {symbol}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Days back : {days_back}")
    logger.info(f"Period    : {start_date.strftime('%Y-%m-%d')} -> {now.strftime('%Y-%m-%d')}")
    logger.info(f"=" * 60)

    for tf in timeframes:
        logger.info(f"\nDownloading {tf} klines ({days_back} days)...")
        save_dir = Path(save_dir_base) / "klines" / tf
        save_dir.mkdir(parents=True, exist_ok=True)

        all_dfs = []

        # Loop bulan per bulan
        y, m = start_date.year, start_date.month
        end_y, end_m = now.year, now.month

        while (y, m) < (end_y, end_m):
            parquet_path = str(save_dir / f"{symbol}_{tf}_{y}_{m:02d}.parquet")
            result = download_bulk_klines(symbol, tf, y, m, parquet_path)
            if result and Path(result).exists():
                df_month = pd.read_parquet(result)
                all_dfs.append(df_month)
                logger.info(f"  {y}-{m:02d}: {len(df_month):,} candles downloaded")
            
            m += 1
            if m > 12:
                m = 1
                y += 1

        # Gabungkan semua bulan
        if all_dfs:
            combined = pd.concat(all_dfs)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            combined = combined[combined.index >= pd.Timestamp(start_date)]

            combined_path = save_dir / "data.parquet"
            combined.to_parquet(combined_path, compression="snappy")
            logger.info(
                f"  [OK] {tf}: {len(combined):,} candles total | "
                f"{str(combined.index.min())[:16]} -> {str(combined.index.max())[:16]}"
            )
        else:
            logger.warning(f"  [WARN] {tf}: No data downloaded")

    logger.info("\n" + "=" * 60)
    logger.info("[OK] Historical download complete!")
    logger.info("     Run: python collect_data.py --mode stream")
    logger.info("     To start live streaming orderbook + trades.")
    logger.info("=" * 60)


def run_stream(config: dict):
    """
    Jalankan WebSocket stream secara terus-menerus:
    - Kline (semua timeframe: 1m, 5m, 15m, 1h) → simpan ke data/raw/klines/
    - Order Book Depth (top 20) → simpan snapshot per 30 detik ke data/raw/orderbook/
    - AggTrade → simpan ke data/raw/trades/

    Biarkan script ini jalan terus — semakin lama berjalan, semakin banyak data terkumpul.
    Tekan Ctrl+C untuk berhenti dengan graceful shutdown.
    """
    symbol = config["binance"]["symbol"]
    raw_path = config["data"].get("raw_path", "data/raw")

    logger.info("=" * 60)
    logger.info("📡 LIVE DATA STREAMING")
    logger.info(f"Symbol      : {symbol}")
    logger.info(f"Storage path: {raw_path}")
    logger.info(f"Streams     : klines (1m/5m/15m/1h) + depth + aggTrade")
    logger.info("=" * 60)
    logger.info("Tekan Ctrl+C untuk berhenti.")

    storage = StorageManager(raw_path=raw_path)
    stream_manager = StreamManager(
        symbol=symbol,
        config=config,
        storage=storage,
        on_candle_closed=None  # Tidak ada prediksi, hanya collect data
    )

    # Graceful shutdown handler
    def _shutdown(sig, frame):
        logger.info("\n⛔ Shutdown signal received. Stopping streams...")
        stream_manager.stop()
        # Print final stats
        stats = storage.get_stats()
        _print_storage_stats(stats)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stream_manager.start()

    # Periodic progress report setiap 10 menit
    report_interval = 600
    last_report = time.time()

    logger.info("✅ Streaming started! Data sedang dikumpulkan...")

    while True:
        time.sleep(30)
        now = time.time()

        if now - last_report >= report_interval:
            stream_stats = stream_manager.get_stats()
            storage_stats = storage.get_stats()
            logger.info(
                f"[PROGRESS] Klines closed: {stream_stats['klines_closed']} | "
                f"Depth snapshots: {stream_stats['depth_snapshots_saved']} | "
                f"Trades received: {stream_stats['trades_received']}"
            )
            last_report = now


def _print_storage_stats(stats: dict):
    """Print statistik data yang tersimpan ke console."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 DATA STORAGE STATS")
    logger.info("=" * 60)

    for tf in ["1m", "5m", "15m", "1h"]:
        key = f"klines_{tf}"
        if key in stats and stats[key]["rows"] > 0:
            s = stats[key]
            logger.info(
                f"Kline {tf:>4s}: {s['rows']:>8,} rows | "
                f"{s.get('from','?')[:16]} → {s.get('to','?')[:16]} | "
                f"{s.get('size_mb', 0):.1f} MB"
            )
        else:
            logger.info(f"Kline {tf:>4s}: (no data)")

    ob = stats.get("orderbook", {})
    logger.info(f"Orderbook   : {ob.get('snapshots', 0):>8,} snapshots in {ob.get('files', 0)} files")

    tr = stats.get("trades", {})
    logger.info(f"AggTrade    : {tr.get('records', 0):>8,} records in {tr.get('files', 0)} files")
    logger.info("=" * 60)


def run_all(config: dict, days_back: int):
    """Download historical lalu langsung mulai streaming."""
    run_historical(config, days_back)
    logger.info("\n🚀 Starting live stream after historical download...")
    time.sleep(2)
    run_stream(config)


def main():
    parser = argparse.ArgumentParser(
        description="BTC 5m Data Collector — kumpulkan kline, orderbook, dan trade data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python collect_data.py --mode historical --days 90
  python collect_data.py --mode stream
  python collect_data.py --mode all --days 90
        """
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "stream", "all"],
        required=True,
        help=(
            "historical: download bulk kline data saja | "
            "stream: live streaming kline+depth+trade | "
            "all: historical dulu, lalu stream"
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Jumlah hari historis yang akan didownload (default: 90)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path ke config.yaml"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Override testnet=False agar WebSocket pakai public stream (bukan testnet)
    # Testnet WebSocket sering tidak reliable untuk data collection
    config["binance"]["testnet"] = False
    logger.info(f"🌐 Using PUBLIC Binance stream (no API key required for data collection)")

    if args.mode == "historical":
        run_historical(config, args.days)
    elif args.mode == "stream":
        run_stream(config)
    elif args.mode == "all":
        run_all(config, args.days)


if __name__ == "__main__":
    main()
