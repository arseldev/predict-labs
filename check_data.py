"""
check_data.py — Data Quality & Statistics Report

Jalankan kapan saja untuk melihat seberapa banyak dan lengkap data yang sudah terkumpul.

Usage:
    python check_data.py
    python check_data.py --verbose
    python check_data.py --gaps 5m      # Cek gap pada timeframe tertentu
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>", level="INFO")

TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
}

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")


def fmt_size(bytes_: int) -> str:
    """Format bytes ke MB/GB."""
    if bytes_ < 1e6:
        return f"{bytes_ / 1e3:.1f} KB"
    elif bytes_ < 1e9:
        return f"{bytes_ / 1e6:.1f} MB"
    return f"{bytes_ / 1e9:.2f} GB"


def check_klines(verbose: bool = False) -> dict:
    """Cek semua kline data per timeframe."""
    results = {}

    logger.info("\n" + "-" * 65)
    logger.info("KLINE DATA")
    logger.info("-" * 65)

    for tf, tf_min in TIMEFRAME_MINUTES.items():
        tf_dir = RAW_PATH / "klines" / tf
        results[tf] = {"ok": False}

        # Cari semua parquet file di folder ini
        parquet_files = list(tf_dir.glob("*.parquet")) if tf_dir.exists() else []
        if not parquet_files:
            logger.info(f"  {tf:>5s} │ ❌ No data found at {tf_dir}")
            continue

        try:
            dfs = []
            for f in sorted(parquet_files):
                df_part = pd.read_parquet(f)
                dfs.append(df_part)
            df = pd.concat(dfs)
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

            n_rows = len(df)
            t_from = df.index.min()
            t_to = df.index.max()
            duration_days = (t_to - t_from).total_seconds() / 86400
            expected_candles = int(duration_days * 24 * 60 / tf_min)
            coverage_pct = (n_rows / expected_candles * 100) if expected_candles > 0 else 0
            total_size = sum(f.stat().st_size for f in parquet_files)

            # Deteksi gaps
            expected_delta = pd.Timedelta(minutes=tf_min)
            actual_deltas = df.index.to_series().diff().dropna()
            gaps = actual_deltas[actual_deltas > expected_delta * 1.5]
            n_gaps = len(gaps)

            status = "[OK]" if coverage_pct >= 95 else ("[WARN]" if coverage_pct >= 80 else "[MISS]")
            logger.info(
                f"  {tf:>5s} │ {status} {n_rows:>8,} candles │ "
                f"{str(t_from)[:16]} → {str(t_to)[:16]} │ "
                f"{coverage_pct:.1f}% coverage │ {fmt_size(total_size)}"
            )
            if n_gaps > 0:
                logger.info(f"        | [!] {n_gaps} gap(s) detected")
                if verbose:
                    for gap_time, gap_dur in gaps.head(5).items():
                        logger.info(f"        │    Gap at {str(gap_time)[:16]}: {gap_dur}")

            results[tf] = {
                "ok": True,
                "rows": n_rows,
                "from": t_from,
                "to": t_to,
                "coverage_pct": coverage_pct,
                "gaps": n_gaps,
                "size_bytes": total_size
            }
        except Exception as e:
            logger.info(f"  {tf:>5s} │ ❌ Error reading data: {e}")
            logger.info(f"  {tf:>5s} │ [MISS] Error reading data: {e}")

    return results


def check_orderbook(verbose: bool = False) -> dict:
    """Cek order book snapshot data."""
    logger.info("\n" + "-" * 65)
    logger.info("ORDER BOOK DATA")
    logger.info("-" * 65)

    ob_dir = RAW_PATH / "orderbook"
    ob_files = sorted(ob_dir.glob("orderbook_*.parquet")) if ob_dir.exists() else []

    if not ob_files:
        logger.info("  [MISS] No orderbook data found. Run: python collect_data.py --mode stream")
        return {}

    total_rows = 0
    total_size = 0
    dates = []
    for f in ob_files:
        try:
            df = pd.read_parquet(f)
            total_rows += len(df)
            total_size += f.stat().st_size
            dates.append(f.stem.replace("orderbook_", ""))
        except Exception:
            pass

    if dates:
        logger.info(f"  Snapshots : {total_rows:>8,} rows in {len(ob_files)} files")
        logger.info(f"  Date range: {min(dates)} → {max(dates)}")
        logger.info(f"  Size      : {fmt_size(total_size)}")

        # Cek frekuensi snapshot
        if verbose and ob_files:
            latest_file = max(ob_files)
            df_latest = pd.read_parquet(latest_file)
            if len(df_latest) > 1:
                avg_interval = df_latest.index.to_series().diff().median()
                logger.info(f"  Avg interval: {avg_interval}")

        # Data per hari
        logger.info(f"  {'Tanggal':>12s} │ {'Snapshots':>10s}")
        for f in ob_files[-7:]:  # Tampilkan 7 hari terakhir
            try:
                n = len(pd.read_parquet(f))
                date = f.stem.replace("orderbook_", "")
                logger.info(f"  {date:>12s} │ {n:>10,}")
            except Exception:
                pass

    return {"total_snapshots": total_rows, "files": len(ob_files), "size_bytes": total_size}


def check_trades(verbose: bool = False) -> dict:
    """Cek aggTrade data."""
    logger.info("\n" + "-" * 65)
    logger.info("AGGTRADE DATA")
    logger.info("-" * 65)

    tr_dir = RAW_PATH / "trades"
    tr_files = sorted(tr_dir.glob("trades_*.parquet")) if tr_dir.exists() else []

    if not tr_files:
        logger.info("  [MISS] No aggTrade data found. Run: python collect_data.py --mode stream")
        return {}

    total_rows = 0
    total_size = 0
    dates = []
    for f in tr_files:
        try:
            df = pd.read_parquet(f)
            total_rows += len(df)
            total_size += f.stat().st_size
            dates.append(f.stem.replace("trades_", ""))
        except Exception:
            pass

    logger.info(f"  Records   : {total_rows:>8,} trades in {len(tr_files)} files")
    if dates:
        logger.info(f"  Date range: {min(dates)} → {max(dates)}")
    logger.info(f"  Size      : {fmt_size(total_size)}")

    # Data per hari
    logger.info(f"  {'Tanggal':>12s} │ {'Trades':>10s}")
    for f in tr_files[-7:]:
        try:
            n = len(pd.read_parquet(f))
            date = f.stem.replace("trades_", "")
            logger.info(f"  {date:>12s} │ {n:>10,}")
        except Exception:
            pass

    return {"total_records": total_rows, "files": len(tr_files), "size_bytes": total_size}


def check_gaps_detail(timeframe: str):
    """Tampilkan semua gap detail untuk satu timeframe."""
    tf_min = TIMEFRAME_MINUTES.get(timeframe)
    if not tf_min:
        logger.error(f"Unknown timeframe: {timeframe}. Valid: {list(TIMEFRAME_MINUTES.keys())}")
        return

    tf_dir = RAW_PATH / "klines" / timeframe
    parquet_files = list(tf_dir.glob("*.parquet")) if tf_dir.exists() else []
    if not parquet_files:
        logger.info(f"No data for {timeframe}")
        return

    dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
    df = pd.concat(dfs)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    expected_delta = pd.Timedelta(minutes=tf_min)
    actual_deltas = df.index.to_series().diff().dropna()
    gaps = actual_deltas[actual_deltas > expected_delta * 1.5]

    logger.info(f"\n{'-'*65}")
    logger.info(f"GAP DETAIL -- {timeframe} ({len(gaps)} gaps)")
    logger.info(f"{'-'*65}")
    if gaps.empty:
        logger.info("  [OK] No gaps found!")
    else:
        for gap_time, gap_dur in gaps.items():
            missing = int(gap_dur.total_seconds() / 60 / tf_min) - 1
            logger.info(f"  {str(gap_time)[:19]} : gap {gap_dur} (~{missing} missing candles)")


def check_readiness() -> None:
    """Hitung apakah data sudah cukup untuk mulai training model."""
    logger.info("\n" + "-" * 65)
    logger.info("READINESS CHECK (untuk training model)")
    logger.info("-" * 65)

    checks = []

    # Cek kline 5m — minimal 60 hari
    p5m_dir = RAW_PATH / "klines" / "5m"
    p5m_files = list(p5m_dir.glob("*.parquet")) if p5m_dir.exists() else []
    if p5m_files:
        try:
            df = pd.concat([pd.read_parquet(f) for f in p5m_files])
            days = (df.index.max() - df.index.min()).days
            ok = days >= 60
            checks.append((ok, f"5m kline : {days} hari {'[OK] cukup' if ok else '[--] butuh min 60 hari'}"))
        except Exception:
            checks.append((False, "5m kline : error reading"))
    else:
        checks.append((False, "5m kline : [MISS] belum ada data"))

    # Cek kline 1h — untuk context
    p1h_dir = RAW_PATH / "klines" / "1h"
    p1h_files = list(p1h_dir.glob("*.parquet")) if p1h_dir.exists() else []
    if p1h_files:
        try:
            df = pd.concat([pd.read_parquet(f) for f in p1h_files])
            days = (df.index.max() - df.index.min()).days
            ok = days >= 30
            checks.append((ok, f"1h kline : {days} hari {'[OK] cukup' if ok else '[--] butuh min 30 hari'}"))
        except Exception:
            checks.append((False, "1h kline : error"))
    else:
        checks.append((False, "1h kline : [MISS] belum ada data"))

    # Cek orderbook
    ob_dir = RAW_PATH / "orderbook"
    ob_files = list(ob_dir.glob("*.parquet")) if ob_dir.exists() else []
    ob_days = len(ob_files)
    ok_ob = ob_days >= 7
    checks.append((ok_ob, f"Orderbook: {ob_days} hari {'[OK]' if ok_ob else '[--] idealnya 7+ hari'}"))

    # Cek trades
    tr_dir = RAW_PATH / "trades"
    tr_files = list(tr_dir.glob("*.parquet")) if tr_dir.exists() else []
    tr_days = len(tr_files)
    ok_tr = tr_days >= 7
    checks.append((ok_tr, f"AggTrade : {tr_days} hari {'[OK]' if ok_tr else '[--] idealnya 7+ hari'}"))

    for ok, msg in checks:
        icon = "[OK]" if ok else "[--]"
        logger.info(f"  {icon} {msg}")

    all_ok = all(ok for ok, _ in checks)
    logger.info("")
    if all_ok:
        logger.info("  >> Data sudah cukup! Siap untuk mulai Phase 2 (Feature Engineering).")
    else:
        logger.info("  >> Belum cukup data. Lanjutkan running: python collect_data.py --mode stream")


def main():
    parser = argparse.ArgumentParser(
        description="Cek kualitas dan statistik data yang sudah terkumpul",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Tampilkan detail lebih banyak")
    parser.add_argument("--gaps", metavar="TIMEFRAME", help="Tampilkan detail gap untuk timeframe (1m/5m/15m/1h)")
    args = parser.parse_args()

    logger.info("\n" + "=" * 65)
    logger.info("predict-labs -- DATA QUALITY REPORT")
    logger.info(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    if args.gaps:
        check_gaps_detail(args.gaps)
        return

    check_klines(verbose=args.verbose)
    check_orderbook(verbose=args.verbose)
    check_trades(verbose=args.verbose)
    check_readiness()

    logger.info("")


if __name__ == "__main__":
    main()
