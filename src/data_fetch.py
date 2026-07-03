"""
data_fetch.py — Historical Data Fetcher

Fungsi utama:
- load_config(): Load config.yaml + merge env variables
- get_binance_client(): Inisialisasi Binance client (testnet/live)
- fetch_klines_rest(): Ambil data kline historis via REST API (< 90 hari)
- download_bulk_klines(): Download kline dari data.binance.vision (ZIP/CSV)
- fetch_all_historical_klines(): Orchestrator historis (REST + bulk)
- fetch_depth_snapshot(): Snapshot order book satu kali via REST
"""

import os
import re
import time
import urllib.request
import urllib.error
import zipfile
import io
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dotenv import load_dotenv
from binance.client import Client
from loguru import logger
from datetime import datetime, timezone, timedelta

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

NUMERIC_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote"
]

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load config.yaml dan merge dengan environment variables (.env).
    Mendukung penggantian placeholder format ${VAR_NAME}.
    """
    load_dotenv()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        content = f.read()
        
    # Replace placeholders like ${BINANCE_API_KEY}
    matches = re.findall(r"\$\{(\w+)\}", content)
    for match in matches:
        env_val = os.getenv(match, "")
        content = content.replace(f"${{{match}}}", env_val)
        
    config = yaml.safe_load(content)
    
    # Pastikan dictionary binance terbentuk
    if "binance" not in config:
        config["binance"] = {}
        
    # Mapping default API key dari env jika tidak diset lewat yaml placeholder
    testnet = config["binance"].get("testnet", True)
    if testnet:
        config["binance"]["api_key"] = config["binance"].get("api_key") or os.getenv("BINANCE_TESTNET_API_KEY", os.getenv("BINANCE_API_KEY", ""))
        config["binance"]["api_secret"] = config["binance"].get("api_secret") or os.getenv("BINANCE_TESTNET_API_SECRET", os.getenv("BINANCE_API_SECRET", ""))
    else:
        config["binance"]["api_key"] = config["binance"].get("api_key") or os.getenv("BINANCE_API_KEY", "")
        config["binance"]["api_secret"] = config["binance"].get("api_secret") or os.getenv("BINANCE_API_SECRET", "")
        
    return config

def get_binance_client(testnet: bool = True, api_key: str = None, api_secret: str = None) -> Client:
    """
    Inisialisasi Binance Client.
    Jika api_key/api_secret None, akan meload dari .env berdasarkan status testnet.
    Mode public (api_key='') bisa dipakai untuk fetch kline historis tanpa auth.
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    load_dotenv()
    if api_key is None or api_secret is None:
        if testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", os.getenv("BINANCE_API_KEY", ""))
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", os.getenv("BINANCE_API_SECRET", ""))
        else:
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")

    # Pass verify=False via requests_params agar ping() di __init__ tidak SSL error
    # Ini diperlukan di lingkungan dengan proxy/SSL interception (umum di Windows)
    client = Client(
        api_key,
        api_secret,
        testnet=testnet,
        requests_params={"verify": False}
    )
    return client

def fetch_klines_rest(
    client: Client,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_str: str = "90 days ago UTC",
    end_str: str = None,
    save_path: str = None
) -> pd.DataFrame:
    """
    Ambil data kline via REST API untuk periode pendek (< 90 hari).
    """
    logger.info(f"Fetching {symbol} {interval} klines from {start_str} to {end_str or 'now'} via REST API")
    
    try:
        raw = client.get_historical_klines(symbol, interval, start_str, end_str)
    except Exception as e:
        logger.error(f"Error calling get_historical_klines: {e}")
        raise e
        
    if not raw:
        logger.warning(f"No kline data returned from REST API for {symbol} {interval}")
        return pd.DataFrame()
        
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    df.drop(columns=["ignore"], inplace=True)
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in NUMERIC_COLS:
        df[col] = df[col].astype("float64")
    df["trades"] = df["trades"].astype("int64")
    
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, compression="snappy")
        logger.info(f"Saved {len(df)} rows of REST data to {save_path}")
        
    return df

def download_bulk_klines(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    save_path: str
) -> str:
    """
    Download kline historis panjang dari data.binance.vision (bulk CSV).
    Sangat efisien untuk data historis > 90 hari.
    """
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"
    logger.info(f"Downloading bulk klines from {url}")
    
    # Check if parquet file already exists
    if Path(save_path).exists():
        logger.info(f"File {save_path} already exists. Skipping bulk download.")
        return save_path
        
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
            
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
            csv_filenames = [name for name in zip_ref.namelist() if name.endswith(".csv")]
            if not csv_filenames:
                raise ValueError("No CSV file found in the ZIP archive")
            csv_filename = csv_filenames[0]
            csv_content = zip_ref.read(csv_filename).decode("utf-8")
            
        df = pd.read_csv(io.StringIO(csv_content), header=None)
        
        # Mapping kolom berdasarkan jumlah kolom di CSV
        if df.shape[1] == 12:
            df.columns = KLINE_COLUMNS
            df.drop(columns=["ignore"], inplace=True)
        elif df.shape[1] == 11:
            df.columns = KLINE_COLUMNS[:-1]
        else:
            cols = KLINE_COLUMNS[:df.shape[1]]
            df.columns = cols
            if "ignore" in df.columns:
                df.drop(columns=["ignore"], inplace=True)

        # Deteksi unit timestamp: Binance bulk CSV mungkin pakai ms atau us
        # Jika nilai open_time terlalu besar untuk ms (> year 3000), kemungkinan us
        sample_ts = int(df["open_time"].iloc[0])
        # year 3000 dalam ms = 32503680000000, kalau lebih dari ini berarti us
        if sample_ts > 32503680000000:
            ts_unit = "us"
        else:
            ts_unit = "ms"

        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit=ts_unit, utc=True)
        if "close_time" in df.columns:
            df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit=ts_unit, utc=True)
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = df[col].astype("float64")
        if "trades" in df.columns:
            df["trades"] = df["trades"].astype("int64")
            
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)
        
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, compression="snappy")
        logger.info(f"Successfully processed and saved bulk monthly klines to {save_path}")
        return save_path
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.warning(f"Bulk data not found (404) for {symbol} {interval} {year}-{month:02d}. Skipping.")
            return ""
        else:
            logger.error(f"HTTP Error {e.code} during bulk download from {url}")
            raise e
    except Exception as e:
        logger.error(f"Error downloading bulk klines: {e}")
        raise e

def fetch_all_historical_klines(
    client: Client,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    days_back: int = 90,
    save_dir: str = "data/raw/klines"
) -> pd.DataFrame:
    """
    Orchestrator untuk mengumpulkan kline data historis sesuai timeframe & range hari.
    """
    logger.info(f"Orchestrating historical kline fetch for {symbol} {interval} ({days_back} days back)")
    
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days_back)
    
    if days_back <= 90:
        start_str = f"{days_back} days ago UTC"
        save_path = Path(save_dir) / f"{symbol}_{interval}_historical.parquet"
        df = fetch_klines_rest(client, symbol, interval, start_str, save_path=str(save_path))
        return df
        
    # expanding window: older data via bulk monthly, recent via REST
    all_dfs = []
    
    # Loop dari start_date bulan per bulan sampai bulan ini
    current_year_month = (start_date.year, start_date.month)
    end_year_month = (now.year, now.month)
    
    y, m = current_year_month
    while (y, m) < end_year_month:
        month_save_path = Path(save_dir) / f"{symbol}_{interval}_{y}_{m:02d}.parquet"
        p = download_bulk_klines(symbol, interval, y, m, str(month_save_path))
        if p and Path(p).exists():
            df_month = pd.read_parquet(p)
            all_dfs.append(df_month)
            
        m += 1
        if m > 12:
            m = 1
            y += 1
            
    # Ambil sisa bulan berjalan via REST
    rest_start_str = f"{now.year}-{now.month:02d}-01 00:00:00 UTC"
    df_rest = fetch_klines_rest(client, symbol, interval, rest_start_str)
    if not df_rest.empty:
        all_dfs.append(df_rest)
        
    if not all_dfs:
        logger.warning("No data files fetched during bulk and REST orchestration")
        return pd.DataFrame()
        
    # Combine & Dedup
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
    combined_df.sort_index(inplace=True)
    
    # Filter to exact range
    combined_df = combined_df[combined_df.index >= start_date]
    
    combined_save_path = Path(save_dir) / f"{symbol}_{interval}_combined.parquet"
    combined_save_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(combined_save_path, compression="snappy")
    logger.info(f"Saved orchestrated combined data ({len(combined_df)} rows) to {combined_save_path}")
    
    return combined_df

def fetch_depth_snapshot(client: Client, symbol: str = "BTCUSDT", limit: int = 20) -> dict:
    """
    Ambil snapshot order book saat ini via REST API.
    """
    logger.info(f"Fetching depth snapshot for {symbol} (limit={limit})")
    try:
        res = client.get_order_book(symbol=symbol, limit=limit)
    except Exception as e:
        logger.error(f"Error fetching order book: {e}")
        raise e
        
    return {
        "timestamp": int(res.get("lastUpdateId", time.time() * 1000)),
        "bids": [[float(b[0]), float(b[1])] for b in res["bids"]],
        "asks": [[float(a[0]), float(a[1])] for a in res["asks"]]
    }
