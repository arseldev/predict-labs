"""
test_data_fetch.py — Unit tests untuk data_fetch.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.data_fetch import fetch_klines_rest, fetch_depth_snapshot

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock data kline: 5 candles 5-menit
    dummy_klines = []
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
    for i in range(5):
        dummy_klines.append([
            base_time + i * 300000,   # open time (+5m)
            str(50000.0 + i * 100),   # open
            str(50200.0 + i * 100),   # high
            str(49800.0 + i * 100),   # low
            str(50100.0 + i * 100),   # close
            str(10.0 * (i + 1)),      # volume
            base_time + i * 300000 + 299999,  # close time
            str(500000.0),            # quote volume
            100 + i,                  # trades
            str(5.0 * (i + 1)),       # taker buy base
            str(250000.0),            # taker buy quote
            "0"                       # ignore
        ])
    client.get_historical_klines.return_value = dummy_klines
    
    # Mock data order book
    client.get_order_book.return_value = {
        "lastUpdateId": 123456,
        "bids": [[str(50000.0 - i * 10), str(1.0 + i)] for i in range(20)],
        "asks": [[str(50010.0 + i * 10), str(1.2 + i)] for i in range(20)]
    }
    return client

class TestFetchKlines:
    def test_returns_dataframe(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_index_is_datetime(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_no_nan_in_ohlcv(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert df[["open", "high", "low", "close", "volume"]].isna().sum().sum() == 0

    def test_sorted_ascending(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert df.index.is_monotonic_increasing

    def test_no_duplicate_timestamps(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert df.index.is_unique

    def test_high_gte_low(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert (df["high"] >= df["low"]).all()

    def test_volume_non_negative(self, mock_client):
        df = fetch_klines_rest(mock_client, "BTCUSDT", "5m", "7 days ago UTC")
        assert (df["volume"] >= 0).all()

class TestDepthSnapshot:
    def test_returns_bids_asks(self, mock_client):
        snapshot = fetch_depth_snapshot(mock_client, "BTCUSDT", limit=20)
        assert "bids" in snapshot
        assert "asks" in snapshot
        assert "timestamp" in snapshot

    def test_bids_sorted_desc(self, mock_client):
        snapshot = fetch_depth_snapshot(mock_client, "BTCUSDT", limit=20)
        bid_prices = [float(b[0]) for b in snapshot["bids"]]
        assert bid_prices == sorted(bid_prices, reverse=True)

    def test_asks_sorted_asc(self, mock_client):
        snapshot = fetch_depth_snapshot(mock_client, "BTCUSDT", limit=20)
        ask_prices = [float(a[0]) for a in snapshot["asks"]]
        assert ask_prices == sorted(ask_prices)
