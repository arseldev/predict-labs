"""
executor.py — Order Execution untuk testnet dan live trading.

Mode testnet WAJIB aktif sampai paper trading selesai.
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
import time
from datetime import date
import os
from dotenv import load_dotenv

class OrderExecutor:
    def __init__(self, config: dict):
        self.config = config
        self.testnet = config["binance"].get("testnet", True)
        self.symbol = config["binance"].get("symbol", "BTCUSDT")
        
        load_dotenv()
        
        # Inisialisasi client
        if self.testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True
            )
            logger.info("🧪 Executor running in TESTNET mode")
        else:
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=False
            )
            logger.warning("⚠️ Executor running in LIVE mode — real money!")
            
        self._active_order = None
        self._position = None
        
        # Parameter proteksi risiko
        # Ambil max loss dari config.yaml, jika tidak diset gunakan default
        self.position_size_pct = config["trading"].get("position_size_pct", 0.02)
        self.max_daily_loss_pct = config["trading"].get("max_daily_loss_pct", 0.03)
        
        # Sederhanakan tracking balance awal untuk menghitung kill switch
        self.initial_balance = None
        self.daily_loss_today = 0.0
        self.today = None
        
    def _get_usdt_balance(self) -> float:
        """Ambil balance USDT saat ini dari akun Binance."""
        try:
            acc = self.client.get_account()
            balances = acc.get("balances", [])
            for asset in balances:
                if asset["asset"] == "USDT":
                    return float(asset["free"])
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
        return 0.0

    def _get_current_price(self) -> float:
        """Ambil harga pasar ticker saat ini."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
        return 0.0

    def submit_order(self, direction: str, proba: float, timestamp):
        """
        Submit order buy market ke Binance dan pasang TP/SL OCO order.
        """
        if direction != "long":
            logger.warning(f"Direction {direction} not supported. Only long order is supported.")
            return

        # Ambil balance awal jika belum di-set
        if self.initial_balance is None:
            self.initial_balance = self._get_usdt_balance()
            
        # Pengecekan Kill Switch
        if not self._check_kill_switch():
            logger.warning("Kill-switch active. Order rejected.")
            return
            
        # Pastikan tidak ada posisi aktif
        if self._position is not None:
            logger.debug("Already in position, skipping new order.")
            return
            
        try:
            balance = self._get_usdt_balance()
            if balance <= 0.0:
                logger.error("USDT balance is zero or negative. Cannot submit order.")
                return
                
            position_size_usdt = balance * self.position_size_pct
            current_price = self._get_current_price()
            if current_price <= 0.0:
                logger.error("Cannot fetch price. Order aborted.")
                return
                
            quantity = round(position_size_usdt / current_price, 5)
            
            logger.info(f"Submitting {direction.upper()} market order: {quantity} BTC @ ~{current_price:.2f} USDT (P={proba:.3f})")
            
            # Market Buy Order
            order = self.client.order_market_buy(
                symbol=self.symbol,
                quantity=quantity
            )
            
            # Dapatkan harga entry fill actual
            fills = order.get("fills", [])
            entry_price = float(fills[0]["price"]) if fills else current_price
            logger.info(f"Market buy filled at price: {entry_price}")
            
            # Pasang OCO order untuk TP & SL
            self._set_tp_sl_order(entry_price=entry_price, quantity=quantity)
            
            self._position = {
                "order_id": order["orderId"],
                "entry_time": timestamp,
                "entry_price": entry_price,
                "quantity": quantity,
                "predicted_proba": proba
            }
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error on order submission: {e}")
        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=True)

    def _set_tp_sl_order(self, entry_price: float, quantity: float):
        """
        Pasang OCO (One-Cancels-the-Other) order untuk Take Profit dan Stop Loss.
        """
        tp_price = round(entry_price * (1.0 + self.config["trading"]["profit_target_pct"]), 2)
        sl_price = round(entry_price * (1.0 - self.config["trading"]["stop_loss_pct"]), 2)
        # Tambahkan limit price sedikit di bawah stop trigger untuk memastikan ter-fill
        sl_limit_price = round(sl_price * 0.999, 2)
        
        try:
            oco_order = self.client.order_oco_sell(
                symbol=self.symbol,
                quantity=quantity,
                price=tp_price,
                stopPrice=sl_price,
                stopLimitPrice=sl_limit_price,
                stopLimitTimeInForce="GTC"
            )
            logger.info(f"OCO Exit Order set successfully: TP={tp_price}, SL={sl_price}")
        except Exception as e:
            logger.error(f"Failed to set OCO exit order: {e}")
            logger.info("Attempting fallback: Setting a plain Stop-Loss Market order.")
            try:
                self.client.create_order(
                    symbol=self.symbol,
                    side="SELL",
                    type="STOP_LOSS_LIMIT",
                    quantity=quantity,
                    stopPrice=sl_price,
                    price=sl_limit_price,
                    timeInForce="GTC"
                )
                logger.info(f"Fallback Stop-Loss set successfully at {sl_price}")
            except Exception as ex:
                logger.error(f"Critical: Fallback stop-loss also failed: {ex}")

    def _check_kill_switch(self) -> bool:
        """Cek apakah bot mencapai ambang batas kerugian harian."""
        current_date = date.today()
        if self.today != current_date:
            self.today = current_date
            self.daily_loss_today = 0.0
            self.initial_balance = self._get_usdt_balance()
            
        current_balance = self._get_usdt_balance()
        # Hitung net change sejak awal hari
        if self.initial_balance and self.initial_balance > 0.0:
            self.daily_loss_today = current_balance - self.initial_balance
            
        loss_limit = -self.initial_balance * self.max_daily_loss_pct if self.initial_balance else 0.0
        
        if self.daily_loss_today < loss_limit:
            logger.warning(f"Daily loss limit hit: {self.daily_loss_today:.2f} USDT (limit: {loss_limit:.2f} USDT)")
            return False
            
        return True

    def close_all_positions(self):
        """Force-close posisi aktif dengan membatalkan order OCO dan mengeksekusi market sell."""
        if self._position:
            try:
                # Batalkan open order untuk symbol ini terlebih dahulu
                open_orders = self.client.get_open_orders(symbol=self.symbol)
                for order in open_orders:
                    self.client.cancel_order(symbol=self.symbol, orderId=order["orderId"])
                    
                # Market Sell
                self.client.order_market_sell(
                    symbol=self.symbol,
                    quantity=self._position["quantity"]
                )
                logger.info(f"Successfully closed position for {self._position['quantity']} BTC")
                self._position = None
            except Exception as e:
                logger.error(f"Error during close_all_positions: {e}")
