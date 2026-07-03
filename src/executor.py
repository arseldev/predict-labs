"""
executor.py — Order Execution untuk testnet dan live trading.

Mode testnet WAJIB aktif sampai paper trading selesai.
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger
import time
import threading
from datetime import date, datetime, timezone
import os
from dotenv import load_dotenv

class OrderExecutor:
    def __init__(self, config: dict, pred_logger=None):
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
        self._oco_order_list_id = None  # Track OCO order untuk monitoring
        
        # Parameter proteksi risiko
        # Ambil max loss dari config.yaml, jika tidak diset gunakan default
        self.position_size_pct = config["trading"].get("position_size_pct", 0.02)
        self.max_daily_loss_pct = config["trading"].get("max_daily_loss_pct", 0.03)
        self.tp_pct = config["trading"].get("profit_target_pct", 0.006)
        self.sl_pct = config["trading"].get("stop_loss_pct", 0.004)
        self.max_hold_candles = config["trading"].get("max_hold_candles", 12)
        
        # Sederhanakan tracking balance awal untuk menghitung kill switch
        self.initial_balance = None
        self.daily_loss_today = 0.0
        self.today = None
        
        # PredictionLogger untuk mencatat hasil trade ke SQLite
        self._pred_logger = pred_logger
        
        # Position monitor thread
        self._monitor_running = False
        self._monitor_thread = None
        
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
            fee_paid = sum(float(f.get("commission", 0)) for f in fills)
            logger.info(f"Market buy filled at price: {entry_price}")
            
            # Pasang OCO order untuk TP & SL
            oco_order = self._set_tp_sl_order(entry_price=entry_price, quantity=quantity)
            
            self._position = {
                "order_id": order["orderId"],
                "entry_time": timestamp,
                "entry_price": entry_price,
                "quantity": quantity,
                "predicted_proba": proba,
                "fee_paid": fee_paid,
                "candles_held": 0,
                "tp_price": round(entry_price * (1.0 + self.tp_pct), 2),
                "sl_price": round(entry_price * (1.0 - self.sl_pct), 2),
            }
            self._oco_order_list_id = oco_order.get("orderListId") if oco_order else None
            
            # Mulai monitor thread jika belum berjalan
            self._start_position_monitor()
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error on order submission: {e}")
        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=True)

    def _set_tp_sl_order(self, entry_price: float, quantity: float) -> dict | None:
        """
        Pasang OCO (One-Cancels-the-Other) order untuk Take Profit dan Stop Loss.
        Mengembalikan response OCO order atau None jika gagal.
        """
        tp_price = round(entry_price * (1.0 + self.tp_pct), 2)
        sl_price = round(entry_price * (1.0 - self.sl_pct), 2)
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
            return oco_order
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
            return None

    def _start_position_monitor(self):
        """Jalankan thread background yang memantau status posisi aktif."""
        if self._monitor_running:
            return
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._position_monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Position monitor thread started.")

    def _position_monitor_loop(self):
        """
        Loop background setiap 30 detik untuk cek apakah posisi aktif sudah exit.
        Saat exit terdeteksi, log trade ke SQLite via PredictionLogger.
        """
        check_interval = 30  # detik
        while self._monitor_running:
            time.sleep(check_interval)
            if self._position is None:
                continue
            try:
                self._check_position_exit()
            except Exception as e:
                logger.error(f"Position monitor error: {e}")

    def _check_position_exit(self):
        """
        Cek apakah posisi aktif sudah exit:
        1. Cek status OCO order via API
        2. Atau cek apakah max_hold_candles sudah terlampaui (timeout)
        Jika exit, hitung PnL dan log ke DB.
        """
        pos = self._position
        if pos is None:
            return

        exit_price = None
        exit_reason = None
        
        # --- Metode 1: Cek melalui open orders (jika OCO masih ada = belum exit) ---
        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            open_order_ids = {o["orderId"] for o in open_orders}
            
            # Cek apakah OCO order sudah tidak ada lagi di open orders (berarti sudah filled/cancelled)
            if self._oco_order_list_id is not None:
                oco_still_open = any(
                    o.get("orderListId") == self._oco_order_list_id
                    for o in open_orders
                )
                if not oco_still_open:
                    # OCO sudah filled — tentukan mana yang hit (TP atau SL)
                    current_price = self._get_current_price()
                    if current_price >= pos["tp_price"]:
                        exit_price = pos["tp_price"]
                        exit_reason = "take_profit"
                    else:
                        exit_price = pos["sl_price"]
                        exit_reason = "stop_loss"
        except Exception as e:
            logger.warning(f"Could not check open orders: {e}")

        # --- Metode 2: Timeout — sudah terlalu lama hold tanpa exit ---
        if exit_price is None:
            pos["candles_held"] = pos.get("candles_held", 0) + 1
            if pos["candles_held"] >= self.max_hold_candles:
                # Force exit via market sell
                current_price = self._get_current_price()
                exit_price = current_price
                exit_reason = "timeout"
                try:
                    # Batalkan open orders dulu
                    for order in self.client.get_open_orders(symbol=self.symbol):
                        self.client.cancel_order(symbol=self.symbol, orderId=order["orderId"])
                    self.client.order_market_sell(symbol=self.symbol, quantity=pos["quantity"])
                    logger.warning(f"Force-sold position after {pos['candles_held']} candles at {exit_price}")
                except Exception as e:
                    logger.error(f"Timeout sell failed: {e}")

        # --- Jika exit terdeteksi, log ke DB ---
        if exit_price is not None:
            fee_entry = pos.get("fee_paid", 0.0)
            fee_exit = exit_price * pos["quantity"] * 0.001  # estimasi fee exit
            gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            net_pnl = gross_pnl - fee_entry - fee_exit
            
            trade_data = {
                "signal_timestamp": str(pos["entry_time"]),
                "entry_time": str(pos["entry_time"]),
                "exit_time": str(datetime.now(timezone.utc)),
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "quantity": pos["quantity"],
                "direction": "long",
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "fee_paid": fee_entry + fee_exit,
                "exit_reason": exit_reason,
                "predicted_proba": pos["predicted_proba"],
            }
            
            if self._pred_logger:
                self._pred_logger.log_trade(trade_data)
                logger.info(
                    f"Trade logged ✅ | Exit: {exit_reason} | "
                    f"Entry: {pos['entry_price']} → Exit: {exit_price:.2f} | "
                    f"Net PnL: ${net_pnl:.4f}"
                )
            else:
                logger.warning("pred_logger not set — trade NOT saved to DB!")
                
            # Reset posisi
            self._position = None
            self._oco_order_list_id = None

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
        self._monitor_running = False  # Hentikan monitor thread
        if self._position:
            try:
                # Batalkan open order untuk symbol ini terlebih dahulu
                open_orders = self.client.get_open_orders(symbol=self.symbol)
                for order in open_orders:
                    self.client.cancel_order(symbol=self.symbol, orderId=order["orderId"])
                    
                # Market Sell
                current_price = self._get_current_price()
                self.client.order_market_sell(
                    symbol=self.symbol,
                    quantity=self._position["quantity"]
                )
                
                pos = self._position
                fee_exit = current_price * pos["quantity"] * 0.001
                gross_pnl = (current_price - pos["entry_price"]) * pos["quantity"]
                net_pnl = gross_pnl - pos.get("fee_paid", 0.0) - fee_exit
                
                trade_data = {
                    "signal_timestamp": str(pos["entry_time"]),
                    "entry_time": str(pos["entry_time"]),
                    "exit_time": str(datetime.now(timezone.utc)),
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "quantity": pos["quantity"],
                    "direction": "long",
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "fee_paid": pos.get("fee_paid", 0.0) + fee_exit,
                    "exit_reason": "manual_close",
                    "predicted_proba": pos["predicted_proba"],
                }
                if self._pred_logger:
                    self._pred_logger.log_trade(trade_data)
                    
                logger.info(f"Successfully closed position for {pos['quantity']} BTC")
                self._position = None
            except Exception as e:
                logger.error(f"Error during close_all_positions: {e}")
