import time
from unicorn_binance_websocket_api import BinanceWebSocketApiManager

ubwa = BinanceWebSocketApiManager(exchange="binance.com-testnet", websocket_base_uri="wss://stream.testnet.binance.vision/")
ubwa.create_stream(["kline_5m"], ["btcusdt"])

print("Waiting for websocket message...")
for _ in range(30):
    msg = ubwa.pop_stream_data_from_stream_buffer()
    if msg:
        print("Message Type:", type(msg))
        print("Message Content:", msg)
        break
    time.sleep(0.5)
else:
    print("No message received.")

ubwa.stop_manager_with_all_streams()
