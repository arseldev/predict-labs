import requests

url = "https://api.binance.com/api/v3/ping"
try:
    res = requests.get(url, verify=True, timeout=5)
    print("api Status (verify=True):", res.status_code)
    print("api Text (verify=True):", repr(res.text))
except Exception as e:
    print("api Error:", e)
