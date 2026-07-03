import sqlite3
import pandas as pd

conn = sqlite3.connect('logs/predictions.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', cursor.fetchall())

cursor.execute("PRAGMA table_info(predictions)")
cols = cursor.fetchall()
print('Predictions columns:', [(c[1], c[2]) for c in cols])

cursor.execute("SELECT COUNT(*) FROM predictions")
print('Predictions count:', cursor.fetchone()[0])

try:
    cursor.execute("SELECT COUNT(*) FROM trades")
    print('Trades count:', cursor.fetchone()[0])
except:
    print('No trades table')

# Show last 5 predictions
try:
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 5", conn)
    print("\nLast 5 predictions:")
    print(df.to_string())
except Exception as e:
    print(f"Error: {e}")

conn.close()
