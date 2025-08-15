import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import ta

# -------------------- API Setup -------------------- #
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
data_client = CryptoHistoricalDataClient()  # Crypto data doesn't require API keys
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

symbol = "BTC/USD"   # crypto pair
qty = 0.001          # fractional qty works for crypto

# -------------------- Load Best Parameters -------------------- #
best_params_csv = "bt_best_params_ranked_20250814_113116.csv"  # adjust path if needed
best_params_df = pd.read_csv(best_params_csv)
best_params = best_params_df.iloc[0].to_dict()  # pick top-ranked parameters

sma_fast = int(best_params.get("sma_fast", 50))
sma_slow = int(best_params.get("sma_slow", 200))
rsi_buy = float(best_params.get("rsi_buy", 30))
rsi_sell = float(best_params.get("rsi_sell", 70))

print(f"Using Best Parameters: SMA Fast={sma_fast}, SMA Slow={sma_slow}, RSI Buy={rsi_buy}, RSI Sell={rsi_sell}")

# -------------------- Fetch Historical Data -------------------- #
end = datetime.now()
start = end - timedelta(days=365)

request_params = CryptoBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame.Hour,
    start=start,
    end=end
)
bars = data_client.get_crypto_bars(request_params)
bars_df = bars.df

if symbol in bars_df.index.get_level_values(0):
    df = bars_df.loc[symbol].copy()
else:
    raise ValueError(f"No data found for {symbol}")

# -------------------- Technical Indicators -------------------- #
df["SMA_fast"] = df["close"].rolling(sma_fast).mean()
df["SMA_slow"] = df["close"].rolling(sma_slow).mean()
df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

latest = df.iloc[-1]
print(latest[["close", "SMA_fast", "SMA_slow", "RSI"]])

# -------------------- Generate Trading Signal -------------------- #
if latest["RSI"] < rsi_buy and latest["SMA_fast"] > latest["SMA_slow"]:
    print("BUY signal!")
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC
    )
    trading_client.submit_order(order)
elif latest["RSI"] > rsi_sell and latest["SMA_fast"] < latest["SMA_slow"]:
    print("SELL signal!")
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    trading_client.submit_order(order)
else:
    print("No trade today.")
