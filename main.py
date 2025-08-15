#!/usr/bin/env python3
"""
Live SMA+RSI Crypto Trading with Auto-Optimized Parameters
- Runs grid search for SMA/RSI strategy
- Picks top parameters by Sharpe ratio
- Trades live with Alpaca (paper trading)
"""

import os
import itertools
from datetime import datetime, timedelta, timezone

import pandas as pd
import ta
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# -------------------- Alpaca API Setup -------------------- #
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
data_client = CryptoHistoricalDataClient()  # No API key needed for crypto data
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

symbol = "BTC/USD"
qty = 0.001  # fractional qty for crypto

# -------------------- Fetch Historical Data -------------------- #
def fetch_data(symbol, days=365, timeframe=TimeFrame.Hour):
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt
    )
    bars = data_client.get_crypto_bars(req)
    df = bars.df
    if symbol not in df.index.get_level_values(0):
        raise ValueError(f"No data found for {symbol}")
    df = df.loc[symbol].copy().sort_index()
    return df

# -------------------- Strategy Logic -------------------- #
def logic_sma_rsi(df, sma_fast, sma_slow, rsi_buy, rsi_sell):
    sma_f = df["close"].rolling(sma_fast).mean()
    sma_s = df["close"].rolling(sma_slow).mean()
    rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    def entry(df_, i):
        return (sma_f.iat[i] > sma_s.iat[i]) and (rsi.iat[i] < rsi_buy)

    def exit(df_, i):
        return (sma_f.iat[i] < sma_s.iat[i]) and (rsi.iat[i] > rsi_sell)

    return entry, exit

# -------------------- Backtest -------------------- #
def backtest_sma_rsi(df, sma_fast, sma_slow, rsi_buy, rsi_sell, sl_pct=0.01, tp_pct=0.02, start_equity=10000):
    cash = start_equity
    position_qty = 0
    entry_price = 0
    equity_curve = []
    trades = []

    entry_logic, exit_logic = logic_sma_rsi(df, sma_fast, sma_slow, rsi_buy, rsi_sell)

    for i in range(1, len(df)):
        price_open = df["open"].iloc[i]
        price_close = df["close"].iloc[i]

        # mark-to-market equity
        equity_curve.append(cash + position_qty * price_close)

        # Exit logic
        if position_qty > 0 and exit_logic(df, i):
            cash += position_qty * price_open
            trades.append((entry_price, price_open))
            position_qty = 0

        # Entry logic
        elif position_qty == 0 and entry_logic(df, i):
            position_qty = cash / price_open
            entry_price = price_open
            cash = 0

    # Final close
    if position_qty > 0:
        cash += position_qty * df["close"].iloc[-1]
        trades.append((entry_price, df["close"].iloc[-1]))

    total_return = (cash - start_equity) / start_equity
    win_rate = sum(1 for e, x in trades if x > e) / len(trades) if trades else 0
    sharpe = total_return / (pd.Series(equity_curve).pct_change().std() * (len(equity_curve)**0.5)) if len(equity_curve) > 1 else 0

    return {
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "rsi_buy": rsi_buy,
        "rsi_sell": rsi_sell,
        "sharpe": sharpe,
        "total_return": total_return,
        "win_rate": win_rate,
    }

# -------------------- Grid Search -------------------- #
def grid_search(df):
    sma_fast_vals = [10, 20, 50]
    sma_slow_vals = [100, 200]
    rsi_buy_vals = [20, 30, 40]
    rsi_sell_vals = [50, 60, 70]

    results = []
    for sma_f, sma_s, rsi_b, rsi_s in itertools.product(sma_fast_vals, sma_slow_vals, rsi_buy_vals, rsi_sell_vals):
        if sma_f >= sma_s or rsi_s <= rsi_b:
            continue
        stats = backtest_sma_rsi(df, sma_f, sma_s, rsi_b, rsi_s)
        results.append(stats)

    ranked = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    best = ranked[0]
    print("\n=== Best Parameters Found ===")
    print(best)
    return best

# -------------------- Live Trading -------------------- #
def trade_live(params):
    sma_fast = params["sma_fast"]
    sma_slow = params["sma_slow"]
    rsi_buy = params["rsi_buy"]
    rsi_sell = params["rsi_sell"]

    print(f"\nTrading Live with SMA_fast={sma_fast}, SMA_slow={sma_slow}, RSI_buy={rsi_buy}, RSI_sell={rsi_sell}")

    df = fetch_data(symbol, days=365)
    df["SMA_fast"] = df["close"].rolling(sma_fast).mean()
    df["SMA_slow"] = df["close"].rolling(sma_slow).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    latest = df.iloc[-1]
    print(latest[["close", "SMA_fast", "SMA_slow", "RSI"]])

    if latest["RSI"] < rsi_buy and latest["SMA_fast"] > latest["SMA_slow"]:
        print("BUY signal!")
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
        trading_client.submit_order(order)
    elif latest["RSI"] > rsi_sell and latest["SMA_fast"] < latest["SMA_slow"]:
        print("SELL signal!")
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
        trading_client.submit_order(order)
    else:
        print("No trade today.")

# -------------------- Main -------------------- #
if __name__ == "__main__":
    # Step 1: Fetch historical data
    historical_df = fetch_data(symbol, days=365)

    # Step 2: Grid search for best parameters
    best_params = grid_search(historical_df)

    # Step 3: Live trade using best parameters
    trade_live(best_params)
=======
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

