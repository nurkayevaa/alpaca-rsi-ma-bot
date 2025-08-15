"""
Real-time crypto scalper (non-async) using Alpaca REST API.
RSI scanning triggers MA + order placement directly in memory.
Paper trading only.
"""

import os
from datetime import datetime, timedelta
from collections import deque, defaultdict
from decimal import Decimal, ROUND_DOWN

import pandas as pd
import ta

from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# === API Credentials ===
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = CryptoHistoricalDataClient()

# === Trading Parameters ===
SYMBOLS = ["BTC/USD", "ETH/USD"]
RSI_PERIOD = 14
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
MA_FAST = 50
MA_SLOW = 200

ORDER_QTY = {"BTC/USD": 0.001, "ETH/USD": 0.01}

STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.02

# === In-memory data ===
close_history = defaultdict(lambda: deque(maxlen=5000))


def round_price(price, decimals=4):
    d = Decimal(str(price))
    return float(d.quantize(Decimal('1.' + '0'*decimals), rounding=ROUND_DOWN))


def compute_indicators(symbol, closes):
    """Compute RSI and moving averages"""
    s = pd.Series(list(closes))
    rsi = ta.momentum.RSIIndicator(s, window=RSI_PERIOD).rsi().iloc[-1]
    ma_fast = s.rolling(MA_FAST).mean().iloc[-1] if len(s) >= MA_FAST else None
    ma_slow = s.rolling(MA_SLOW).mean().iloc[-1] if len(s) >= MA_SLOW else None
    return rsi, ma_fast, ma_slow


def scan_and_trade(symbol):
    """Fetch latest bars, update history, compute indicators, place orders if conditions met"""
    end = datetime.now()
    start = end - timedelta(days=1)  # last 1 day of data
    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end
    )
    bars = data_client.get_crypto_bars(req).df
    if symbol not in bars.index.get_level_values(0):
        print(f"No data for {symbol}")
        return

    df = bars.loc[symbol].copy()
    for close in df["close"]:
        close_history[symbol].append(close)

    rsi, ma_fast, ma_slow = compute_indicators(symbol, close_history[symbol])

    print(f"[{symbol}] Close:{df['close'].iloc[-1]:.2f} RSI:{rsi:.2f} MA50:{ma_fast} MA200:{ma_slow}")

    # === Trading logic ===
    if rsi < BUY_THRESHOLD and ma_fast and ma_slow and ma_fast > ma_slow:
        qty = ORDER_QTY.get(symbol, 0.001)
        sl = round_price(df['close'].iloc[-1] * (1 - STOP_LOSS_PCT))
        tp = round_price(df['close'].iloc[-1] * (1 + TAKE_PROFIT_PCT))
        print(f"BUY {symbol} signal! Submitting order...")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class="bracket",
            take_profit={"limit_price": str(tp)},
            stop_loss={"stop_price": str(sl)},
        )
        trading_client.submit_order(order)

    elif rsi > SELL_THRESHOLD and ma_fast and ma_slow and ma_fast < ma_slow:
        qty = ORDER_QTY.get(symbol, 0.001)
        sl = round_price(df['close'].iloc[-1] * (1 + STOP_LOSS_PCT))
        tp = round_price(df['close'].iloc[-1] * (1 - TAKE_PROFIT_PCT))
        print(f"SELL {symbol} signal! Submitting order...")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            order_class="bracket",
            take_profit={"limit_price": str(tp)},
            stop_loss={"stop_price": str(sl)},
        )
        trading_client.submit_order(order)


def main():
    for symbol in SYMBOLS:
        scan_and_trade(symbol)


if __name__ == "__main__":
    main()
