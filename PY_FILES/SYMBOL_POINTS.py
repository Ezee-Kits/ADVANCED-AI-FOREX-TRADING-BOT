import MetaTrader5 as mt5
import json
import os


# ==========================================================
# SETTINGS
# ==========================================================

CSV_FOLDER = r"C:\Users\HP\Desktop\PYTHON FILES\#PYTHON\FOREX TRADING\CSV_FILES"

POINTS_FILE = os.path.join(CSV_FOLDER,"symbol_points.json")

# ==========================================================
# SYMBOLS
# ==========================================================

SYMBOLS = [
    "AUDCAD","AUDJPY","AUDUSD","BTCUSD","CADJPY",
    "ETHUSD","EURJPY","EURUSD","EURGBP","GBPCAD",
    "GBPJPY","GBPUSD","NZDJPY","NZDUSD","USDCAD",
    "USDCHF","USDJPY","XAGUSD","XAUUSD",
]

SYMBOLS = [symbol + 'c' for symbol in SYMBOLS]


# ==========================================================
# INITIALIZE MT5
# ==========================================================

if not mt5.initialize():
    print("MT5 initialization failed")
    print(mt5.last_error())
    raise SystemExit


print("=" * 60)
print("COLLECTING MT5 SYMBOL POINT VALUES")
print("=" * 60)


symbol_points = {}

failed_symbols = []


# ==========================================================
# GET POINT FOR EACH SYMBOL
# ==========================================================

for symbol in SYMBOLS:

    print(f"\nChecking: {symbol}")

    # Make sure symbol is available in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"  FAILED: Could not select {symbol}")
        failed_symbols.append(symbol)
        continue

    symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        print(f"  FAILED: No symbol information for {symbol}")
        failed_symbols.append(symbol)
        continue

    point = symbol_info.point

    if point is None or point <= 0:
        print(f"  FAILED: Invalid point value for {symbol}")
        failed_symbols.append(symbol)
        continue

    symbol_points[symbol] = {
        "point": float(point),
        "digits": int(symbol_info.digits),
        "trade_tick_size": float(symbol_info.trade_tick_size),
        "trade_tick_value": float(symbol_info.trade_tick_value)
    }

    print(f"  Point: {point}")
    print(f"  Digits: {symbol_info.digits}")
    print(f"  Tick Size: {symbol_info.trade_tick_size}")
    print(f"  Tick Value: {symbol_info.trade_tick_value}")


# ==========================================================
# SAVE JSON
# ==========================================================

with open(POINTS_FILE, "w", encoding="utf-8") as file:

    json.dump(
        symbol_points,
        file,
        indent=4
    )


# ==========================================================
# SHUTDOWN
# ==========================================================

mt5.shutdown()


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("POINT COLLECTION COMPLETE")
print("=" * 60)

print(f"Successful symbols : {len(symbol_points)}")
print(f"Failed symbols     : {len(failed_symbols)}")

print(f"\nJSON saved to:")
print(POINTS_FILE)


if failed_symbols:

    print("\nFailed symbols:")

    for symbol in failed_symbols:
        print(f"  - {symbol}")