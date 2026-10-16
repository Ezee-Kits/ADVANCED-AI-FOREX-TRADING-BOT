import MetaTrader5 as mt5
import json
import time
import re


# ==========================================================
# INIT MT5
# ==========================================================
if not mt5.initialize():
    raise Exception("MT5 initialization failed")


# ==========================================================
# YOUR SYMBOL LIST (WITH SUFFIX)
# ==========================================================
SYMBOLS = [
    "AUDCAD","AUDJPY","AUDUSD","BTCUSD","CADJPY",
    "ETHUSD","EURJPY","EURUSD","EURGBP","GBPCAD",
    "GBPJPY","GBPUSD","NZDJPY","NZDUSD","USDCAD",
    "USDCHF","USDJPY","XAGUSD","XAUUSD",
]

SYMBOLS = [symbol + 'c' for symbol in SYMBOLS]

# ==========================================================
# CLEAN SYMBOL FUNCTION
# ==========================================================
def clean_symbol(symbol):
    return re.match(r"[A-Z]+", symbol).group()


# ==========================================================
# GET ACCOUNT INFO
# ==========================================================
account = mt5.account_info()

if account is None:
    raise Exception(f"Failed to get account information: {mt5.last_error()}")


# ==========================================================
# ACCOUNT INFORMATION TO STORE
# ==========================================================
ACCOUNT_INFO = {
    "balance": account.balance,
    "equity": account.equity,
    "currency": account.currency,
    "leverage": account.leverage
}


print("\n================ ACCOUNT INFO ================")

print(f"Balance : {account.balance}")
print(f"Equity  : {account.equity}")
print(f"Currency: {account.currency}")
print(f"Leverage: 1:{account.leverage}")


# ==========================================================
# GET SYMBOL INFO
# ==========================================================
def get_symbol_info(symbol):

    if not mt5.symbol_select(symbol, True):

        print(f"[[SKIP]] Cannot select: {symbol}")
        print(f"         Error: {mt5.last_error()}")

        return None


    info = mt5.symbol_info(symbol)

    if info is None:

        print(f"[[SKIP]] No symbol information: {symbol}")

        return None


    # ======================================================
    # PIP SIZE
    # ======================================================

    digits = info.digits

    if digits in (3, 5):
        pip_size = info.point * 10
    else:
        pip_size = info.point


    # ======================================================
    # PIP VALUE PER LOT
    # ======================================================

    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size

    if tick_value > 0 and tick_size > 0:

        pip_value_per_lot = (
            tick_value / tick_size
        ) * pip_size

    else:

        pip_value_per_lot = None


    # ======================================================
    # CURRENT SPREAD
    # ======================================================

    spread = info.ask - info.bid


    # ======================================================
    # MINIMUM STOP DISTANCE
    # ======================================================

    min_gap = (info.trade_stops_level * info.point)

    if min_gap <= 0:

        min_gap = pip_size * 5


    # ======================================================
    # LOT INFORMATION
    # ======================================================

    vol_info = {
        "min": info.volume_min,
        "max": info.volume_max,
        "step": info.volume_step
    }


    # ======================================================
    # COMBINE ACCOUNT + SYMBOL INFORMATION
    # ======================================================

    return {

        # ACCOUNT INFORMATION
        "balance": ACCOUNT_INFO["balance"],
        "equity": ACCOUNT_INFO["equity"],
        "currency": ACCOUNT_INFO["currency"],
        "leverage": ACCOUNT_INFO["leverage"],

        # SYMBOL INFORMATION
        "pip_size": pip_size,
        "pip_value_per_lot": pip_value_per_lot,
        "spread": spread,
        "min_gap": min_gap,
        "vol_info": vol_info
    }


# ==========================================================
# BUILD FINAL DICTIONARY
# ==========================================================
SYMBOL_INFO = {}


for symbol in SYMBOLS:

    data = get_symbol_info(symbol)

    if data is not None:

        key = clean_symbol(symbol)

        SYMBOL_INFO[key] = data

        print(
            f"[[GOOD]] Saved: {symbol} → {key} | "
            f"Balance: {data['balance']} | "
            f"Min Lot: {data['vol_info']['min']} | "
            f"Max Lot: {data['vol_info']['max']} | "
            f"Step: {data['vol_info']['step']}"
        )

    time.sleep(0.2)


# ==========================================================
# SAVE FILE
# ==========================================================
with open("CSV_FILES/SYMBOL_INFO.json","w",encoding="utf-8") as f:

    json.dump(SYMBOL_INFO,f,indent=4)


print("\n[[GOOD]] Saved SYMBOL_INFO.json")


# ==========================================================
# CLOSE MT5
# ==========================================================
mt5.shutdown()