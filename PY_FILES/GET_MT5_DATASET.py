
import MetaTrader5 as mt5
import pandas as pd
import os
import time


# ============================================================
# 1. CONFIGURATION
# ============================================================

CD_TIME = "15M"

# ------------------------------------------------------------
# DATA RANGE
# ------------------------------------------------------------
# Change these dates when necessary.
START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"


# ------------------------------------------------------------
# SYMBOLS
# ------------------------------------------------------------
# SYMBOLS = [
#     "AUDCAD","AUDJPY","AUDUSD","BTCUSD","CADJPY","ETHUSD",
#     "EURJPY","EURUSD","EURGBP","GBPCAD","GBPJPY","GBPUSD",
#     "NZDJPY","NZDUSD","USDCAD","USDCHF","USDJPY","XAGUSD","XAUUSD",
# ]

SYMBOLS = ["EURUSD"]

# Add Exness Cent suffix
SYMBOLS = [symbol + "c" for symbol in SYMBOLS]



# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------
OUTPUT_DIR = "CSV_FILES"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. TIMEFRAME CONFIGURATION
# ============================================================

# MT5 native timeframe mapping
NATIVE_TIMEFRAMES = {

    "1M": mt5.TIMEFRAME_M1,
    "2M": mt5.TIMEFRAME_M2,
    "3M": mt5.TIMEFRAME_M3,
    "4M": mt5.TIMEFRAME_M4,
    "5M": mt5.TIMEFRAME_M5,
    "6M": mt5.TIMEFRAME_M6,
    "10M": mt5.TIMEFRAME_M10,
    "12M": mt5.TIMEFRAME_M12,
    "15M": mt5.TIMEFRAME_M15,
    "20M": mt5.TIMEFRAME_M20,
    "30M": mt5.TIMEFRAME_M30,

    "1H": mt5.TIMEFRAME_H1,
    "2H": mt5.TIMEFRAME_H2,
    "3H": mt5.TIMEFRAME_H3,
    "4H": mt5.TIMEFRAME_H4,

    "1D": mt5.TIMEFRAME_D1,
}


# ============================================================
# 3. INITIALIZE MT5
# ============================================================

print("\n============================================================")
print("        MT5 HISTORICAL DATA DOWNLOADER")
print("===============================================================")
print(f"TIMEFRAME : {CD_TIME}")
print(f"START     : {START_DATE}")
print(f"END       : {END_DATE}")
print(f"SYMBOLS   : {len(SYMBOLS)}")
print("================================================================\n")


if not mt5.initialize():
    print("[[ERROR]] Failed to initialize MetaTrader 5")
    print(f"Error: {mt5.last_error()}")
    raise SystemExit


print("[[GOOD]] MetaTrader 5 initialized successfully.\n")


# ============================================================
# 4. HELPER: GET MT5 TIMEFRAME
# ============================================================

def get_timeframe(timeframe_string):

    timeframe_string = timeframe_string.upper().strip()

    if timeframe_string not in NATIVE_TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe_string}\n"
            f"Supported timeframes: {list(NATIVE_TIMEFRAMES.keys())}"
        )

    return NATIVE_TIMEFRAMES[timeframe_string]


# ============================================================
# 5. DOWNLOAD DATA
# ============================================================

def download_mt5_data(symbol, timeframe_string, start_date, end_date):

    timeframe = get_timeframe(timeframe_string)

    print("\n------------------------------------------------------------")
    print(f"Downloading: {symbol}")
    print(f"Timeframe : {timeframe_string}")
    print(f"From      : {start_date}")
    print(f"To        : {end_date}")
    print("------------------------------------------------------------")

    # --------------------------------------------------------
    # Select symbol
    # --------------------------------------------------------
    if not mt5.symbol_select(symbol, True):

        print(f"[[ERROR]] Could not select {symbol}")
        print(f"MT5 Error: {mt5.last_error()}")

        return None


    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)


    # --------------------------------------------------------
    # Download rates
    # --------------------------------------------------------
    rates = mt5.copy_rates_range(symbol,timeframe,start.to_pydatetime(),end.to_pydatetime())

    if rates is None:
        print(f"[[ERROR]] No data returned for {symbol}")
        print(f"MT5 Error: {mt5.last_error()}")

        return None


    if len(rates) == 0:
        print(f"[[ERROR]] Empty dataset for {symbol}")

        return None


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------
    data = pd.DataFrame(rates)


    # ========================================================
    # 6. CONVERT MT5 COLUMNS
    # ========================================================

    data["Date"] = pd.to_datetime(data["time"],unit="s")

    data = data.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
        "spread": "Spread",
    })


    # ========================================================
    # 7. SELECT REQUIRED COLUMNS
    # ========================================================

    data = data[["Date","Open","High","Low","Close","Volume","Spread",]]


    # ========================================================
    # 8. CLEAN DATA
    # ========================================================

    # Sort chronologically
    data = data.sort_values("Date")

    # Remove duplicate rows
    data = data.drop_duplicates(subset=["Date"],keep="first")

    # Remove invalid OHLC rows
    data = data.dropna(subset=["Date","Open","High","Low","Close","Volume","Spread",])

    # Reset index
    data = data.reset_index(drop=True)


    # ========================================================
    # 9. SAVE DATASET
    # ========================================================

    # Remove the 'c' suffix from output filename
    clean_symbol = symbol[:-1] if symbol.endswith("c") else symbol


    output_file = (f"{OUTPUT_DIR}/MT5_{timeframe_string}_{clean_symbol}_Exchange_Rate_Dataset.csv")

    data.to_csv(output_file,index=False)


    # ========================================================
    # 10. REPORT
    # ========================================================

    print(f"[[GOOD]] Dataset created successfully!")
    print(f"Symbol     : {symbol}")
    print(f"Rows       : {len(data):,}")
    print(f"Start      : {data['Date'].iloc[0]}")
    print(f"End        : {data['Date'].iloc[-1]}")
    print(f"Saved to   : {output_file}")


    return data


# ============================================================
# 11. DOWNLOAD ALL SYMBOLS
# ============================================================

successful = []
failed = []


for symbol in SYMBOLS:

    try:

        result = download_mt5_data(
            symbol=symbol,
            timeframe_string=CD_TIME,
            start_date=START_DATE,
            end_date=END_DATE
        )


        if result is not None and len(result) > 0:

            successful.append(symbol)

        else:

            failed.append(symbol)


    except Exception as e:

        print("\n[[ERROR]] Unexpected error")
        print(f"Symbol : {symbol}")
        print(f"Error  : {e}")

        failed.append(symbol)


    # Small pause between symbols
    time.sleep(1)


# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n\n============================================================")
print("                 DOWNLOAD COMPLETED")
print("============================================================")

print(f"\nTimeframe: {CD_TIME}")

print("\nSuccessful:")
for symbol in successful:
    print(f"  [GOOD] {symbol}")


print("\nFailed:")
for symbol in failed:
    print(f"  [FAILED] {symbol}")


print("\n------------------------------------------------------------")
print(f"Successful: {len(successful)} / {len(SYMBOLS)}")
print(f"Failed    : {len(failed)} / {len(SYMBOLS)}")
print("------------------------------------------------------------")

print(f"\nDatasets saved inside: {OUTPUT_DIR}/")


# ============================================================
# 13. SHUTDOWN MT5
# ============================================================

mt5.shutdown()

print("\n[[GOOD]] MT5 connection closed.")
print("============================================================")
