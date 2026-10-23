import re
import ta
import os
import time
import atexit
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
from datetime import datetime, timedelta
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("ignore", InconsistentVersionWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# def info_init():
#     url = "https://trying-20541-default-rtdb.firebaseio.com/Main_info.json"
#     response = requests.get(url)
#     data = response.json()['main_init']
#     LOGGER.info(data)
# info_init()

LOGGER = None


def set_logger(logger):
    global LOGGER
    LOGGER = logger


def filter_normalize(series):
    return (series - series.rolling(200).min()) / (series.rolling(200).max() - series.rolling(200).min() + 1e-9)


def apply_features(df, point):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Close"] = pd.to_numeric(df["Close"])
    df["High"] = pd.to_numeric(df["High"])
    df["Low"] = pd.to_numeric(df["Low"])
    df["Open"] = pd.to_numeric(df["Open"])
    df["Volume"] = pd.to_numeric(df["Volume"])

    df["Spread"] = pd.to_numeric(df["Spread"], errors="coerce")
    df["Spread_Points"] = df["Spread"]
    df["Spread_Price"] = df["Spread_Points"] * point

    df["ATR"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()

    # ==========================================
    # 1. PRICE / RETURNS
    # ==========================================
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_6"] = df["Close"].pct_change(6)
    df["Return_12"] = df["Close"].pct_change(12)
    df["Candle_Body"] = (df["Close"] - df["Open"]) / (df["Close"] + 1e-9)
    df["Candle_Range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["Upper_Wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (df["Close"] + 1e-9)
    df["Lower_Wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (df["Close"] + 1e-9)
    df["Price_Position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    # ==========================================
    # 2. TREND (12 Features)
    # ==========================================
    df["EMA_3"] = df["Close"].ewm(span=3).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["EMA_10_Ratio"] = (df["Close"] - df["EMA_10"]) / (df["EMA_10"] + 1e-9)
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_20_Ratio"] = (df["Close"] - df["EMA_20"]) / (df["EMA_20"] + 1e-9)
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["EMA_50_Ratio"] = (df["Close"] - df["EMA_50"]) / (df["EMA_50"] + 1e-9)
    df["EMA_100"] = df["Close"].ewm(span=100).mean()
    df["EMA_100_Ratio"] = (df["Close"] - df["EMA_100"]) / (df["EMA_100"] + 1e-9)
    df["EMA10_vs_EMA20"] = (df["EMA_10"] - df["EMA_20"]) / (df["EMA_20"] + 1e-9)
    df["EMA50_vs_EMA100"] = (df["EMA_50"] - df["EMA_100"]) / (df["EMA_100"] + 1e-9)
    df["EMA10_Slope"] = df["EMA_10"].diff(5) / (df["EMA_10"] + 1e-9)
    df["EMA20_Slope"] = df["EMA_20"].diff(10) / (df["EMA_20"] + 1e-9)
    df["EMA50_Slope"] = df["EMA_50"].diff(20) / (df["EMA_50"] + 1e-9)
    df["Bull_Candle"] = (df["Close"] > df["Open"]).astype(int)
    df["Trend_Consistency_20"] = df["Bull_Candle"].rolling(20).mean()
    df["Trend_Strength"] = abs(df["EMA_20"] - df["EMA_50"]) / (df["ATR"] + 1e-9)

    # ==========================================
    # 3. VOLATILITY
    # ==========================================
    df["ATR_2"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=2).average_true_range()
    df["ATR_7"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=7).average_true_range()
    df["ATR_28"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=28).average_true_range()
    df["ATR_56"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=56).average_true_range()
    df["ATR_Percent"] = df["ATR"] / (df["Close"] + 1e-9)
    df["ATR_Short_Medium_Ratio"] = df["ATR_7"] / (df["ATR_28"] + 1e-9)
    df["ATR_Medium_Long_Ratio"] = df["ATR_28"] / (df["ATR_56"] + 1e-9)
    df["ATR_Expansion"] = df["ATR"] / (df["ATR_56"] + 1e-9)
    df["ATR_Percentile_200"] = df["ATR"].rolling(200).rank(pct=True)
    df["Return_STD_20"] = df["Return_1"].rolling(20).std()
    df["Return_STD_60"] = df["Return_1"].rolling(60).std()
    df["Volatility_Acceleration"] = df["ATR"].pct_change(5)
    df["Range_Ratio"] = df["Candle_Range"] / (df["Candle_Range"].rolling(20).mean() + 1e-9)
    df["Volatility_Compression"] = df["ATR"] / (df["ATR"].rolling(100).median() + 1e-9)
    df["Market_Activity"] = df["Range_Ratio"].rolling(20).mean() / (df["Range_Ratio"].rolling(100).mean() + 1e-9)
    df["volume_weighted_momentum"] = df["Return_1"] * df["Volume"]
    df["momentum"] = df["Close"].pct_change(5)
    df["price_acceleration"] = df["momentum"] - df["momentum"].shift(1)
    mid_price = (df["High"].rolling(20).max() + df["Low"].rolling(20).min()) / 2
    df["liquidity_vacuum"] = (df["Close"] - mid_price) / (df["ATR"] + 1e-9)

    # ==========================================
    # 4. MOMENTUM
    # ==========================================
    df["RSI_7"] = ta.momentum.RSIIndicator(close=df["Close"], window=7).rsi()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["RSI_28"] = ta.momentum.RSIIndicator(close=df["Close"], window=28).rsi()
    df["ROC_3"] = ta.momentum.ROCIndicator(close=df["Close"], window=3).roc()
    df["ROC_6"] = ta.momentum.ROCIndicator(close=df["Close"], window=6).roc()
    df["ROC_12"] = ta.momentum.ROCIndicator(close=df["Close"], window=12).roc()
    df["ROC_24"] = ta.momentum.ROCIndicator(close=df["Close"], window=24).roc()
    macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD_Hist"] = macd.macd_diff()
    df["MACD_Hist_Slope"] = df["MACD_Hist"].diff(3)
    stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    adx = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ADX_14"] = adx.adx()
    df["DI_Difference"] = adx.adx_pos() - adx.adx_neg()
    df["Momentum_Acceleration"] = df["ROC_6"].diff(3)
    df["Momentum_Consistency_20"] = np.sign(df["Return_1"]).rolling(20).mean()

    # ==========================================
    # 5. MARKET STRUCTURE / PRICE ACTION
    # ==========================================
    recent_high_20 = df["High"].rolling(20).max()
    recent_low_20 = df["Low"].rolling(20).min()
    recent_high_50 = df["High"].rolling(50).max()
    recent_low_50 = df["Low"].rolling(50).min()
    recent_high_100 = df["High"].rolling(100).max()
    recent_low_100 = df["Low"].rolling(100).min()
    df["Distance_To_Recent_High_20"] = (df["Close"] - recent_high_20) / (df["Close"] + 1e-9)
    df["Distance_To_Recent_Low_20"] = (df["Close"] - recent_low_20) / (df["Close"] + 1e-9)
    df["Distance_To_Recent_High_50"] = (df["Close"] - recent_high_50) / (df["Close"] + 1e-9)
    df["Distance_To_Recent_Low_50"] = (df["Close"] - recent_low_50) / (df["Close"] + 1e-9)
    df["Distance_To_Recent_High_100"] = (df["Close"] - recent_high_100) / (df["Close"] + 1e-9)
    df["Distance_To_Recent_Low_100"] = (df["Close"] - recent_low_100) / (df["Close"] + 1e-9)
    df["Structure_Position_20"] = (df["Close"] - recent_low_20) / (recent_high_20 - recent_low_20 + 1e-9)
    df["Structure_Position_50"] = (df["Close"] - recent_low_50) / (recent_high_50 - recent_low_50 + 1e-9)
    df["Structure_Position_100"] = (df["Close"] - recent_low_100) / (recent_high_100 - recent_low_100 + 1e-9)
    swing_high = df["High"].rolling(5).max()
    swing_low = df["Low"].rolling(5).min()
    higher_high = (swing_high > swing_high.shift(5)).astype(int)
    higher_low = (swing_low > swing_low.shift(5)).astype(int)
    lower_high = (swing_high < swing_high.shift(5)).astype(int)
    lower_low = (swing_low < swing_low.shift(5)).astype(int)
    df["HH_HL_Structure"] = higher_high + higher_low
    df["LH_LL_Structure"] = lower_high + lower_low
    previous_high_20 = df["High"].rolling(20).max().shift(1)
    df["Breakout_20"] = (df["Close"] > previous_high_20).astype(int)
    previous_low_20 = df["Low"].rolling(20).min().shift(1)
    df["Breakdown_20"] = (df["Close"] < previous_low_20).astype(int)
    df["Swing_Direction"] = np.sign(swing_high.diff(5)) + np.sign(swing_low.diff(5))
    df["Structure_Strength"] = df["Swing_Direction"].rolling(20).mean()

    # ==========================================
    # 6. LIQUIDITY / TRADEABILITY
    # ==========================================
    # df["Spread_ATR_Ratio"] = (df["Spread_Price"] / (df["ATR"] + 1e-9))
    df["Spread_ATR_Ratio"] = df["Spread"] / (df["ATR"] + 1e-9)
    df["Spread_Percentile_5"] = df["Spread"].rolling(5).rank(pct=True)
    df["Spread_Percentile_30"] = df["Spread"].rolling(30).rank(pct=True)
    df["Spread_Percentile_100"] = df["Spread"].rolling(100).rank(pct=True)
    df["Spread_Percentile_200"] = df["Spread"].rolling(200).rank(pct=True)
    df["Spread_Shock"] = df["Spread"] / (df["Spread"].rolling(100).median() + 1e-9)
    df["Volume_Ratio_20"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["Volume_Ratio_100"] = df["Volume"] / (df["Volume"].rolling(100).mean() + 1e-9)
    df["Volume_Percentile_200"] = df["Volume"].rolling(200).rank(pct=True)
    candle_range = df["High"] - df["Low"]
    df["Range_Activity_100"] = candle_range / (candle_range.rolling(100).mean() + 1e-9)
    df["Candle_Efficiency"] = abs(df["Close"] - df["Open"]) / (candle_range + 1e-9)
    df["Directional_Efficiency"] = (df["Close"] - df["Open"]) / (candle_range + 1e-9)
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Wick_Ratio"] = (upper_wick + lower_wick) / (candle_range + 1e-9)
    spread_stress = df["Spread_ATR_Ratio"].rolling(200).rank(pct=True)
    volume_stress = 1 - df["Volume_Percentile_200"]
    df["Liquidity_Stress"] = 0.6 * spread_stress + 0.4 * volume_stress
    df["Tradeability_Score"] = 1 - df["Liquidity_Stress"]

    # ==========================================
    # 7. SUPPORT / RESISTANCE
    # ==========================================
    resistance_20 = df["High"].rolling(20).max().shift(1)
    support_20 = df["Low"].rolling(20).min().shift(1)
    resistance_50 = df["High"].rolling(50).max().shift(1)
    support_50 = df["Low"].rolling(50).min().shift(1)

    df["Distance_To_Resistance_20"] = (resistance_20 - df["Close"]) / (df["ATR"] + 1e-9)
    df["Distance_To_Support_20"] = (df["Close"] - support_20) / (df["ATR"] + 1e-9)
    df["Distance_To_Resistance_50"] = (resistance_50 - df["Close"]) / (df["ATR"] + 1e-9)
    df["Distance_To_Support_50"] = (df["Close"] - support_50) / (df["ATR"] + 1e-9)

    tolerance = df["ATR"] * 0.15
    resistance_test = ((df["High"] >= resistance_20 - tolerance) & (df["Close"] <= resistance_20 + tolerance)).astype(int)
    support_test = ((df["Low"] <= support_20 + tolerance) & (df["Close"] >= support_20 - tolerance)).astype(int)
    df["Resistance_Test_Count"] = resistance_test.rolling(20).sum()
    df["Support_Test_Count"] = support_test.rolling(20).sum()
    body = abs(df["Close"] - df["Open"])
    candle_range = df["High"] - df["Low"]
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Doji_Score"] = 1 - (body / (candle_range + 1e-9))
    df["Doji_Score"] = df["Doji_Score"].clip(0, 1)
    previous_bearish = df["Close"].shift(1) < df["Open"].shift(1)
    current_bullish = df["Close"] > df["Open"]
    body_engulfs_previous = (df["Open"] <= df["Close"].shift(1)) & (df["Close"] >= df["Open"].shift(1))
    df["Bullish_Engulfing"] = (previous_bearish & current_bullish & body_engulfs_previous).astype(int)
    previous_bullish = df["Close"].shift(1) > df["Open"].shift(1)
    current_bearish = df["Close"] < df["Open"]
    body_engulfs_previous_bearish = (df["Open"] >= df["Close"].shift(1)) & (df["Close"] <= df["Open"].shift(1))
    df["Bearish_Engulfing"] = (previous_bullish & current_bearish & body_engulfs_previous_bearish).astype(int)
    df["Hammer_Score"] = (lower_wick / (body + 1e-9)) * (1 - (upper_wick / (candle_range + 1e-9)))
    df["Hammer_Score"] = df["Hammer_Score"].clip(lower=0)
    df["Shooting_Star_Score"] = (upper_wick / (body + 1e-9)) * (1 - (lower_wick / (candle_range + 1e-9)))
    df["Shooting_Star_Score"] = df["Shooting_Star_Score"].clip(lower=0)
    bollinger = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    bb_high = bollinger.bollinger_hband()
    bb_low = bollinger.bollinger_lband()
    bb_mid = bollinger.bollinger_mavg()
    df["BB_Position"] = (df["Close"] - bb_low) / (bb_high - bb_low + 1e-9)
    df["BB_Position"] = df["BB_Position"].clip(-1, 2)
    df["BB_Width"] = (bb_high - bb_low) / (df["Close"] + 1e-9)
    bb_width_median = df["BB_Width"].rolling(100).median()
    df["BB_Width_Ratio"] = df["BB_Width"] / (bb_width_median + 1e-9)
    bb_width_percentile = df["BB_Width"].rolling(200).rank(pct=True)
    df["BB_Squeeze"] = 1 - bb_width_percentile
    df["BB_Squeeze"] = df["BB_Squeeze"].clip(0, 1)

    # ==========================================
    # 8. TIME / SESSION
    # ==========================================
    hour_decimal = df["Date"].dt.hour + df["Date"].dt.minute / 60
    df["Hour_Sin"] = np.sin(2 * np.pi * hour_decimal / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * hour_decimal / 24)
    minute = df["Date"].dt.minute
    df["Minute_Sin"] = np.sin(2 * np.pi * minute / 60)
    df["Minute_Cos"] = np.cos(2 * np.pi * minute / 60)
    hour = df["Date"].dt.hour
    df["Session_Asia"] = ((hour >= 0) & (hour < 8)).astype(int)
    df["Session_London"] = ((hour >= 8) & (hour < 16)).astype(int)
    df["Session_NewYork"] = ((hour >= 13) & (hour < 21)).astype(int)
    df["Session_Overlap"] = ((df["Session_London"] == 1) & (df["Session_NewYork"] == 1)).astype(int)

    return df




def normalize_symbol(symbol):
    try:
        return re.match(r"[A-Z]+", symbol).group()
    except:
        return symbol




def rollover_sleep(mt5, symbol):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return

    # ==========================================
    # GET MT5 SERVER TIME
    # ==========================================
    print("tick.time :", tick.time)

    server_time = datetime.fromtimestamp(tick.time)
    print(server_time)

    LOGGER.info(f"MT5 SERVER TIME: {server_time}")

    hour = server_time.hour

    # ==========================================
    # CHECK ROLLOVER WINDOW
    # ==========================================

    if hour >= 23 or hour < 3:

        LOGGER.info("ROLL OVER PERIOD DETECTED")

        # ======================================
        # CREATE TARGET WAKE TIME
        # ======================================

        wake_time = server_time.replace(hour=3, minute=10, second=40, microsecond=0)

        # ======================================
        # HANDLE NEXT DAY CASE
        # ======================================

        if hour >= 23:

            wake_time += timedelta(days=1)

        # ======================================
        # CALCULATE SLEEP SECONDS
        # ======================================

        sleep_seconds = (wake_time - server_time).total_seconds()

        if sleep_seconds > 0:

            LOGGER.info(f"SLEEPING FOR {sleep_seconds:.0f} SECONDS")

            time.sleep(sleep_seconds)

        LOGGER.info("ROLLOVER FINISHED")




def wait_for_new_candle(mt5, symbol, timeframe):

    TIMEFRAME_MINUTES = 10  # 10  = 10 minutes

    LOGGER.info(f"\n [WAITING FOR NEW CANDLE : {symbol} | {TIMEFRAME_MINUTES}M]\n")

    # ========================================================
    # STAGE 1
    # Efficient local waiting
    # ========================================================

    while True:

        now = datetime.now()

        minute = now.minute
        second = now.second

        # ----------------------------------------------------
        # Calculate seconds elapsed inside current timeframe
        # ----------------------------------------------------

        total_seconds = (minute * 60) + second

        timeframe_seconds = TIMEFRAME_MINUTES * 60

        elapsed = total_seconds % timeframe_seconds

        seconds_remaining = timeframe_seconds - elapsed

        # ----------------------------------------------------
        # Exact candle boundary
        # ----------------------------------------------------

        if seconds_remaining == timeframe_seconds:
            seconds_remaining = 0

        # ----------------------------------------------------
        # Start precise MT5 checking near candle boundary
        # ----------------------------------------------------

        if seconds_remaining <= 10:
            LOGGER.info("[STARTING PRECISE MT5 CHECK]\n")
            break

        # ----------------------------------------------------
        # Sleep efficiently
        # ----------------------------------------------------

        sleep_time = max(seconds_remaining - 5, 1)

        LOGGER.info(
            f"Waiting efficiently... "
            f"{seconds_remaining}s remaining | "
            f"Sleeping {sleep_time}s"
        )

        time.sleep(sleep_time)

    # ========================================================
    # STAGE 2
    # Precise MT5 synchronization
    # ========================================================

    last_candle = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)

    if last_candle is None or len(last_candle) == 0:
        raise RuntimeError(f"Failed to fetch MT5 candle for {symbol}")

    last_time = last_candle[0]["time"]

    # ========================================================
    # Wait until MT5 actually reports a new candle
    # ========================================================

    while True:

        current_candle = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)

        if current_candle is None or len(current_candle) == 0:
            time.sleep(0.5)
            continue

        current_time = current_candle[0]["time"]

        # ----------------------------------------------------
        # NEW CANDLE DETECTED
        # ----------------------------------------------------

        if current_time != last_time:
            LOGGER.info(
                f">>>> NEW {TIMEFRAME_MINUTES}M CANDLE DETECTED : {symbol} <<<<\n"
            )
            return current_time

        time.sleep(0.3)


def forex_market_open(mt5, symbol):

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return False

    server_time = datetime.fromtimestamp(tick.time)

    weekday = server_time.weekday()
    hour = server_time.hour

    # Saturday
    if weekday == 5:
        return False

    # Sunday before open
    if weekday == 6 and hour < 22:
        return False

    return True




def MULTICLASS_DATA_PREDICTION(mt5, SYMBOL,MODEL,NUM_CLASSES,LAST_NUM=3,extension_func=False):

    LOGGER.info(f"\n[DATA PREDICTION ON NEW CANDLE : {SYMBOL}]\n")

    TIMEFRAME = mt5.TIMEFRAME_M10
    N_BARS = 300

    if extension_func:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, N_BARS)

    else:
        wait_for_new_candle(mt5, SYMBOL, TIMEFRAME)
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, N_BARS)

    if rates is None or len(rates) < N_BARS:
        mt5.shutdown()
        raise RuntimeError("[[BAD]] Failed to fetch enough closed candles")

    data = pd.DataFrame(rates)
    data["Date"] = pd.to_datetime(data["time"], unit="s")

    data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
            "spread": "Spread",
        },
        inplace=True,
    )

    new_df = data[["Date", "Open", "High", "Low", "Close", "Volume", "Spread"]]
    new_df.sort_values("Date", inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    LOGGER.info('LAST 5 ROWS OF MARKET DATA')
    LOGGER.info(new_df.tail(5))

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        raise ValueError(f"Could not get symbol information for {SYMBOL}")

    POINT = symbol_info.point
    LOGGER.info(f"{SYMBOL} POINT: {POINT}")

    df = apply_features(df=new_df, point=POINT)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    MODEL_model = MODEL["model"]
    MODEL_cols = MODEL["features"]
    MODEL_SETTINGS = MODEL["target_settings"]

    current_candle = df.tail(1)
    if current_candle.empty:
        raise ValueError(f"[{SYMBOL}] CURRENT CANDLE IS EMPTY")

    missing_features = [ col for col in MODEL_cols if col not in current_candle.columns]

    if missing_features:
        raise ValueError(f"[{SYMBOL}] MISSING MODEL FEATURES: {missing_features}")


    X_MODEL = current_candle[MODEL_cols]

    # Safety check
    assert set(MODEL_cols) == set(X_MODEL.columns)

    MODEL_probabilities = MODEL_model.predict_proba(X_MODEL)

    # ============================================================
    # CHECK MODEL CLASSES
    # ============================================================

    if MODEL_probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(f"\n[[BAD]] Model returned {MODEL_probabilities.shape[1]} classes, "
            f"but {NUM_CLASSES} classes were expected."
        )

    MODEL_classes = MODEL_model.classes_

    if len(MODEL_classes) != NUM_CLASSES:
        raise ValueError(f"\n[[BAD]] Model classes = {MODEL_classes}, but NUM_CLASSES = {NUM_CLASSES}")

    MODEL_classes = MODEL_model.classes_


    # ============================================================
    # HISTORICAL MODEL PREDICTIONS FOR THE LAST LAST_NUM CANDLES
    # ============================================================


    # # Historical candles only, excluding the current prediction candle
    # last_preds = df.iloc[-(last_num + 1):-1]

    if LAST_NUM > len(df):
        LAST_NUM = len(df)
        LOGGER.info(f">>> SELECTED LAST NUMBER > DF, DEFAULT BACK TO LENGHT OF DF" )
        
    last_preds = df.tail(LAST_NUM)
    if last_preds.empty:
        raise ValueError(f"[{SYMBOL}] CURRENT CANDLE IS EMPTY")

    X_LAST_PRED = last_preds[MODEL_cols]

    # Predict every row in the selected historical window
    HISTORY_probabilities = MODEL_model.predict_proba(X_LAST_PRED)

    if HISTORY_probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(f"[{SYMBOL}] Historical prediction returned {HISTORY_probabilities.shape[1]} classes,but {NUM_CLASSES} classes were expected.")

    # Create a separate DataFrame so the original df is not modified
    HISTORICAL_PREDICTIONS = last_preds.copy()

    # Assign each class its probability for every historical candle
    for class_index, class_id in enumerate(MODEL_classes):
        HISTORICAL_PREDICTIONS[f"CLASS_{int(class_id)}_PROBABILITY"] = HISTORY_probabilities[:, class_index]

    # Sum DOWN probabilities: Class 0 + Class 1
    HISTORICAL_PREDICTIONS["DOWN_PROBABILITY"] = (
        HISTORICAL_PREDICTIONS["CLASS_0_PROBABILITY"] + HISTORICAL_PREDICTIONS["CLASS_1_PROBABILITY"]
    )

    # Sum UP probabilities: Class 3 + Class 4
    HISTORICAL_PREDICTIONS["UP_PROBABILITY"] = (
        HISTORICAL_PREDICTIONS["CLASS_3_PROBABILITY"] + HISTORICAL_PREDICTIONS["CLASS_4_PROBABILITY"]
    )

    # Class 2 remains NEUTRAL
    HISTORICAL_PREDICTIONS["NEUTRAL_PROBABILITY"] = (HISTORICAL_PREDICTIONS["CLASS_2_PROBABILITY"])

    # Get the highest-probability predicted class for each candle
    HISTORY_class_indices = np.argmax(HISTORY_probabilities, axis=1)
    HISTORICAL_PREDICTIONS["PREDICTED_CLASS"] = [ int(MODEL_classes[index]) for index in HISTORY_class_indices]
    HISTORICAL_PREDICTIONS["PREDICTED_PROBABILITY"] = (
        HISTORY_probabilities[np.arange(len(HISTORY_probabilities)),HISTORY_class_indices]
    )

    # Map each historical prediction to its saved direction
    HISTORICAL_PREDICTIONS["PREDICTED_DIRECTION"] = (
        HISTORICAL_PREDICTIONS["PREDICTED_CLASS"].map(
            lambda class_id: MODEL_SETTINGS["classes"][class_id]["direction"]
        )
    )

    # Get the most recent row's historical directional probabilities
    LAST_HISTORY_ROW = HISTORICAL_PREDICTIONS.iloc[-1]

    historical_up_probability = float(
        LAST_HISTORY_ROW["UP_PROBABILITY"]
    )

    historical_down_probability = float(
        LAST_HISTORY_ROW["DOWN_PROBABILITY"]
    )

    historical_neutral_probability = float(
        LAST_HISTORY_ROW["NEUTRAL_PROBABILITY"]
    )

    # Log the historical predictions
    LOGGER.info(
        f"\n"
        f"========== HISTORICAL MODEL PREDICTIONS ==========\n"
        f"SYMBOL                   : {SYMBOL}\n"
        f"REQUESTED LAST NUM       : {LAST_NUM}\n"
        f"ACTUAL ROWS CHECKED      : {len(HISTORICAL_PREDICTIONS)}\n"
        f"--------------------------------------------------\n"
        f"{HISTORICAL_PREDICTIONS[['Date', 'PREDICTED_CLASS', 'PREDICTED_DIRECTION', 'PREDICTED_PROBABILITY', 'DOWN_PROBABILITY', 'UP_PROBABILITY']].to_string(index=False)}\n"
        f"--------------------------------------------------\n"
        f"LATEST UP PROBABILITY    : {historical_up_probability:.6f}\n"
        f"LATEST DOWN PROBABILITY  : {historical_down_probability:.6f}\n"
        f"LATEST NEUTRAL PROB.     : {historical_neutral_probability:.6f}\n"
        f"==================================================\n"
    )



    # ============================================================
    # GET PROBABILITY OF EACH CLASS
    # ============================================================

    class_probabilities = {
        int(class_id): float(probability)
        for class_id, probability in zip(MODEL_classes,MODEL_probabilities[0])
    }


    # ============================================================
    # GET HIGHEST-PROBABILITY CLASS
    # ============================================================

    probability_index = np.argmax(MODEL_probabilities[0])

    predicted_class = int(MODEL_classes[probability_index])

    predicted_probability = float(MODEL_probabilities[0][probability_index])

    # ============================================================
    # CLASS INFORMATION SAVED DURING TRAINING
    # ============================================================

    CLASS_INFO = MODEL_SETTINGS["classes"]

    predicted_class_info = CLASS_INFO[predicted_class]

    predicted_class_name = predicted_class_info["name"]

    predicted_direction = predicted_class_info["direction"]

    median_movement_pct = float(predicted_class_info["median_movement_pct"])

    # ============================================================
    # LOG RESULT
    # ============================================================

    LOGGER.info(
        f"\n"
        f"========== MULTICLASS PREDICTION ==========\n"
        f"SYMBOL              : {SYMBOL}\n"
        f"CANDLE              : {current_candle['Date'].iloc[0]}\n"
        f"PREDICTED CLASS     : {predicted_class}\n"
        f"CLASS NAME          : {predicted_class_name}\n"
        f"DIRECTION           : {predicted_direction}\n"
        f"PROBABILITY         : {predicted_probability:.6f} "
        f"({predicted_probability * 100:.2f}%)\n"
        f"MEDIAN MOVEMENT     : {median_movement_pct:.6f}%\n"
        f"============================================\n"
    )

    return {
        "symbol": SYMBOL,
        "candle_time": current_candle["Date"].iloc[0],

        "predicted_class": predicted_class,
        "predicted_class_name": predicted_class_name,
        "predicted_direction": predicted_direction,

        "predicted_probability": predicted_probability,
        "median_movement_pct": median_movement_pct,

        'Data_Frame':df,
        'CLASS_INFO':CLASS_INFO,

        "class_probabilities": class_probabilities,

        # Historical predictions for the selected last_num candles
        "historical_predictions": HISTORICAL_PREDICTIONS,

        # Directional probabilities of the most recent historical row
        "historical_up_probability": historical_up_probability,
        "historical_down_probability": historical_down_probability,
        "historical_neutral_probability": historical_neutral_probability,

    }






def calc_lot_size(mt5, balance, risk_percent, sl_pips, pip_value_per_lot, SYMBOL):

    info = mt5.symbol_info(SYMBOL)

    min_lot = info.volume_min
    max_lot = info.volume_max
    step = info.volume_step

    risk_amount = balance * (risk_percent / 100)
    lot_cal = risk_amount / (sl_pips * pip_value_per_lot)

    lot = max(min_lot, min(lot_cal, max_lot))
    lot = np.floor(lot / step) * step

    return lot


def normalize_lot(lot, vol_min, vol_max, vol_step):
    lot = max(vol_min, min(lot, vol_max))
    lot = np.floor(lot / vol_step) * vol_step
    return round(lot, 2)


def split_pair(symbol):

    base = symbol[:3]
    quote = symbol[3:]

    return base, quote


def get_trade_bias(symbol, trade_type):
    """
    Returns currency exposure.

    BUY EURUSD:
        EUR = +1
        USD = -1

    SELL EURUSD:
        EUR = -1
        USD = +1
    """

    base, quote = split_pair(symbol)

    if trade_type == "BUY":

        return {base: 1, quote: -1}

    else:

        return {base: -1, quote: 1}


def correlation_check(mt5, new_symbol, new_trade_type):

    positions = mt5.positions_get()

    if positions is None:
        return True

    new_bias = get_trade_bias(new_symbol, new_trade_type)

    total_exposure = {}

    # ==========================================
    # EXISTING OPEN TRADES
    # ==========================================

    for pos in positions:

        symbol = pos.symbol

        if pos.type == mt5.ORDER_TYPE_BUY:
            trade_type = "BUY"

        else:
            trade_type = "SELL"

        bias = get_trade_bias(symbol, trade_type)

        for currency, value in bias.items():

            if currency not in total_exposure:
                total_exposure[currency] = 0

            total_exposure[currency] += value

    # ==========================================
    # CHECK NEW TRADE IMPACT
    # ==========================================

    for currency, value in new_bias.items():

        current = total_exposure.get(currency, 0)

        new_total = current + value

        # ======================================
        # OVEREXPOSURE LIMIT
        # ======================================

        if abs(new_total) > 2:

            LOGGER.info(f"BLOCKED: Too much exposure on {currency}")

            return False

    return True


def spread_filter(mt5, SYMBOL, SL_distance):

    tick = mt5.symbol_info_tick(SYMBOL)

    if tick is None:
        LOGGER.info("[[BAD]] No tick data")
        return False

    ask_price = tick.ask
    bid_price = tick.bid

    spread = ask_price - bid_price
    spread_ratio = spread / SL_distance

    symbol_info = mt5.symbol_info(SYMBOL)
    spread_points = spread / symbol_info.point

    sl_pips = SL_distance / symbol_info.point

    LOGGER.info(f"""
    ====================================================================
    [[ ASK VALUE]] :{ask_price}
    [[ BID VALUE]] :{bid_price}
    [[WARNING]]>>>>>> ACTUAL SPREAD VALUE IS {spread} <<<<<<<<<<<<<<     
    Spread Ratio: {spread_ratio:.2%}    
    [[WARNING]] SPREAD POINT VALUE IS { round(spread_points,2) } PIPS   
    SL DISTANCE   : {sl_pips:.2f} pips
    ====================================================================
    """)

    min_stop_distance = symbol_info.point * 10
    if SL_distance < min_stop_distance:
        LOGGER.info("[[BAD]] SL too small, skipping trade")
        return False

    if spread_ratio > 0.16:

        LOGGER.info(
            f"[[WARNING]] Spread Too Expensive Relative To SL | "
            f"Spread Ratio: {spread_ratio:.2%}"
        )

        return False
    LOGGER.info("[[GOOD]]  SPREAD PIP SIZING CONFIRMATION PASSED ")
    return True


def check_trade_result(mt5, result):
    if result is None:
        LOGGER.info("[[BAD]] Order failed: result is None")
        LOGGER.info(f"MT5 last error: {mt5.last_error()}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        LOGGER.info("[[BAD]] Order rejected")
        LOGGER.info(f"Retcode: {result.retcode}")
        LOGGER.info(f"Comment: {result.comment}")
        LOGGER.info(f"Request ID: {result.request_id}")
        return False

    LOGGER.info("[[GOOD]] Trade placed successfully")
    LOGGER.info(f"Order Ticket: {result.order}")
    LOGGER.info(f"Deal Ticket: {result.deal}")
    LOGGER.info(f"Volume: {result.volume}")
    LOGGER.info(f"Price: {result.price}")
    return True


def get_symbol_volume_info(mt5, symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError("Failed to get symbol info")

    return {"min": info.volume_min, "max": info.volume_max, "step": info.volume_step}


def place_sell(mt5, symbol, lot, entry_price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 25,
        "magic": 10002,
        "comment": "Auto SELL",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = mt5.order_send(request)
    check_trade_result(mt5, result)
    return result


def place_buy(mt5, symbol, lot, entry_price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "deviation": 25,
        "magic": 10001,
        "comment": "Auto BUY",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = mt5.order_send(request)
    check_trade_result(mt5, result)
    return result


def drop_duplicate(path):
    all_df = pd.read_csv(path)
    all_df = all_df.drop_duplicates(keep="first")
    all_df = all_df.reset_index()
    all_df.drop(["index"], axis=1, inplace=True)
    all_df.to_csv(path, index=False)


def get_pip_info(mt5, symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol info not found for {symbol}")

    tick_size = info.trade_tick_size
    tick_value = info.trade_tick_value
    digits = info.digits

    # Determine pip size
    if digits in (3, 5):
        pip_size = tick_size * 10
    else:
        pip_size = tick_size

    # Pip value per 1 lot
    pip_value_per_lot = (pip_size / tick_size) * tick_value

    return {
        "pip_size": pip_size,
        "pip_value_per_lot": pip_value_per_lot,
        "tick_size": tick_size,
        "tick_value": tick_value,
    }



def MULTICLASS_LOG_TRADE(mt5,symbol,direction,entry_price,SL,TP,lot_size,model_class,class_direction,class_proba,order_result,):

    BASE_DIR = Path(__file__).resolve().parent
    CSV_DIR = BASE_DIR / "CSV_FILES"
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    TRADE_LOG_FILE = CSV_DIR / "MULTICLASS_TRADE_LOG.csv"

    account_info = mt5.account_info()
    current_balance = account_info.balance

    # Load existing log if it exists
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        prev_balance = df.iloc[-1]["Balance"] if not df.empty else current_balance
    else:
        df = pd.DataFrame()
        prev_balance = current_balance

    # Determine PnL status
    if current_balance > prev_balance:
        pnl_status = "Profit"
    elif current_balance < prev_balance:
        pnl_status = "Loss"
    else:
        pnl_status = "No Change"

    # Create new row
    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Direction": direction,
        "Entry": entry_price,
        "SL": SL,
        "TP": TP,
        "Lot": lot_size,
        "Model_Class": model_class,
        "Class_Direction": class_direction,
        "Class_Proba": class_proba,
        "OrderResult": order_result,
        "Balance": current_balance,
        "PnL_Status": pnl_status,
    }

    # Append row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save back to CSV
    df.to_csv(TRADE_LOG_FILE, index=False)

    LOGGER.info(f"[[GOOD]] Trade logged. Current Balance: {current_balance}, Status: {pnl_status}")



def saving_files(data, path):
    df = pd.DataFrame(data)
    LOGGER.info(df.to_string())

    try:
        df2 = pd.read_csv(path)
        all_df = pd.concat([df2, df], ignore_index=True)
        all_df.to_csv(path, index=False)
        LOGGER.info(" ------------------------------------ ALL FILES SAVED  ------------------------------------- \n \n")

    except:
        df.to_csv(path, index=False)
        LOGGER.info("============================= SECOND FILE SAVED ==========================")


def get_monday(date_obj):
    return date_obj - timedelta(days=date_obj.weekday())


def ff_week_format(date_obj):
    return (date_obj.strftime("%b").lower() + str(date_obj.day) + "." + str(date_obj.year))


def ff_day_format(date_obj):
    return (
        date_obj.strftime("%b").lower() + str(date_obj.day) + "." + str(date_obj.year)
    )


def normalize_datetime(date_str, time_str, year):
    if not date_str or not time_str:
        return None

    if time_str.lower() == "all day":
        time_str = "12:00am"

    dt_str = f"{date_str} {year} {time_str}"

    try:
        dt = datetime.strptime(dt_str, "%a %b %d %Y %I:%M%p")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None


def build_5class_target(df,target_horizon=18,neutral_percentile=0.30,
    strong_percentile=0.80,target_name="TARGET_5CLASS",drop_ambiguous=False,print_info=True,):
    
    """
    Build a 5-class directional + magnitude target.

    Classes
    -------
    0 = Strong DOWN
    1 = Moderate DOWN
    2 = Small / NEUTRAL
    3 = Moderate UP
    4 = Strong UP

    Also calculates class-specific historical movement statistics
    for later use in live/forward TP/SL placement.
    """

    required_columns = {"Close", "High", "Low"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing_columns)}")

    if not isinstance(target_horizon, int) or target_horizon <= 0:
        raise ValueError("target_horizon must be a positive integer.")

    if not 0 < neutral_percentile < 1:
        raise ValueError("neutral_percentile must be between 0 and 1.")

    if not 0 < strong_percentile < 1:
        raise ValueError("strong_percentile must be between 0 and 1.")

    if neutral_percentile >= strong_percentile:
        raise ValueError("neutral_percentile must be smaller than strong_percentile.")

    if len(df) <= target_horizon:
        raise ValueError(
            f"DataFrame has only {len(df)} rows, but "
            f"target_horizon={target_horizon} requires more data."
        )

    df_result = df.copy()

    usable_rows = len(df_result) - target_horizon

    close = df_result["Close"].to_numpy(dtype=float)
    high = df_result["High"].to_numpy(dtype=float)
    low = df_result["Low"].to_numpy(dtype=float)

    dominant_directions = np.full(usable_rows, -1, dtype=np.int8)

    dominant_movements = np.zeros(usable_rows, dtype=float)

    dominant_percentages = np.zeros(usable_rows, dtype=float)

    ambiguous_mask = np.zeros(usable_rows, dtype=bool)

    # ==========================================================
    # BUILD RAW DOMINANT MOVEMENT
    # ==========================================================

    for i in range(usable_rows):

        entry = close[i]

        future_high = high[i + 1 : i + 1 + target_horizon]
        future_low = low[i + 1 : i + 1 + target_horizon]

        highest_price = np.max(future_high)
        lowest_price = np.min(future_low)

        up_excursion = highest_price - entry
        down_excursion = entry - lowest_price

        if up_excursion > down_excursion:

            dominant_directions[i] = 1
            dominant_move = up_excursion

        elif down_excursion > up_excursion:

            dominant_directions[i] = 0
            dominant_move = down_excursion

        else:

            dominant_directions[i] = -1
            dominant_move = 0.0
            ambiguous_mask[i] = True

        dominant_movements[i] = dominant_move

        if entry > 0:

            dominant_percentages[i] = dominant_move / entry

        else:

            dominant_percentages[i] = np.nan

    # ==========================================================
    # CALCULATE PERCENTILE BOUNDARIES
    # ==========================================================

    valid_movement_percentages = dominant_percentages[np.isfinite(dominant_percentages)]

    if len(valid_movement_percentages) == 0:
        raise ValueError("No valid dominant movement percentages were calculated.")

    neutral_threshold = np.quantile(valid_movement_percentages, neutral_percentile)

    strong_threshold = np.quantile(valid_movement_percentages, strong_percentile)

    # ==========================================================
    # ASSIGN 5 CLASSES
    # ==========================================================

    labels = np.full(usable_rows, np.nan)

    for i in range(usable_rows):

        direction = dominant_directions[i]
        dominant_pct = dominant_percentages[i]

        if direction == -1:

            if drop_ambiguous:
                labels[i] = np.nan
            else:
                labels[i] = 2

            continue

        if dominant_pct < neutral_threshold:

            labels[i] = 2

        elif dominant_pct < strong_threshold:

            if direction == 1:
                labels[i] = 3
            else:
                labels[i] = 1

        else:

            if direction == 1:
                labels[i] = 4
            else:
                labels[i] = 0

    # ==========================================================
    # ADD TARGET TO DATAFRAME
    # ==========================================================

    target_values = np.full(len(df_result), np.nan)

    target_values[:usable_rows] = labels

    df_result[target_name] = target_values

    # ==========================================================
    # CLASS DEFINITIONS
    # ==========================================================

    class_definitions = {
        0: {"name": "Strong DOWN", "direction": "DOWN"},
        1: {"name": "Moderate DOWN", "direction": "DOWN"},
        2: {"name": "Small / NEUTRAL", "direction": "NEUTRAL"},
        3: {"name": "Moderate UP", "direction": "UP"},
        4: {"name": "Strong UP", "direction": "UP"},
    }

    # ==========================================================
    # CALCULATE CLASS MOVEMENT STATISTICS
    # ==========================================================

    class_statistics = {}

    valid_label_mask = np.isfinite(labels)

    for class_id in range(5):

        class_mask = valid_label_mask & (labels == class_id)

        class_movements = dominant_percentages[class_mask]

        class_movements = class_movements[np.isfinite(class_movements)]

        if len(class_movements) == 0:

            class_statistics[class_id] = {
                **class_definitions[class_id],
                "count": 0,
                "median_movement_pct": np.nan,
                "mean_movement_pct": np.nan,
                "p25_movement_pct": np.nan,
                "p75_movement_pct": np.nan,
            }

            continue

        class_statistics[class_id] = {
            **class_definitions[class_id],
            "count": int(len(class_movements)),
            "median_movement_pct": float(np.median(class_movements) * 100),
            "mean_movement_pct": float(np.mean(class_movements) * 100),
            "p25_movement_pct": float(np.percentile(class_movements, 25) * 100),
            "p75_movement_pct": float(np.percentile(class_movements, 75) * 100),
        }

    # ==========================================================
    # COMPLETE TARGET SETTINGS
    # ==========================================================

    target_settings = {
        "target_name": target_name,
        "target_horizon": target_horizon,
        "neutral_percentile": neutral_percentile,
        "strong_percentile": strong_percentile,
        "neutral_threshold_pct": float(neutral_threshold * 100),
        "strong_threshold_pct": float(strong_threshold * 100),
        "neutral_threshold_fraction": float(neutral_threshold),
        "strong_threshold_fraction": float(strong_threshold),
        "classes": class_statistics,
    }

    # ==========================================================
    # PRINT INFORMATION
    # ==========================================================

    if print_info:

        valid_labels = labels[np.isfinite(labels)].astype(int)

        class_counts = pd.Series(valid_labels).value_counts().sort_index()

        class_percentages = class_counts / len(valid_labels) * 100

        print()
        print("=" * 70)
        print(f"5-CLASS TARGET: {target_name}")
        print("=" * 70)

        print(f"TARGET HORIZON       : " f"{target_horizon} candles")

        print(f"NEUTRAL PERCENTILE   : " f"{neutral_percentile:.0%}")

        print(f"STRONG PERCENTILE    : " f"{strong_percentile:.0%}")

        print()

        print(f"NEUTRAL THRESHOLD    : " f"{neutral_threshold * 100:.6f}%")

        print(f"STRONG THRESHOLD     : " f"{strong_threshold * 100:.6f}%")

        print()

        print("CLASS DISTRIBUTION")
        print("-" * 70)

        for class_id in range(5):

            count = class_counts.get(class_id, 0)

            percentage = class_percentages.get(class_id, 0.0)

            name = class_definitions[class_id]["name"]

            median_pct = class_statistics[class_id]["median_movement_pct"]

            print(
                f"Class {class_id} "
                f"{name:<18} : "
                f"{count:>8,} "
                f"({percentage:>6.2f}%) "
                f"| Median Move: "
                f"{median_pct:.6f}%"
            )

        print()

        print(f"AMBIGUOUS EQUAL MOVEMENT: " f"{ambiguous_mask.sum():,}")

        print("=" * 70)
        print()

    return df_result, target_settings


def confirmation_entry(df, direction):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if direction == "BUY":
        return last["Close"] > last["Open"] and last["Close"] > prev["Close"]

    elif direction == "SELL":
        return last["Close"] < last["Open"] and last["Close"] < prev["Close"]

    return False




def MULTICLASS_FORWARD_TEST_THRESHOLD(SYMBOL):
    if 'AUDCAD' in SYMBOL :
        W_threshold = 0.399

    elif 'AUDJPY' in SYMBOL:
        W_threshold = 0.2551

    elif 'AUDUSD' in SYMBOL:
        W_threshold = 0.312

    elif 'BTCUSD' in SYMBOL :
        W_threshold = 0.276

    elif 'CADJPY' in SYMBOL :
        W_threshold = 0.364

    elif 'ETHUSD' in SYMBOL:
        W_threshold = 0.296

    elif 'EURGBP' in SYMBOL:
        W_threshold = 0.350

    elif 'EURJPY' in SYMBOL:
        W_threshold = 0.418

    elif 'EURUSD' in SYMBOL:
        W_threshold = 0.338

    elif 'GBPCAD' in SYMBOL:
        W_threshold = 0.294

    elif 'GBPJPY' in SYMBOL:
        W_threshold = 0.306

    elif 'GBPUSD' in SYMBOL:
        W_threshold = 0.323

    elif 'NZDJPY' in SYMBOL:
        W_threshold = 0.347

    elif 'NZDUSD' in SYMBOL:
        W_threshold = 0.330

    elif 'USDCAD' in SYMBOL:
        W_threshold = 0.338

    elif 'USDCHF' in SYMBOL:
        W_threshold = 0.303

    elif 'USDJPY' in SYMBOL:
        W_threshold = 0.287

    elif 'XAGUSD' in SYMBOL:
        W_threshold = 0.647

    elif 'XAUUSD' in SYMBOL:
        W_threshold = 0.552
        
    else:
        W_threshold = 1.0 # DIRECTION


def normalize_volume(volume, min_lot, step):
    return round(max(min_lot, round(volume / step) * step), 2)


def MULTICLASS_Entry_Filtering(mt5, SYMBOL, df, direction, median_movement_pct):

    # ============================================================
    # 1. GET INITIAL MARKET PRICE
    # ============================================================
    tick = mt5.symbol_info_tick(SYMBOL)

    if tick is None:
        LOGGER.info("[[BAD]] Could not get market tick")
        return None

    entry_base = tick.ask if direction == "BUY" else tick.bid

    # median_movement_pct is a FRACTION
    # Example: 0.00165 = 0.165%
    #
    # Convert percentage movement into an actual PRICE distance.
    median_movement_price = entry_base * median_movement_pct

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ============================================================
    # 2. MOMENTUM STRENGTH
    # ============================================================
    body_size = abs(last["Close"] - last["Open"])

    if body_size < 0.10 * median_movement_price:
        LOGGER.info("[[BAD]] Weak momentum candle")
        return None

    # ============================================================
    # 3. MARKET DIRECTION CONFIRMATION
    # ============================================================
    if direction == "BUY":
        if last["Close"] <= prev["Close"]:
            LOGGER.info("[[BAD]] MARKET MOVEMENT IS NOT CONFIRMING THE BUY SIGNAL")
            return None

    elif direction == "SELL":
        if last["Close"] >= prev["Close"]:
            LOGGER.info("[[BAD]] MARKET MOVEMENT IS NOT CONFIRMING THE SELL SIGNAL")
            return None

    else:
        LOGGER.info(f"[[BAD]] Invalid direction: {direction}")
        return None

    LOGGER.info("[[GOOD]] Confirmation passed")

    # ============================================================
    # 4. STRUCTURED PULLBACK SETTINGS
    # ============================================================
    pullback_ratio = 0.20
    invalid_ratio = 0.70
    reversal_pullback_ratio = 0.30
    momentum_entry_ratio = 0.20

    max_wait = 60 * 120       # 120 minutes

    # Convert all ratios into actual PRICE distances.
    pullback_distance = pullback_ratio * median_movement_price
    invalidation_distance = invalid_ratio * median_movement_price
    reversal_distance = reversal_pullback_ratio * median_movement_price
    momentum_entry_distance = momentum_entry_ratio * median_movement_price

    # ============================================================
    # 5. DEFINE ENTRY BASE
    # ============================================================
    entry_base = tick.ask if direction == "BUY" else tick.bid

    pullback_level = (entry_base - pullback_distance if direction == "BUY" else entry_base + pullback_distance)
    invalidation_level = ( entry_base - invalidation_distance if direction == "BUY" else entry_base + invalidation_distance)

    LOGGER.info(f"[[TARGET]] Entry base: {entry_base} | Median movement: {median_movement_price} | "
        f"Pullback: {pullback_level} | Invalidation: {invalidation_level}"
    )

    pullback_hit = False

    # For BUY  -> tracks lowest price
    # For SELL -> tracks highest price
    extreme_price = entry_base

    # ============================================================
    # 6. WAIT FOR PULLBACK / REVERSAL / MOMENTUM
    # ============================================================
    for _ in range(max_wait):

        tick = mt5.symbol_info_tick(SYMBOL)

        if tick is None:
            LOGGER.info("[[WARNING]] Failed to get current tick")
            time.sleep(1)
            continue

        price = tick.ask if direction == "BUY" else tick.bid

        # ========================================================
        # INVALIDATION
        # ========================================================
        if direction == "BUY" and price <= invalidation_level:
            LOGGER.info("BUY : PRICE HIT INVALIDATION POINT, SO TRADE IS ABANDONED")
            return None

        if direction == "SELL" and price >= invalidation_level:
            LOGGER.info( "SELL : PRICE HIT INVALIDATION POINT, SO TRADE IS ABANDONED")
            return None

        # ========================================================
        # TRACK EXTREME PRICE
        # ========================================================
        if direction == "BUY":

            # Track lowest price during pullback
            if price < extreme_price:
                extreme_price = price

            # Pullback reached
            if price <= pullback_level:
                pullback_hit = True

            # Reversal + recovery back to entry
            if (pullback_hit and price > extreme_price + reversal_distance and price >= entry_base):
                LOGGER.info("[[GOOD]] Confirmed pullback entry (BUY)")
                return price

        else:

            # Track highest price during pullback
            if price > extreme_price:
                extreme_price = price

            # Pullback reached
            if price >= pullback_level:
                pullback_hit = True

            # Reversal + recovery back to entry
            if (pullback_hit and price < extreme_price - reversal_distance and price <= entry_base):
                LOGGER.info("[[GOOD]] Confirmed pullback entry (SELL)")
                return price


        # ========================================================
        # MOMENTUM ENTRY
        # ========================================================
        if (direction == "BUY" and price >= entry_base + momentum_entry_distance):
            LOGGER.info("[[GOOD]] Confirmed clean movement entry (BUY)")
            return price

        if (direction == "SELL" and price <= entry_base - momentum_entry_distance):
            LOGGER.info("[[GOOD]] Confirmed clean movement entry (SELL)" )
            return price

        time.sleep(1)

    # ============================================================
    # 7. NO VALID ENTRY
    # ============================================================
    LOGGER.info("[[WARNING]] No clean pullback or momentum entry → skip trade")

    return None




MULTICLASS_TRADE_STATE = {}  # ticket -> state

def MULTICLASS_MOVE_SL_AND_PARTIAL_CLOSE(mt5, SYMBOL ,MODEL, NUM_CLASSES):

    PARTIAL_CLOSE_PCT = 0.4  # AMOUT OF LOT SIZE THAT CAN BE CLOSED E.G 0.4 = 40% OF LOT SIZE
    PARTIAL_CLOSE_LEVEL = 0.35  # LEVEL THAT PARTIAL PROFIT WILL BE TRIGGERED

    MIN_TICKS = 1.1
    MAX_TP_EXTENSION_MULT = 2  # MAXIMUN DISTANCE THE NEW TP CAN BE EXTENDED TO 

    trades = mt5.positions_get(symbol=SYMBOL)
    if trades is None or len(trades) == 0:
        return 1

    symbol_info = mt5.symbol_info(SYMBOL)

    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    tick_size = symbol_info.trade_tick_size
    point = symbol_info.point
    base_unit = tick_size if tick_size and tick_size > 0 else point

    min_sl_distance = MIN_TICKS * base_unit

    # =========================
    # CLEAN CLOSED TRADES
    # =========================
    active_tickets = {t.ticket for t in trades}

    for ticket in list(MULTICLASS_TRADE_STATE.keys()):
        if ticket not in active_tickets:
            del MULTICLASS_TRADE_STATE[ticket]

    # =========================
    # LOOP THROUGH TRADES
    # =========================
    for trade in trades:

        tick = mt5.symbol_info_tick(SYMBOL)

        ticket = trade.ticket
        entry = trade.price_open
        sl = trade.sl
        tp = trade.tp
        lot = trade.volume
        order_type = trade.type

        price = tick.bid if order_type == mt5.ORDER_TYPE_BUY else tick.ask

        # =========================
        # INIT STATE
        # =========================
        if ticket not in MULTICLASS_TRADE_STATE:
            MULTICLASS_TRADE_STATE[ticket] = {
                "original_tp": tp,
                "partial_done": False,
                "extended": False,
                "moved_sl": False,
            }

        state = MULTICLASS_TRADE_STATE[ticket]
        original_tp = state["original_tp"]

        # =========================
        # TRIGGER LEVEL
        # =========================
        distance = abs(original_tp - entry)

        trigger_level = (
            entry + (PARTIAL_CLOSE_LEVEL * distance)
            if order_type == mt5.ORDER_TYPE_BUY
            else entry - (PARTIAL_CLOSE_LEVEL * distance)
        )

        trigger_hit = (order_type == mt5.ORDER_TYPE_BUY and price >= trigger_level) or (
            order_type == mt5.ORDER_TYPE_SELL and price <= trigger_level
        )

        if not trigger_hit:
            continue

        # =========================
        # 1. PARTIAL CLOSE (SAFE)
        # =========================
        if not state["partial_done"]:

            raw_close = lot * PARTIAL_CLOSE_PCT
            close_lots = normalize_volume(raw_close, min_lot, lot_step)

            # [[ALARM]] Prevent full close
            if close_lots >= lot:
                close_lots = normalize_volume(lot - min_lot, min_lot, lot_step)

            if close_lots <= 0:
                LOGGER.info("[[BAD]] Invalid partial close volume")
                continue

            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": close_lots,
                "type": (
                    mt5.ORDER_TYPE_SELL
                    if order_type == mt5.ORDER_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY
                ),
                "position": ticket,
                "price": price,
                "deviation": 50,
                "magic": trade.magic,
                "comment": "Partial close",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            result = mt5.order_send(close_request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                LOGGER.info(f"[[GOOD]] Partial close success: {close_lots}")
                state["partial_done"] = True
            else:
                LOGGER.info(f"[[BAD]] Partial close failed: {result.retcode if result else 'No response'}")
                continue  # don't proceed if failed

        # =====================================
        # 2. MOVE SL TO BREAKEVEN
        # =====================================
        if not state["moved_sl"]:

            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": entry,  # >>>> new_sl, <<<<<< USE FOR LITTLE BUFFER
                "tp": original_tp,
            }

            result = mt5.order_send(modify_request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                LOGGER.info("[[GOOD]] SL moved to BE+buffer")
                state["moved_sl"] = True
            else:
                LOGGER.info("[[BAD]] SL move failed")



        # ======================================
        # 3. TP EXTENSION
        # ======================================
        if state["extended"]:
            continue

        MULTICLASS_MODEL_2 = MULTICLASS_DATA_PREDICTION(mt5=mt5,SYMBOL=SYMBOL,MODEL=MODEL,NUM_CLASSES=NUM_CLASSES,extension_func=True)

        model2_class = MULTICLASS_MODEL_2["predicted_class"]
        model2_class_name = MULTICLASS_MODEL_2["predicted_class_name"]
        model2_probability = MULTICLASS_MODEL_2["predicted_probability"]

        CLASS_INFO = MULTICLASS_MODEL_2["CLASS_INFO"]

        # ============================================================
        # DETERMINE WHICH CLASS MEDIAN TO USE FOR TP EXTENSION
        # ============================================================

        extension_class = None

        if order_type == mt5.ORDER_TYPE_BUY:

            if model2_class == 4:
                # Strong UP -> use Moderate UP median
                extension_class = 3

            elif model2_class == 3:
                # Moderate UP -> use Neutral median
                extension_class = 2

            elif model2_class == 2:
                # Neutral -> NO EXTENSION
                LOGGER.info("[[WARNING]] MODEL 2 = CLASS 2 (NEUTRAL) -> NO TP EXTENSION [[ BUY ]]")
                state["extended"] = True
                continue

            else:
                # Model predicts DOWN while holding BUY
                LOGGER.info(
                    f"[[WARNING]] MODEL 2 = {model2_class} <|> ({model2_class_name}) -> "
                    f"NOT COMPATIBLE WITH BUY -> NO TP EXTENSION")
                state["extended"] = True
                continue

        elif order_type == mt5.ORDER_TYPE_SELL:

            if model2_class == 0:
                # Strong DOWN -> use Moderate DOWN median
                extension_class = 1

            elif model2_class == 1:
                # Moderate DOWN -> use Neutral median
                extension_class = 2

            elif model2_class == 2:
                # Neutral -> NO EXTENSION
                LOGGER.info("[[WARNING]] MODEL 2 = CLASS 2 (NEUTRAL) -> NO TP EXTENSION [[ SELL ]]")
                state["extended"] = True
                continue

            else:
                # Model predicts UP while holding SELL
                LOGGER.info(f"[[WARNING]] MODEL 2 = {model2_class} <|> ({model2_class_name}) -> "
                    f"NOT COMPATIBLE WITH SELL -> NO TP EXTENSION")
                state["extended"] = True
                continue

        # ============================================================
        # GET TARGET CLASS MEDIAN MOVEMENT
        # ============================================================

        extension_class_info = CLASS_INFO[extension_class]
        extension_median_pct = float(extension_class_info["median_movement_pct"])

        # Stored value is percentage points.
        # Example:
        # 0.120000 = 0.12%
        #
        # Convert to fraction:
        extension_median_fraction = extension_median_pct / 100.0

        # ============================================================
        # CALCULATE TP EXTENSION
        # ============================================================

        extension_distance = original_tp * extension_median_fraction

        if order_type == mt5.ORDER_TYPE_BUY:

            new_tp = original_tp + extension_distance

        else:

            new_tp = original_tp - extension_distance

        # ============================================================
        # MAXIMUM TP EXTENSION PROTECTION
        # ============================================================

        original_distance = abs(original_tp - entry)

        max_extension_distance = (original_distance * MAX_TP_EXTENSION_MULT)

        if order_type == mt5.ORDER_TYPE_BUY:

            max_allowed_tp = original_tp + max_extension_distance

            new_tp = min(new_tp, max_allowed_tp)

        else:

            max_allowed_tp = original_tp - max_extension_distance

            new_tp = max(new_tp, max_allowed_tp)

        # ============================================================
        # MODIFY TP
        # ============================================================

        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": entry,
            "tp": new_tp,
        }

        result = mt5.order_send(modify_request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:

            state["extended"] = True

            LOGGER.info(
                f"[[GOOD]] TP EXTENDED\n"
                f"MODEL 2 CLASS       : {model2_class}\n"
                f"MODEL 2 NAME        : {model2_class_name}\n"
                f"MODEL 2 PROBABILITY : {model2_probability:.4f}\n"
                f"EXTENSION CLASS     : {extension_class}\n"
                f"EXTENSION MEDIAN    : {extension_median_pct:.6f}%\n"
                f"ORIGINAL TP         : {original_tp}\n"
                f"NEW TP              : {new_tp}"
            )
            return 1

        else:

            LOGGER.info(
                f"[[BAD]] TP extension failed: "
                f"{result.retcode if result else 'No response'}"
            )
