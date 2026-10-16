# ===============================================================================================
# MULTI-CLASS FORWARD TEST [5 CLASSES ]
# ===============================================================================================


import json
import os
import joblib
import numpy as np
import pandas as pd

from func import apply_features, normalize_symbol

# ===============================================================================================
# SETTINGS
# ===============================================================================================

SYMBOL = "EURUSD"

CD_TIME = "10M"

TARGET_NAME = "TARGET_5CLASS"

NUM_CLASSES = 5

RISK_PERCENT = 1


# -----------------------------------------------------------------------------------------------
# SINGLE FORWARD-TEST THRESHOLD
# -----------------------------------------------------------------------------------------------

FORWARD_THRESHOLD = 0.338


# -----------------------------------------------------------------------------------------------
# 5-CLASS DEFINITIONS
# -----------------------------------------------------------------------------------------------

CLASS_NAMES = {
    0: "Strong DOWN",
    1: "Moderate DOWN",
    2: "Small / NEUTRAL",
    3: "Moderate UP",
    4: "Strong UP",
}

DOWN_CLASSES = {0, 1}

NEUTRAL_CLASS = 2

UP_CLASSES = {3, 4}


# -----------------------------------------------------------------------------------------------
# TP / SL
#
# Class-specific median movement percentage.
#
# TP / SL = 1:1
# -----------------------------------------------------------------------------------------------

TP_SL_RATIO = 1.0


# ===============================================================================================
# PATHS
# ===============================================================================================

BASE_PATH = "/content/drive/MyDrive/FOREX TRADING"

MODEL_FOLDER = os.path.join(BASE_PATH, "ALL_MODELS")

CSV_FOLDER = os.path.join(BASE_PATH, "CSV_FILES")

IMPORTANCE_FOLDER = os.path.join(BASE_PATH, "SHAP RESULT")


# ===============================================================================================
# MODEL PATH
# ===============================================================================================

MODEL_SYMBOL = normalize_symbol(symbol=SYMBOL)

MODEL_PATH = os.path.join(
    MODEL_FOLDER, f"HL_MAIN_{TARGET_NAME}_{CD_TIME}_{MODEL_SYMBOL}_model.pkl"
)


print("=" * 100)
print("MULTI-CLASS FORWARD TEST")
print("=" * 100)

print(f"SYMBOL             : {SYMBOL}")
print(f"TARGET             : {TARGET_NAME}")
print(f"CD TIME            : {CD_TIME}")
print(f"FORWARD THRESHOLD  : {FORWARD_THRESHOLD}")
print(f"TP / SL            : 1:{TP_SL_RATIO}")
print("=" * 100)


# ===============================================================================================
# LOAD MODEL BUNDLE
# ===============================================================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\n[[BAD]] Model not found:\n{MODEL_PATH}")


MODEL_BUNDLE = joblib.load(MODEL_PATH)

print(f"\n[[GOOD]] Loaded model:\n{MODEL_PATH}")

# ===============================================================================================
# MODEL SAFETY CHECKS
# ===============================================================================================

if "model" not in MODEL_BUNDLE:
    raise KeyError("Model bundle does not contain 'model'.")


if "features" not in MODEL_BUNDLE:

    raise KeyError("Model bundle does not contain 'features'.")


if "target" not in MODEL_BUNDLE:

    raise KeyError("Model bundle does not contain 'target'.")


if MODEL_BUNDLE["target"] != TARGET_NAME:

    raise ValueError(
        f"\n[[BAD]] TARGET MISMATCH\n"
        f"Expected: {TARGET_NAME}\n"
        f"Loaded:   {MODEL_BUNDLE['target']}"
    )


MODEL = MODEL_BUNDLE["model"]

MODEL_FEATURES = MODEL_BUNDLE["features"]

TARGET_SETTINGS = MODEL_BUNDLE.get("target_settings", {})


print(f"[[GOOD]] Model target confirmed: " f"{MODEL_BUNDLE['target']}")

print(f"[[GOOD]] Number of model features: " f"{len(MODEL_FEATURES)}")


# ===============================================================================================
# LOAD CLASS-SPECIFIC MEDIAN MOVEMENT PERCENTAGES
# ===============================================================================================

TARGET_SETTINGS = MODEL_BUNDLE["target_settings"]

CLASS_SETTINGS = TARGET_SETTINGS.get("classes")


if not isinstance(CLASS_SETTINGS, dict):

    raise KeyError(
        "\n[[BAD]] Could not find class-specific settings.\n"
        "Expected:\n"
        "    target_settings['classes']\n"
    )


MEDIAN_MOVEMENT_PCT = {}


for class_id in range(NUM_CLASSES):

    # -------------------------------------------------------------------------------------------
    # Handle either integer or string dictionary keys
    # -------------------------------------------------------------------------------------------

    class_data = CLASS_SETTINGS.get(class_id)

    if class_data is None:

        class_data = CLASS_SETTINGS.get(str(class_id))

    if not isinstance(class_data, dict):

        raise KeyError(f"\n[[BAD]] Missing settings for class {class_id}.\n")

    median_pct = class_data.get("median_movement_pct")

    if median_pct is None or not np.isfinite(median_pct):

        raise KeyError(
            f"\n[[BAD]] Missing median_movement_pct " f"for class {class_id}.\n"
        )

    MEDIAN_MOVEMENT_PCT[class_id] = float(median_pct)


print("\n[[GOOD]] Class-specific median movement percentages:")


for class_id in range(NUM_CLASSES):

    print(
        f"    Class {class_id} "
        f"({CLASS_NAMES[class_id]:<18}) : "
        f"{MEDIAN_MOVEMENT_PCT[class_id]:.6f}%"
    )


# ===============================================================================================
# CONVERT CLASS MEDIANS TO INTEGER-KEY DICTIONARY
# ===============================================================================================

CLASS_MEDIAN_MOVEMENT = {}


for class_id in range(NUM_CLASSES):

    if class_id in MEDIAN_MOVEMENT_PCT:

        value = MEDIAN_MOVEMENT_PCT[class_id]

    elif str(class_id) in MEDIAN_MOVEMENT_PCT:
        value = MEDIAN_MOVEMENT_PCT[str(class_id)]

    else:
        raise KeyError(
            f"\n[[BAD]] Median movement percentage for class {class_id} was not found."
        )

    CLASS_MEDIAN_MOVEMENT[class_id] = float(value)


# ===============================================================================================
# DISPLAY CLASS MOVEMENT CONFIGURATION
# ===============================================================================================

print("\nCLASS MEDIAN MOVEMENT PERCENTAGES")

for class_id in range(NUM_CLASSES):
    print(
        f"Class {class_id} ({CLASS_NAMES[class_id]}) : {CLASS_MEDIAN_MOVEMENT[class_id]:.6f}%"
    )


# ===============================================================================================
# LOAD SYMBOL INFORMATION
# ===============================================================================================

SYMBOL_INFO_FILE = os.path.join(CSV_FOLDER, "SYMBOL_INFO.json")


with open(SYMBOL_INFO_FILE, "r", encoding="utf-8") as f:

    SYMBOL_INFO = json.load(f)


if SYMBOL not in SYMBOL_INFO:

    raise ValueError(f"Symbol information not found for {SYMBOL}")


info = SYMBOL_INFO[SYMBOL]


balance = info["balance"]

equity = info["equity"]

currency = info["currency"]

leverage = info["leverage"]

pip_size = info["pip_size"]

pip_value_per_lot = info["pip_value_per_lot"]

spread = info["spread"]

min_gap = info["min_gap"]

vol_info = info["vol_info"]


min_lot = vol_info["min"]

max_lot = vol_info["max"]

lot_step = vol_info["step"]


# ===============================================================================================
# LOAD POINT INFORMATION
# ===============================================================================================

POINTS_FILE = os.path.join(CSV_FOLDER, "symbol_points.json")


MT5_SYMBOL = SYMBOL + "c"


print(f"\nCURRENTLY RUNNING SYMBOL: {SYMBOL}")

print(f"MT5 SYMBOL: {MT5_SYMBOL}")


with open(POINTS_FILE, "r", encoding="utf-8") as file:

    SYMBOL_POINTS = json.load(file)


if MT5_SYMBOL not in SYMBOL_POINTS:

    raise ValueError(
        f"Point information not found for " f"{MT5_SYMBOL} in {POINTS_FILE}"
    )


POINT = SYMBOL_POINTS[MT5_SYMBOL]["point"]


print(f"POINT: {POINT}")


# ===============================================================================================
# LOAD DATASET
# ===============================================================================================

DATA_FILE = os.path.join(
    CSV_FOLDER, f"MT5_{CD_TIME}_BT_{SYMBOL}_Exchange_Rate_Dataset.csv"
)


print("\n[[INFO]] Loading dataset...")


data = pd.read_csv(DATA_FILE)


print("[[GOOD]] DATASET LOADED SUCCESSFULLY")


# ===============================================================================================
# APPLY FEATURES
# ===============================================================================================

print("\nSTARTING APPLY FEATURES...")


df = apply_features(df=data, point=POINT)


# ===============================================================================================
# OPTIONAL FORWARD-TEST DATE RANGE
#
# IMPORTANT:
#
# Features are calculated BEFORE filtering dates.
#
# This preserves historical rolling-feature context.
# ===============================================================================================

# -----------------------------------------------------------------------
# Example:
#
# df = df[
#     (df["Date"] >= "2026-01-01")
#     &
#     (df["Date"] < "2026-10-01")
# ].copy()
#
# -----------------------------------------------------------------------


# ===============================================================================================
# CLEAN DATA
# ===============================================================================================

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)


print("\nDF INFO:", df.shape)

print("\t APPLY FEATURES SUCCESSFULLY")


# ===============================================================================================
# ENSURE DATE IS DATETIME
#
# This is important because history_df will be used for:
#
#   - hour analysis
#   - day analysis
#   - month analysis
#   - duration analysis
#   - session analysis
# ===============================================================================================

df["Date"] = pd.to_datetime(df["Date"])


# ===============================================================================================
# MODEL INPUT SAFETY CHECK
# ===============================================================================================

missing_features = [feature for feature in MODEL_FEATURES if feature not in df.columns]


if len(missing_features) > 0:

    raise ValueError(
        "\n[[BAD]] Missing model features:\n" + "\n".join(missing_features)
    )


X = df[MODEL_FEATURES].copy()


# ===============================================================================================
# PREDICT 5 CLASSES
# ===============================================================================================

print("\n[[INFO]] Generating 5-class predictions...")


probabilities = MODEL.predict_proba(X)


# ===============================================================================================
# SAFETY CHECK
# ===============================================================================================

if probabilities.shape[1] != NUM_CLASSES:

    raise ValueError(
        f"\n[[BAD]] Model returned "
        f"{probabilities.shape[1]} classes, "
        f"but {NUM_CLASSES} classes were expected."
    )


# ===============================================================================================
# WORKING DATAFRAME
# ===============================================================================================

working_df = df[["Date", "High", "Low", "Open", "Close", "ATR"]].copy()


# ===============================================================================================
# STORE CLASS PROBABILITIES
# ===============================================================================================

for class_id in range(NUM_CLASSES):

    working_df[f"CLASS_{class_id}_PROB"] = probabilities[:, class_id]


# ===============================================================================================
# PREDICTED CLASS
# ===============================================================================================

working_df["PREDICTED_CLASS"] = np.argmax(probabilities, axis=1)


working_df["PREDICTED_PROBABILITY"] = np.max(probabilities, axis=1)


# ===============================================================================================
# CLASS MOVEMENT
# ===============================================================================================

working_df["MOVEMENT_PCT"] = working_df["PREDICTED_CLASS"].map(CLASS_MEDIAN_MOVEMENT)


# ===============================================================================================
# DIRECTION
#
#   0/1 -> SELL -> -1
#   2   -> NO TRADE -> 0
#   3/4 -> BUY -> +1
# ===============================================================================================

working_df["DIRECTION"] = 0


working_df.loc[working_df["PREDICTED_CLASS"].isin(list(DOWN_CLASSES)), "DIRECTION"] = -1


working_df.loc[working_df["PREDICTED_CLASS"].isin(list(UP_CLASSES)), "DIRECTION"] = 1


# ===============================================================================================
# ENTRY PRICE
#
# Signal at candle i.
#
# Entry at next candle Open.
#
# BUY:
#     Ask = Open + spread
#
# SELL:
#     Bid = Open
# ===============================================================================================

working_df["NEXT_ENTRY"] = working_df["Open"].shift(-1)


working_df["ASK_PRICE"] = working_df["NEXT_ENTRY"] + spread


working_df["BID_PRICE"] = working_df["NEXT_ENTRY"]


working_df["ENTRY_BUY"] = working_df["ASK_PRICE"]


working_df["ENTRY_SELL"] = working_df["BID_PRICE"]


# ===============================================================================================
# CLASS-SPECIFIC TP / SL DISTANCE
#
# 1:1 TP / SL
# ===============================================================================================

movement_fraction = working_df["MOVEMENT_PCT"] / 100.0


working_df["BUY_DISTANCE"] = working_df["ENTRY_BUY"] * movement_fraction


working_df["SELL_DISTANCE"] = working_df["ENTRY_SELL"] * movement_fraction


# ===============================================================================================
# BUY TP / SL
# ===============================================================================================

working_df["TP_BUY"] = (
    working_df["ENTRY_BUY"] + working_df["BUY_DISTANCE"] * TP_SL_RATIO
)


working_df["SL_BUY"] = working_df["ENTRY_BUY"] - working_df["BUY_DISTANCE"]


# ===============================================================================================
# SELL TP / SL
# ===============================================================================================

working_df["TP_SELL"] = (
    working_df["ENTRY_SELL"] - working_df["SELL_DISTANCE"] * TP_SL_RATIO
)


working_df["SL_SELL"] = working_df["ENTRY_SELL"] + working_df["SELL_DISTANCE"]


# ===============================================================================================
# PIP DISTANCES
# ===============================================================================================

working_df["BUY_DISTANCE_PIPS"] = working_df["BUY_DISTANCE"] / pip_size


working_df["SELL_DISTANCE_PIPS"] = working_df["SELL_DISTANCE"] / pip_size


# ===============================================================================================
# LOT SIZE FUNCTION
# ===============================================================================================


def calc_lot_size(balance, risk_percent, sl_pips, pip_value_per_lot, min_lot, max_lot):

    risk_amount = balance * (risk_percent / 100)

    if sl_pips <= 0:

        return min_lot

    lot_cal = risk_amount / (sl_pips * pip_value_per_lot)

    lot = max(min_lot, min(lot_cal, max_lot))

    return lot


# ===============================================================================================
# LOT NORMALIZATION
# ===============================================================================================


def normalize_lot(lot, vol_min, vol_max, vol_step):

    lot = max(vol_min, min(lot, vol_max))

    lot = np.floor(lot / vol_step) * vol_step

    return round(lot, 2)


# ===============================================================================================
# FORWARD TEST ONE THRESHOLD
# ===============================================================================================


def run_forward_test(working_df, threshold):

    # ===========================================================================================
    # SIGNAL GENERATION
    # ===========================================================================================

    predicted_class = working_df["PREDICTED_CLASS"].to_numpy()

    predicted_probability = working_df["PREDICTED_PROBABILITY"].to_numpy()

    movement_pct = working_df["MOVEMENT_PCT"].to_numpy()

    # ===========================================================================================
    # DATES
    #
    # We keep the complete timestamp.
    #
    # IMPORTANT:
    #
    # i     = signal candle
    # i + 1 = actual entry candle
    # j     = exit candle
    #
    # ===========================================================================================

    dates = pd.to_datetime(working_df["Date"]).to_numpy()

    # ===========================================================================================
    # SIGNAL ARRAY
    # ===========================================================================================

    signals = np.zeros(len(working_df), dtype=np.int8)

    # ===========================================================================================
    # CONFIDENCE FILTER
    # ===========================================================================================

    confident = predicted_probability >= threshold

    # ===========================================================================================
    # SELL
    # ===========================================================================================

    sell_mask = confident & np.isin(predicted_class, list(DOWN_CLASSES))

    signals[sell_mask] = -1

    # ===========================================================================================
    # BUY
    # ===========================================================================================

    buy_mask = confident & np.isin(predicted_class, list(UP_CLASSES))

    signals[buy_mask] = 1

    # ===========================================================================================
    # MARKET ARRAYS
    # ===========================================================================================

    highs = working_df["High"].to_numpy()

    lows = working_df["Low"].to_numpy()

    tp_buy = working_df["TP_BUY"].to_numpy()

    sl_buy = working_df["SL_BUY"].to_numpy()

    tp_sell = working_df["TP_SELL"].to_numpy()

    sl_sell = working_df["SL_SELL"].to_numpy()

    entry_buy = working_df["ENTRY_BUY"].to_numpy()

    entry_sell = working_df["ENTRY_SELL"].to_numpy()

    buy_distance_pips = working_df["BUY_DISTANCE_PIPS"].to_numpy()

    sell_distance_pips = working_df["SELL_DISTANCE_PIPS"].to_numpy()

    # ===========================================================================================
    # ACCOUNT VARIABLES
    # ===========================================================================================

    threshold_balance = balance

    starting_balance = balance

    wins = 0

    losses = 0

    total_gross_profit = 0.0

    total_gross_loss = 0.0

    equity_curve = []

    trade_returns = []

    # ===========================================================================================
    # CLASS STATISTICS
    # ===========================================================================================

    trade_classes = []

    trade_movements = []

    trade_probabilities = []

    # ===========================================================================================
    # TRADE DURATION
    # ===========================================================================================

    trade_durations = []

    # ===========================================================================================
    # COMPLETE TRADE HISTORY
    #
    # One dictionary = one completed trade.
    #
    # This is intentionally separate from the summary result.
    # ===========================================================================================

    history = []

    # ===========================================================================================
    # FORWARD TEST LOOP
    # ===========================================================================================

    i = 0

    n = len(signals)

    while i < n - 1:

        signal = signals[i]

        # ---------------------------------------------------------------------------------------
        # NO TRADE
        # ---------------------------------------------------------------------------------------

        if signal == 0:

            i += 1

            continue

        # =======================================================================================
        # LOT SIZE
        # =======================================================================================

        if signal == 1:

            sl_pips = buy_distance_pips[i]

        else:

            sl_pips = sell_distance_pips[i]

        if not np.isfinite(sl_pips) or sl_pips <= 0:

            i += 1

            continue

        lot_size = calc_lot_size(
            balance=threshold_balance,
            risk_percent=RISK_PERCENT,
            sl_pips=sl_pips,
            pip_value_per_lot=pip_value_per_lot,
            min_lot=min_lot,
            max_lot=max_lot,
        )

        lot_size = normalize_lot(
            lot=lot_size,
            vol_min=vol_info["min"],
            vol_max=vol_info["max"],
            vol_step=vol_info["step"],
        )

        balance_before_trade = threshold_balance

        predicted_class_i = int(predicted_class[i])

        predicted_probability_i = float(predicted_probability[i])

        trade_movement = float(movement_pct[i])

        # =======================================================================================
        # IMPORTANT TIMESTAMPS
        # =======================================================================================

        signal_date = dates[i]

        # Actual entry happens on next candle.
        #
        # The outer loop guarantees i < n - 1.
        entry_index = i + 1

        entry_date = dates[entry_index]

        # =======================================================================================
        # BUY
        # =======================================================================================

        if signal == 1:

            sl = sl_buy[i]

            tp = tp_buy[i]

            entry = entry_buy[i]

            j = i + 1

            trade_closed = False

            while j < n:

                # ===============================================================================
                # STOP LOSS
                #
                # IMPORTANT:
                #
                # SL is checked BEFORE TP.
                #
                # Therefore if both TP and SL are touched on the
                # same candle, this conservative backtest counts LOSS.
                # ===============================================================================

                if lows[j] <= sl:

                    loss_amount = (
                        ((entry - sl) / pip_size) * pip_value_per_lot * lot_size
                    )

                    threshold_balance -= loss_amount

                    total_gross_loss += loss_amount

                    losses += 1

                    trade_return = (
                        threshold_balance - balance_before_trade
                    ) / balance_before_trade

                    trade_returns.append(trade_return)

                    equity_curve.append(threshold_balance)

                    trade_classes.append(predicted_class_i)

                    trade_movements.append(trade_movement)

                    trade_probabilities.append(predicted_probability_i)

                    duration = j - i

                    trade_durations.append(duration)

                    # ===========================================================================
                    # SAVE COMPLETE TRADE HISTORY
                    # ===========================================================================

                    history.append(
                        {
                            "SIGNAL_DATE": signal_date,
                            "ENTRY_DATE": entry_date,
                            "EXIT_DATE": dates[j],
                            "SIGNAL_HOUR": pd.Timestamp(signal_date).hour,
                            "ENTRY_HOUR": pd.Timestamp(entry_date).hour,
                            "EXIT_HOUR": pd.Timestamp(dates[j]).hour,
                            "DIRECTION": "BUY",
                            "CLASS": predicted_class_i,
                            "CLASS_NAME": CLASS_NAMES[predicted_class_i],
                            "PROBABILITY": predicted_probability_i,
                            "MOVEMENT_PCT": trade_movement,
                            "ENTRY": float(entry),
                            "TP": float(tp),
                            "SL": float(sl),
                            "LOT_SIZE": float(lot_size),
                            "RESULT": "LOSS",
                            "P_AND_L": float(-loss_amount),
                            "BALANCE_BEFORE": float(balance_before_trade),
                            "BALANCE_AFTER": float(threshold_balance),
                            "DURATION_CANDLES": int(duration),
                            "DURATION_HOURS": float(duration * 10 / 60),
                        }
                    )

                    # ===========================================================================
                    # NEXT OPPORTUNITY
                    #
                    # Trade closed on candle j.
                    #
                    # Next opportunity begins on j + 1.
                    # ===========================================================================

                    i = j + 1

                    trade_closed = True

                    break

                # ===============================================================================
                # TAKE PROFIT
                # ===============================================================================

                if highs[j] >= tp:

                    profit_amount = (
                        ((tp - entry) / pip_size) * pip_value_per_lot * lot_size
                    )

                    threshold_balance += profit_amount

                    total_gross_profit += profit_amount

                    wins += 1

                    trade_return = (
                        threshold_balance - balance_before_trade
                    ) / balance_before_trade

                    trade_returns.append(trade_return)

                    equity_curve.append(threshold_balance)

                    trade_classes.append(predicted_class_i)

                    trade_movements.append(trade_movement)

                    trade_probabilities.append(predicted_probability_i)

                    duration = j - i

                    trade_durations.append(duration)

                    # ===========================================================================
                    # SAVE COMPLETE TRADE HISTORY
                    # ===========================================================================

                    history.append(
                        {
                            "SIGNAL_DATE": signal_date,
                            "ENTRY_DATE": entry_date,
                            "EXIT_DATE": dates[j],
                            "SIGNAL_HOUR": pd.Timestamp(signal_date).hour,
                            "ENTRY_HOUR": pd.Timestamp(entry_date).hour,
                            "EXIT_HOUR": pd.Timestamp(dates[j]).hour,
                            "DIRECTION": "BUY",
                            "CLASS": predicted_class_i,
                            "CLASS_NAME": CLASS_NAMES[predicted_class_i],
                            "PROBABILITY": predicted_probability_i,
                            "MOVEMENT_PCT": trade_movement,
                            "ENTRY": float(entry),
                            "TP": float(tp),
                            "SL": float(sl),
                            "LOT_SIZE": float(lot_size),
                            "RESULT": "WIN",
                            "P_AND_L": float(profit_amount),
                            "BALANCE_BEFORE": float(balance_before_trade),
                            "BALANCE_AFTER": float(threshold_balance),
                            "DURATION_CANDLES": int(duration),
                            "DURATION_HOURS": float(duration * 10 / 60),
                        }
                    )

                    # ===========================================================================
                    # NEXT OPPORTUNITY
                    # ===========================================================================

                    i = j + 1

                    trade_closed = True

                    break

                # ===============================================================================
                # HOLD
                # ===============================================================================

                j += 1

            # -----------------------------------------------------------------------------------
            # DATASET ENDED BEFORE TP / SL
            # -----------------------------------------------------------------------------------

            if not trade_closed:

                i = n

                break

        # =======================================================================================
        # SELL
        # =======================================================================================

        else:

            sl = sl_sell[i]

            tp = tp_sell[i]

            entry = entry_sell[i]

            j = i + 1

            trade_closed = False

            while j < n:

                # ===============================================================================
                # STOP LOSS
                #
                # Conservative same-candle behavior:
                # SL checked before TP.
                # ===============================================================================

                if highs[j] >= sl:

                    loss_amount = (
                        ((sl - entry) / pip_size) * pip_value_per_lot * lot_size
                    )

                    threshold_balance -= loss_amount

                    total_gross_loss += loss_amount

                    losses += 1

                    trade_return = (
                        threshold_balance - balance_before_trade
                    ) / balance_before_trade

                    trade_returns.append(trade_return)

                    equity_curve.append(threshold_balance)

                    trade_classes.append(predicted_class_i)

                    trade_movements.append(trade_movement)

                    trade_probabilities.append(predicted_probability_i)

                    duration = j - i

                    trade_durations.append(duration)

                    # ===========================================================================
                    # SAVE COMPLETE TRADE HISTORY
                    # ===========================================================================

                    history.append(
                        {
                            "SIGNAL_DATE": signal_date,
                            "ENTRY_DATE": entry_date,
                            "EXIT_DATE": dates[j],
                            "SIGNAL_HOUR": pd.Timestamp(signal_date).hour,
                            "ENTRY_HOUR": pd.Timestamp(entry_date).hour,
                            "EXIT_HOUR": pd.Timestamp(dates[j]).hour,
                            "DIRECTION": "SELL",
                            "CLASS": predicted_class_i,
                            "CLASS_NAME": CLASS_NAMES[predicted_class_i],
                            "PROBABILITY": predicted_probability_i,
                            "MOVEMENT_PCT": trade_movement,
                            "ENTRY": float(entry),
                            "TP": float(tp),
                            "SL": float(sl),
                            "LOT_SIZE": float(lot_size),
                            "RESULT": "LOSS",
                            "P_AND_L": float(-loss_amount),
                            "BALANCE_BEFORE": float(balance_before_trade),
                            "BALANCE_AFTER": float(threshold_balance),
                            "DURATION_CANDLES": int(duration),
                            "DURATION_HOURS": float(duration * 10 / 60),
                        }
                    )

                    # ===========================================================================
                    # NEXT OPPORTUNITY
                    # ===========================================================================

                    i = j + 1

                    trade_closed = True

                    break

                # ===============================================================================
                # TAKE PROFIT
                # ===============================================================================

                if lows[j] <= tp:

                    profit_amount = (
                        ((entry - tp) / pip_size) * pip_value_per_lot * lot_size
                    )

                    threshold_balance += profit_amount

                    total_gross_profit += profit_amount

                    wins += 1

                    trade_return = (
                        threshold_balance - balance_before_trade
                    ) / balance_before_trade

                    trade_returns.append(trade_return)

                    equity_curve.append(threshold_balance)

                    trade_classes.append(predicted_class_i)

                    trade_movements.append(trade_movement)

                    trade_probabilities.append(predicted_probability_i)

                    duration = j - i

                    trade_durations.append(duration)

                    # ===========================================================================
                    # SAVE COMPLETE TRADE HISTORY
                    # ===========================================================================

                    history.append(
                        {
                            "SIGNAL_DATE": signal_date,
                            "ENTRY_DATE": entry_date,
                            "EXIT_DATE": dates[j],
                            "SIGNAL_HOUR": pd.Timestamp(signal_date).hour,
                            "ENTRY_HOUR": pd.Timestamp(entry_date).hour,
                            "EXIT_HOUR": pd.Timestamp(dates[j]).hour,
                            "DIRECTION": "SELL",
                            "CLASS": predicted_class_i,
                            "CLASS_NAME": CLASS_NAMES[predicted_class_i],
                            "PROBABILITY": predicted_probability_i,
                            "MOVEMENT_PCT": trade_movement,
                            "ENTRY": float(entry),
                            "TP": float(tp),
                            "SL": float(sl),
                            "LOT_SIZE": float(lot_size),
                            "RESULT": "WIN",
                            "P_AND_L": float(profit_amount),
                            "BALANCE_BEFORE": float(balance_before_trade),
                            "BALANCE_AFTER": float(threshold_balance),
                            "DURATION_CANDLES": int(duration),
                            "DURATION_HOURS": float(duration * 10 / 60),
                        }
                    )

                    # ===========================================================================
                    # NEXT OPPORTUNITY
                    # ===========================================================================

                    i = j + 1

                    trade_closed = True

                    break

                # ===============================================================================
                # HOLD
                # ===============================================================================

                j += 1

            # -----------------------------------------------------------------------------------
            # DATASET ENDED BEFORE TP / SL
            # -----------------------------------------------------------------------------------

            if not trade_closed:

                i = n

                break

    # ===========================================================================================
    # TOTAL TRADES
    # ===========================================================================================

    total_trades = wins + losses

    # ===========================================================================================
    # WIN RATE
    # ===========================================================================================

    if total_trades > 0:

        win_rate = wins / total_trades

    else:

        win_rate = 0.0

    # ===========================================================================================
    # PROFIT FACTOR
    # ===========================================================================================

    if total_gross_loss > 0:

        profit_factor = total_gross_profit / total_gross_loss

    else:

        profit_factor = float("inf")

    # ===========================================================================================
    # EXPECTANCY
    # ===========================================================================================

    if wins > 0:

        avg_win = total_gross_profit / wins

    else:

        avg_win = 0.0

    if losses > 0:

        avg_loss = total_gross_loss / losses

    else:

        avg_loss = 0.0

    win_probability = wins / total_trades if total_trades > 0 else 0

    loss_probability = losses / total_trades if total_trades > 0 else 0

    expectancy = (win_probability * avg_win) - (loss_probability * avg_loss)

    # ===========================================================================================
    # MAXIMUM DRAWDOWN
    # ===========================================================================================

    if len(equity_curve) > 0:

        equity_series = pd.Series(equity_curve)

        running_peak = equity_series.cummax()

        drawdowns = (running_peak - equity_series) / running_peak

        max_drawdown = drawdowns.max()

    else:

        max_drawdown = 0.0

    # ===========================================================================================
    # TRADE SHARPE
    # ===========================================================================================

    if len(trade_returns) > 1:

        trade_returns_array = np.array(trade_returns)

        returns_std = trade_returns_array.std(ddof=1)

        if returns_std > 0:

            trade_sharpe = (trade_returns_array.mean() / returns_std) * np.sqrt(252)

        else:

            trade_sharpe = 0.0

    else:

        trade_sharpe = 0.0

    # ===========================================================================================
    # CLASS STATISTICS
    # ===========================================================================================

    class_trade_counts = {class_id: 0 for class_id in range(NUM_CLASSES)}

    class_wins = {class_id: 0 for class_id in range(NUM_CLASSES)}

    for class_id, trade_return in zip(trade_classes, trade_returns):

        class_trade_counts[class_id] += 1

        if trade_return > 0:

            class_wins[class_id] += 1

    class_win_rates = {}

    for class_id in range(NUM_CLASSES):

        count = class_trade_counts[class_id]

        if count > 0:

            class_win_rates[class_id] = class_wins[class_id] / count

        else:

            class_win_rates[class_id] = 0.0

    # ===========================================================================================
    # TRADE DURATION STATISTICS
    # ===========================================================================================

    if len(trade_durations) > 0:

        duration_array = np.array(trade_durations)

        avg_duration_candles = duration_array.mean()

        median_duration_candles = np.median(duration_array)

        max_duration_candles = duration_array.max()

        trades_over_18_candles = (duration_array > 18).sum()

        trades_at_or_under_18_candles = (duration_array <= 18).sum()

        trades_over_36_candles = (duration_array > 36).sum()

        trades_over_54_candles = (duration_array > 54).sum()

        trades_over_72_candles = (duration_array > 72).sum()

    else:

        avg_duration_candles = 0.0

        median_duration_candles = 0.0

        max_duration_candles = 0

        trades_over_18_candles = 0

        trades_at_or_under_18_candles = 0

        trades_over_36_candles = 0

        trades_over_54_candles = 0

        trades_over_72_candles = 0

    # ===========================================================================================
    # CREATE HISTORY DATAFRAME
    #
    # IMPORTANT:
    #
    # This is intentionally NOT part of result_df.
    #
    # result_df = one row / summary
    #
    # history_df = one row / completed trade
    # ===========================================================================================

    history_df = pd.DataFrame(history)

    # ===========================================================================================
    # HISTORY DATA TYPES
    # ===========================================================================================

    if not history_df.empty:

        history_df["SIGNAL_DATE"] = pd.to_datetime(history_df["SIGNAL_DATE"])

        history_df["ENTRY_DATE"] = pd.to_datetime(history_df["ENTRY_DATE"])

        history_df["EXIT_DATE"] = pd.to_datetime(history_df["EXIT_DATE"])

        # ---------------------------------------------------------------------------------------
        # Additional analysis columns
        #
        # These do NOT affect the backtest.
        # They only make history_df easier to analyze.
        # ---------------------------------------------------------------------------------------

        history_df["SIGNAL_DAY"] = history_df["SIGNAL_DATE"].dt.day_name()

        history_df["ENTRY_DAY"] = history_df["ENTRY_DATE"].dt.day_name()

        history_df["EXIT_DAY"] = history_df["EXIT_DATE"].dt.day_name()

        history_df["SIGNAL_MONTH"] = (
            history_df["SIGNAL_DATE"].dt.to_period("M").astype(str)
        )

        history_df["ENTRY_MONTH"] = (
            history_df["ENTRY_DATE"].dt.to_period("M").astype(str)
        )

    # ===========================================================================================
    # FINAL RESULT
    # ===========================================================================================

    result = {
        "THRESHOLD": round(threshold, 3),
        "TRADES": total_trades,
        "WINS": wins,
        "LOSSES": losses,
        "WIN_RATE": round(win_rate * 100, 2),
        "PROFIT_FACTOR": (
            round(profit_factor, 2) if np.isfinite(profit_factor) else np.inf
        ),
        "EXPECTANCY": round(expectancy, 2),
        "MAX_DRAWDOWN": round(max_drawdown * 100, 2),
        "TRADE_SHARPE": round(trade_sharpe, 2),
        "STARTING_BALANCE": round(starting_balance, 2),
        "FINAL_BALANCE": round(threshold_balance, 2),
        "GROSS_PROFIT": round(total_gross_profit, 2),
        "GROSS_LOSS": round(total_gross_loss, 2),
        # ---------------------------------------------------------------------------------------
        # CLASS 0
        # ---------------------------------------------------------------------------------------
        "CLASS_0_TRADES": class_trade_counts[0],
        "CLASS_0_WIN_RATE": round(class_win_rates[0] * 100, 2),
        # ---------------------------------------------------------------------------------------
        # CLASS 1
        # ---------------------------------------------------------------------------------------
        "CLASS_1_TRADES": class_trade_counts[1],
        "CLASS_1_WIN_RATE": round(class_win_rates[1] * 100, 2),
        # ---------------------------------------------------------------------------------------
        # CLASS 2
        # ---------------------------------------------------------------------------------------
        "CLASS_2_TRADES": class_trade_counts[2],
        "CLASS_2_WIN_RATE": round(class_win_rates[2] * 100, 2),
        # ---------------------------------------------------------------------------------------
        # CLASS 3
        # ---------------------------------------------------------------------------------------
        "CLASS_3_TRADES": class_trade_counts[3],
        "CLASS_3_WIN_RATE": round(class_win_rates[3] * 100, 2),
        # ---------------------------------------------------------------------------------------
        # CLASS 4
        # ---------------------------------------------------------------------------------------
        "CLASS_4_TRADES": class_trade_counts[4],
        "CLASS_4_WIN_RATE": round(class_win_rates[4] * 100, 2),
        # ---------------------------------------------------------------------------------------
        # DURATION
        # ---------------------------------------------------------------------------------------
        "AVG_DURATION_CANDLES": round(avg_duration_candles, 2),
        "AVG_DURATION_HOURS": round(avg_duration_candles * 10 / 60, 2),
        "MEDIAN_DURATION_CANDLES": round(median_duration_candles, 2),
        "MEDIAN_DURATION_HOURS": round(median_duration_candles * 10 / 60, 2),
        "MAX_DURATION_CANDLES": int(max_duration_candles),
        "MAX_DURATION_HOURS": round(max_duration_candles * 10 / 60, 2),
        "TRADES_OVER_18_CANDLES": int(trades_over_18_candles),
        "TRADES_AT_OR_UNDER_18_CANDLES": int(trades_at_or_under_18_candles),
        "TRADES_OVER_36_CANDLES": int(trades_over_36_candles),
        "TRADES_OVER_54_CANDLES": int(trades_over_54_candles),
        "TRADES_OVER_72_CANDLES": int(trades_over_72_candles),
    }

    # ===========================================================================================
    # RETURN BOTH:
    #
    #   result       -> summary
    #   history_df   -> individual trade history
    # ===========================================================================================

    return result, history_df


# ===============================================================================================
# RUN ONE FORWARD TEST
# ===============================================================================================

print("\n")
print("=" * 100)
print("STARTING SINGLE-THRESHOLD FORWARD TEST")
print("=" * 100)

print(f"FORWARD THRESHOLD: {FORWARD_THRESHOLD}")


result, history_df = run_forward_test(
    working_df=working_df, threshold=FORWARD_THRESHOLD
)


# ===============================================================================================
# RESULT DATAFRAME
#
# ONE ROW = COMPLETE FORWARD TEST SUMMARY
# ===============================================================================================

result_df = pd.DataFrame([result])


# ===============================================================================================
# DISPLAY MAIN RESULT
# ===============================================================================================

print("\n")
print("=" * 100)
print("FORWARD TEST RESULT")
print("=" * 100)

print(result_df.to_string(index=False))


# ===============================================================================================
# CLASS BREAKDOWN
# ===============================================================================================

print("\n")
print("=" * 100)
print("CLASS BREAKDOWN")
print("=" * 100)


for class_id in range(NUM_CLASSES):

    trades = result[f"CLASS_{class_id}_TRADES"]

    win_rate = result[f"CLASS_{class_id}_WIN_RATE"]

    print(
        f"CLASS {class_id} "
        f"({CLASS_NAMES[class_id]:<18}) | "
        f"TRADES: {trades:<5} | "
        f"WIN RATE: {win_rate:.2f}%"
    )


# ===============================================================================================
# DURATION BREAKDOWN
# ===============================================================================================

print("\n")
print("=" * 100)
print("TRADE DURATION BREAKDOWN")
print("=" * 100)

print(
    f"Average Duration       : "
    f"{result['AVG_DURATION_CANDLES']} candles "
    f"({result['AVG_DURATION_HOURS']} hours)"
)

print(
    f"Median Duration        : "
    f"{result['MEDIAN_DURATION_CANDLES']} candles "
    f"({result['MEDIAN_DURATION_HOURS']} hours)"
)

print(
    f"Maximum Duration       : "
    f"{result['MAX_DURATION_CANDLES']} candles "
    f"({result['MAX_DURATION_HOURS']} hours)"
)

print(f"Trades <= 18 candles   : " f"{result['TRADES_AT_OR_UNDER_18_CANDLES']}")

print(f"Trades > 18 candles    : " f"{result['TRADES_OVER_18_CANDLES']}")

print(f"Trades > 36 candles    : " f"{result['TRADES_OVER_36_CANDLES']}")

print(f"Trades > 54 candles    : " f"{result['TRADES_OVER_54_CANDLES']}")

print(f"Trades > 72 candles    : " f"{result['TRADES_OVER_72_CANDLES']}")


# ===============================================================================================
# COMPLETE TRADE HISTORY
#
# IMPORTANT:
#
# This is a DIFFERENT dataframe from result_df.
#
# result_df:
#     One row containing the overall test performance.
#
# history_df:
#     One row for EVERY completed trade.
# ===============================================================================================

print("\n")
print("=" * 100)
print("COMPLETE TRADE HISTORY")
print("=" * 100)


if not history_df.empty:

    print(history_df.to_string(index=False))

else:

    print("[[INFO]] No completed trades.")


# ===============================================================================================
# TRADE HISTORY SUMMARY
# ===============================================================================================

print("\n")
print("=" * 100)
print("TRADE HISTORY SUMMARY")
print("=" * 100)


print(f"COMPLETED TRADES IN HISTORY_DF : " f"{len(history_df)}")


if not history_df.empty:

    print(f"FIRST SIGNAL TIME              : " f"{history_df['SIGNAL_DATE'].min()}")

    print(f"LAST SIGNAL TIME               : " f"{history_df['SIGNAL_DATE'].max()}")

    print(f"FIRST ENTRY TIME               : " f"{history_df['ENTRY_DATE'].min()}")

    print(f"LAST EXIT TIME                 : " f"{history_df['EXIT_DATE'].max()}")


# ===============================================================================================
# OPTIONAL: VERIFY RESULT TRADES == HISTORY TRADES
# ===============================================================================================

if len(history_df) != result["TRADES"]:

    print("\n")
    print("[[WARNING]] HISTORY / RESULT TRADE COUNT MISMATCH")

    print(f"result_df TRADES : {result['TRADES']}")

    print(f"history_df rows  : {len(history_df)}")

else:

    print("\n")
    print("[[GOOD]] HISTORY_DF TRADE COUNT MATCHES RESULT_DF")


# ===============================================================================================
# FORWARD TEST INFORMATION
# ===============================================================================================

print("\n")
print("=" * 100)
print("FORWARD TEST INFORMATION")
print("=" * 100)

print(f"SYMBOL              : {SYMBOL}")

print(f"MT5 SYMBOL          : {MT5_SYMBOL}")

print(f"TARGET              : {TARGET_NAME}")

print(f"TIMEFRAME           : {CD_TIME}")

print(f"THRESHOLD           : {FORWARD_THRESHOLD}")

print(f"RISK PER TRADE      : {RISK_PERCENT}%")

print(f"TP / SL             : 1:1")

print(f"CLASS 2             : NO TRADE")

print(f"FORWARD TEST START  : {df['Date'].min()}")

print(f"FORWARD TEST END    : {df['Date'].max()}")

print("=" * 100)
