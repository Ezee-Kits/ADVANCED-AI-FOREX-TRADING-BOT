##======================================================================
# FIND THRESHOLD FOR MAX EXCURSION [ 5 TARGET CLASSES [3] ]
##======================================================================

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

TARGET_HORIZON = 18  # 18 x 10M = 3 hours

RISK_PERCENT = 1

TRADE_NO = 225  # minimum trades required for threshold

NUM_CLASSES = 5

# Class mapping
CLASS_NAMES = {
    0: "Strong DOWN",
    1: "Moderate DOWN",
    2: "Small / NEUTRAL",
    3: "Moderate UP",
    4: "Strong UP",
}

# Tradable classes
DOWN_CLASSES = {0, 1}
NEUTRAL_CLASS = 2
UP_CLASSES = {3, 4}


# ===============================================================================================
# PATHS
# ===============================================================================================

BASE_PATH = "/content/drive/MyDrive/FOREX TRADING"
MODEL_FOLDER = os.path.join(BASE_PATH, "ALL_MODELS")
CSV_FOLDER = os.path.join(BASE_PATH, "CSV_FILES")

IMPORTANCE_FOLDER = os.path.join(BASE_PATH, "SHAP RESULT")


# ===============================================================================================
# LOAD MODEL
# ===============================================================================================


def load_model(cd_time, symbol):

    symbol_normalized = normalize_symbol(symbol=symbol)
    model_path = os.path.join(
        MODEL_FOLDER, f"HL_MAIN_{TARGET_NAME}_{cd_time}_{symbol_normalized}_model.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"\nModel not found:\n{model_path}")

    bundle = joblib.load(model_path)

    required_keys = ["model", "features", "target", "target_settings"]

    missing_keys = [key for key in required_keys if key not in bundle]

    if missing_keys:
        raise KeyError(f"\nModel is missing required keys: {missing_keys}")

    model = bundle["model"]

    features = bundle["features"]

    target = bundle["target"]

    target_settings = bundle["target_settings"]

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    if target != TARGET_NAME:

        raise ValueError(
            f"Loaded model target is '{target}', but expected '{TARGET_NAME}'."
        )

    if target_settings["target_horizon"] != TARGET_HORIZON:

        raise ValueError(
            f"Model target horizon is "
            f"{target_settings['target_horizon']}, "
            f"but threshold finder expects {TARGET_HORIZON}."
        )

    classes = target_settings["classes"]

    for class_id in range(NUM_CLASSES):

        if class_id not in classes:
            raise ValueError(f"Class {class_id} is missing from target_settings.")

        if "median_movement_pct" not in classes[class_id]:
            raise ValueError(f"median_movement_pct missing for class {class_id}.")

    print("\n[[ GOOD ]] Multiclass model loaded")
    print(f"MODEL : {model_path}")
    print(f"TARGET: {target}")
    print(f"FEATURES: {len(features)}")
    print(f"HORIZON: {TARGET_HORIZON} candles")

    return {
        "model": model,
        "features": features,
        "target": target,
        "target_settings": target_settings,
    }


# ===============================================================================================
# LOAD MODEL
# ===============================================================================================

MODEL_BUNDLE = load_model(cd_time=CD_TIME, symbol=SYMBOL)

MODEL = MODEL_BUNDLE["model"]

MODEL_FEATURES = MODEL_BUNDLE["features"]

TARGET_SETTINGS = MODEL_BUNDLE["target_settings"]


# ===============================================================================================
# DISPLAY TARGET SETTINGS
# ===============================================================================================

print("\n" + "=" * 80)
print("MODEL TARGET SETTINGS")
print("=" * 80)

print(f"Target Name       : {TARGET_SETTINGS['target_name']}")
print(f"Target Horizon    : {TARGET_SETTINGS['target_horizon']} candles")
print(f"Neutral Threshold : {TARGET_SETTINGS['neutral_threshold_pct']:.6f}%")
print(f"Strong Threshold  : {TARGET_SETTINGS['strong_threshold_pct']:.6f}%")

print("\nCLASS MOVEMENT SETTINGS")

for class_id in range(NUM_CLASSES):
    class_info = TARGET_SETTINGS["classes"][class_id]
    print(
        f"Class {class_id} {class_info['name']:<20} | Median Move: {class_info['median_movement_pct']:.6f}%"
    )


# ===============================================================================================
# SYMBOL INFORMATION
# ===============================================================================================

with open(os.path.join(CSV_FOLDER, "SYMBOL_INFO.json"), "r") as f:
    SYMBOL_INFO = json.load(f)


if SYMBOL not in SYMBOL_INFO:
    raise ValueError(f"Symbol information not found for {SYMBOL}.")


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
# SYMBOL POINT
# ===============================================================================================

POINTS_FILE = os.path.join(CSV_FOLDER, "symbol_points.json")

MT5_SYMBOL = SYMBOL + "c"

print("\n" + "=" * 80)
print(f"CURRENTLY RUNNING SYMBOL: {SYMBOL}")
print(f"MT5 SYMBOL: {MT5_SYMBOL}")


with open(POINTS_FILE, "r", encoding="utf-8") as file:
    SYMBOL_POINTS = json.load(file)


if MT5_SYMBOL not in SYMBOL_POINTS:
    raise ValueError(f"Point information not found for {MT5_SYMBOL} in {POINTS_FILE}")

POINT = SYMBOL_POINTS[MT5_SYMBOL]["point"]
print(f"POINT: {POINT}")


# ===============================================================================================
# LOAD DATA
# ===============================================================================================

DATA_FILE = os.path.join(
    CSV_FOLDER, f"MT5_{CD_TIME}_BT_{SYMBOL}_Exchange_Rate_Dataset.csv"
)
print("\nLOADING DATASET...")

data = pd.read_csv(DATA_FILE)

print("\tDATASET LOADED SUCCESSFULLY")
print("\tSTARTING APPLY FEATURES...")

# ===============================================================================================
# APPLY FEATURES
# ===============================================================================================

df = apply_features(df=data, point=POINT)


# ===============================================================================================
# OPTIONAL FORWARD TEST WINDOW
# ===============================================================================================

# df = df[(df["Date"] >= "2026-01-01") & (df["Date"] < "2026-10-01")].copy()


# ===============================================================================================
# CLEAN DATA
# ===============================================================================================

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nDF INFO:", df.shape)
print("\tAPPLY FEATURE SUCCESSFULLY")


# ===============================================================================================
# SAFETY CHECK MODEL FEATURES
# ===============================================================================================

missing_features = [feature for feature in MODEL_FEATURES if feature not in df.columns]

if missing_features:
    raise ValueError(
        "\nThe following model features are missing from the dataframe:\n{missing_features}"
    )


# ===============================================================================================
# MODEL INPUT
# ===============================================================================================

X = df[MODEL_FEATURES].copy()


# ===============================================================================================
# MULTICLASS PREDICTIONS
# ===============================================================================================

print("\n" + "=" * 80)

print("GENERATING MULTICLASS PROBABILITIES")

print("=" * 80)


probabilities = MODEL.predict_proba(X)


# ===============================================================================================
# SAFETY CHECK
# ===============================================================================================

if probabilities.ndim != 2:
    raise ValueError(
        f"Expected 2D probability matrix, received shape {probabilities.shape}"
    )


if probabilities.shape[1] != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, but model returned {probabilities.shape[1]} probability columns."
    )


# ===============================================================================================
# WORKING DATAFRAME
# ===============================================================================================

working_df = df[["Date", "Open", "High", "Low", "Close"]].copy()


# ===============================================================================================
# STORE ALL FIVE CLASS PROBABILITIES
# ===============================================================================================

for class_id in range(NUM_CLASSES):
    working_df[f"CLASS_{class_id}_PROB"] = probabilities[:, class_id]


# ===============================================================================================
# PREDICTED CLASS
# ===============================================================================================

working_df["PREDICTED_CLASS"] = np.argmax(probabilities, axis=1)

# ===============================================================================================
# MAX PREDICTED PROBABILITY
# ===============================================================================================

working_df["PREDICTED_PROBABILITY"] = np.max(probabilities, axis=1)

# ===============================================================================================
# PREDICTED CLASS MOVEMENT
# ===============================================================================================

working_df["MOVEMENT_PCT"] = working_df["PREDICTED_CLASS"].map(
    {
        class_id: TARGET_SETTINGS["classes"][class_id]["median_movement_pct"]
        for class_id in range(NUM_CLASSES)
    }
)


# ===============================================================================================
# PREDICTED DIRECTION
# ===============================================================================================

working_df["DIRECTION"] = working_df["PREDICTED_CLASS"].map(
    {0: -1, 1: -1, 2: 0, 3: 1, 4: 1}
)


# ===============================================================================================
# ENTRY PRICES
# ===============================================================================================

working_df["NEXT_ENTRY"] = working_df["Open"].shift(-1)

# BUY enters at ASK
working_df["ASK_PRICE"] = working_df["NEXT_ENTRY"] + spread

# SELL enters at BID
working_df["BID_PRICE"] = working_df["NEXT_ENTRY"]

working_df["ENTRY_BUY"] = working_df["ASK_PRICE"]
working_df["ENTRY_SELL"] = working_df["BID_PRICE"]


# ===============================================================================================
# TP / SL DISTANCE
#
# Class median movement percentage
#
# Example:
#
# Entry = 1.10000
# Movement = 0.164388%
#
# distance = 1.10000 * 0.00164388
#          = 0.001808268
#
# BUY:
#   TP = entry + distance
#   SL = entry - distance
#
# SELL:
#   TP = entry - distance
#   SL = entry + distance
# ===============================================================================================

movement_fraction = working_df["MOVEMENT_PCT"] / 100.0

working_df["BUY_DISTANCE"] = working_df["ENTRY_BUY"] * movement_fraction
working_df["SELL_DISTANCE"] = working_df["ENTRY_SELL"] * movement_fraction


# ===============================================================================================
# BUY TP / SL
# ===============================================================================================

working_df["TP_BUY"] = working_df["ENTRY_BUY"] + working_df["BUY_DISTANCE"]
working_df["SL_BUY"] = working_df["ENTRY_BUY"] - working_df["BUY_DISTANCE"]


# ===============================================================================================
# SELL TP / SL
# ===============================================================================================

working_df["TP_SELL"] = working_df["ENTRY_SELL"] - working_df["SELL_DISTANCE"]
working_df["SL_SELL"] = working_df["ENTRY_SELL"] + working_df["SELL_DISTANCE"]


# ===============================================================================================
# LOT SIZE
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


def run_backtest(working_df, threshold):

    # ===========================================================================================
    # SIGNAL GENERATION
    #
    # IMPORTANT:
    #
    # Class 2 is NEVER tradable.
    #
    # The highest-probability class is selected first.
    #
    # Then:
    #
    #   Class 0/1 -> SELL
    #   Class 2   -> NO TRADE
    #   Class 3/4 -> BUY
    #
    # Finally, the predicted class probability must be
    # >= threshold.
    # ===========================================================================================

    predicted_class = working_df["PREDICTED_CLASS"].to_numpy()

    predicted_probability = working_df["PREDICTED_PROBABILITY"].to_numpy()

    signals = np.zeros(len(working_df), dtype=np.int8)

    # Probability confidence filter
    confident = predicted_probability >= threshold

    # SELL classes
    sell_mask = confident & np.isin(predicted_class, list(DOWN_CLASSES))

    signals[sell_mask] = -1

    # BUY classes
    buy_mask = confident & np.isin(predicted_class, list(UP_CLASSES))

    signals[buy_mask] = 1

    # Class 2 remains 0 = NO TRADE

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

    # Movement distance in pips
    buy_distance_pips = working_df["BUY_DISTANCE"].to_numpy() / pip_size

    sell_distance_pips = working_df["SELL_DISTANCE"].to_numpy() / pip_size

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

    trade_classes = []

    trade_movements = []

    trade_probabilities = []

    trade_durations = []

    # ===========================================================================================
    # BACKTEST LOOP
    # ===========================================================================================

    i = 0

    n = len(signals)

    while i < n - 1:

        signal = signals[i]

        # ---------------------------------------------------------------------------------------
        # No trade
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

        if sl_pips <= 0:

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

        # =======================================================================================
        # BUY
        # =======================================================================================

        if signal == 1:

            sl = sl_buy[i]

            tp = tp_buy[i]

            entry = entry_buy[i]

            predicted_class_i = predicted_class[i]

            predicted_probability_i = predicted_probability[i]

            trade_movement = working_df["MOVEMENT_PCT"].iloc[i]

            # -----------------------------------------------------------------------------------
            # IMPORTANT:
            #
            # NO TARGET_HORIZON IS USED HERE.
            #
            # The trade remains open until:
            #
            #     1. STOP LOSS is hit
            #     2. TAKE PROFIT is hit
            #
            # There is NO 3-hour maximum holding period.
            # -----------------------------------------------------------------------------------

            j = i + 1

            trade_closed = False

            while j < n:

                # -------------------------------------------------------------------------------
                # STOP LOSS
                # -------------------------------------------------------------------------------

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

                    trade_durations.append(j - i)

                    i = j + 1

                    trade_closed = True

                    break

                # -------------------------------------------------------------------------------
                # TAKE PROFIT
                # -------------------------------------------------------------------------------

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

                    trade_durations.append(j - i)

                    i = j + 1

                    trade_closed = True

                    break

                # -------------------------------------------------------------------------------
                # Neither TP nor SL was hit on this candle.
                #
                # Continue holding the trade.
                # --------------------------------------------------------------------------------

                j += 1

            # -----------------------------------------------------------------------------------
            # DATASET ENDED BEFORE TP/SL
            #
            # IMPORTANT:
            #
            # We DO NOT:
            #
            #   - close the trade at the final candle
            #   - use the final Close price
            #   - create artificial P&L
            #   - count it as a WIN
            #   - count it as a LOSS
            #
            # The trade simply remains unresolved because the historical
            # dataset ended before the actual TP or SL was reached.
            #
            # This is NOT a premature trade closure.
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

            predicted_class_i = predicted_class[i]

            predicted_probability_i = predicted_probability[i]

            trade_movement = working_df["MOVEMENT_PCT"].iloc[i]

            # -----------------------------------------------------------------------------------
            # IMPORTANT:
            #
            # NO TARGET_HORIZON IS USED HERE.
            #
            # The trade remains open until:
            #
            #     1. STOP LOSS is hit
            #     2. TAKE PROFIT is hit
            #
            # There is NO 3-hour maximum holding period.
            # -----------------------------------------------------------------------------------

            j = i + 1

            trade_closed = False

            while j < n:

                # -------------------------------------------------------------------------------
                # STOP LOSS
                # -------------------------------------------------------------------------------

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

                    trade_durations.append(j - i)

                    i = j + 1

                    trade_closed = True

                    break

                # -------------------------------------------------------------------------------
                # TAKE PROFIT
                # -------------------------------------------------------------------------------

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

                    trade_durations.append(j - i)

                    i = j + 1

                    trade_closed = True

                    break

                # -------------------------------------------------------------------------------
                # Neither TP nor SL was hit on this candle.
                #
                # Continue holding the trade.
                # --------------------------------------------------------------------------------

                j += 1

            # -----------------------------------------------------------------------------------
            # DATASET ENDED BEFORE TP/SL
            #
            # IMPORTANT:
            #
            # We DO NOT:
            #
            #   - close the trade at the final candle
            #   - use the final Close price
            #   - create artificial P&L
            #   - count it as a WIN
            #   - count it as a LOSS
            #
            # The trade remains unresolved because the historical
            # dataset ended before the actual TP or SL was reached.
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
    # FINAL RESULT
    # ===========================================================================================

    return {
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
        "CLASS_0_TRADES": class_trade_counts[0],
        "CLASS_0_WIN_RATE": round(class_win_rates[0] * 100, 2),
        "CLASS_1_TRADES": class_trade_counts[1],
        "CLASS_1_WIN_RATE": round(class_win_rates[1] * 100, 2),
        "CLASS_2_TRADES": class_trade_counts[2],
        "CLASS_2_WIN_RATE": round(class_win_rates[2] * 100, 2),
        "CLASS_3_TRADES": class_trade_counts[3],
        "CLASS_3_WIN_RATE": round(class_win_rates[3] * 100, 2),
        "CLASS_4_TRADES": class_trade_counts[4],
        "CLASS_4_WIN_RATE": round(class_win_rates[4] * 100, 2),
    }


# ===============================================================================================
# THRESHOLD CANDIDATES
# ===============================================================================================

# Only probabilities of TRADABLE classes are relevant.
#
# Class 2 is deliberately excluded because it is NEVER a trade.
#
# We round to 3 decimals so that the number of backtests remains
# manageable while still giving us 0.001 probability resolution.

tradable_probability_columns = [
    "CLASS_0_PROB",
    "CLASS_1_PROB",
    "CLASS_3_PROB",
    "CLASS_4_PROB",
]
threshold_values = pd.concat(
    [working_df[column] for column in tradable_probability_columns], ignore_index=True
)
thresholds = sorted(set(threshold_values.round(3).dropna()))

print("\nNUMBER OF THRESHOLD CANDIDATES:", len(thresholds))


# ===============================================================================================
# RUN ALL THRESHOLDS
# ===============================================================================================

results = []


for index, threshold in enumerate(thresholds, start=1):

    print(f"CURRENTLY ON THRESHOLD {threshold:.3f} >> {index}/{len(thresholds)}")

    result = run_backtest(working_df=working_df, threshold=threshold)
    results.append(result)


# ===============================================================================================
# RESULT DATAFRAME
# ===============================================================================================

result_df = pd.DataFrame(results)


# ===============================================================================================
# FILTER MINIMUM TRADE COUNT
# ===============================================================================================
result_df = result_df[result_df["TRADES"] >= TRADE_NO].copy()


# ===============================================================================================
# SORT RESULTS
# ===============================================================================================

# result_df = result_df.sort_values(by=["WIN_RATE","PROFIT_FACTOR"],ascending=[ False,False])

result_df = result_df.sort_values(by="WIN_RATE", ascending=False)

# result_df = result_df[(result_df["TRADES"] >= 225) & (result_df["PROFIT_FACTOR"] > 1.0) &
# (result_df["MAX_DRAWDOWN"] <= 6.0)].copy()
# result_df = result_df.sort_values(by="WIN_RATE",ascending=False).reset_index(drop=True)

# ===============================================================================================
# DISPLAY
# ===============================================================================================

print("\n\n" + "=" * 100)

print(f"SYMBOL: {SYMBOL}")
print(f"TARGET: {TARGET_NAME}")
print(f"HORIZON: {TARGET_HORIZON} candles = {TARGET_HORIZON * 10 / 60:.1f} hours")

print(f"MINIMUM TRADES: {TRADE_NO}")

print(f"RISK PER TRADE: {RISK_PERCENT}%")
print("TP / SL: 1:1")
print("CLASS 2: NO TRADE")

print(f"FORWARD TEST START: {df['Date'].min()}")
print(f"FORWARD TEST END: {df['Date'].max()}")
print("=" * 100)


if len(result_df) == 0:
    print(f"\nNO THRESHOLD PASSED THE MINIMUM TRADE REQUIREMENT OF {TRADE_NO}.")

else:

    print(f"\nTHRESHOLDS PASSING >= {TRADE_NO} TRADES: {len(result_df)}")
    print("\nTOP 20 THRESHOLDS:\n")

    print(result_df.head(20).to_string(index=False))


# ===============================================================================================
# SAVE RESULTS
# ===============================================================================================

RESULT_FILE = os.path.join(
    MODEL_FOLDER, f"THRESHOLD_RESULTS_{TARGET_NAME}_{CD_TIME}_{SYMBOL}.csv"
)

result_df.to_csv(RESULT_FILE, index=False)
print(f"\nThreshold results saved to:\n{RESULT_FILE}")
