
import os
import time
import joblib
import MetaTrader5 as mt5
from func import calc_lot_size,place_buy,place_sell,MULTICLASS_MOVE_SL_AND_PARTIAL_CLOSE,get_pip_info
from func import MULTICLASS_LOG_TRADE,MULTICLASS_Entry_Filtering,MULTICLASS_DATA_PREDICTION,set_logger,correlation_check,spread_filter,normalize_symbol






# Initialize MT5 once
if not mt5.initialize():
    raise RuntimeError("[[BAD]] MT5 initialization failed")
print("[[GOOD]] MT5 initialized successfully")





def load_model(CD_TIME, SYMBOL,TARGET_NAME, MODEL_FOLDER, NUM_CLASSES):

    symbol_normalized = normalize_symbol(symbol=SYMBOL)
    model_path = os.path.join(MODEL_FOLDER, f"HL_MAIN_{TARGET_NAME}_{CD_TIME}_{symbol_normalized}_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"\nModel not found:\n{model_path}")

    bundle = joblib.load(model_path)

    required_keys = ["model", "features", "target_settings"]

    missing_keys = [key for key in required_keys if key not in bundle]

    if missing_keys:
        raise KeyError(f"\nModel is missing required keys: {missing_keys}")

    model = bundle["model"]
    features = bundle["features"]
    target_settings = bundle["target_settings"]


    classes = target_settings["classes"]

    for class_id in range(NUM_CLASSES):

        if class_id not in classes:
            raise ValueError(f"Class {class_id} is missing from target_settings.")

        if "median_movement_pct" not in classes[class_id]:
            raise ValueError(f"median_movement_pct missing for class {class_id}.")

    print("\n[[ GOOD ]] Multiclass model loaded")
    print(f"MODEL : {model_path}")
    print(f"FEATURES: {len(features)}")

    return {
        "model": model,
        "features": features,
        "target_settings": target_settings,
    }




def FOREX_TRADING(SYMBOL, logger):
    set_logger(logger)
    logger.info("BOT STARTED")

    logger.info(f'\n [[[[[  THIS FUNCTION IS FOR : {SYMBOL} ]]]]')

    # ===============================================================================================
    # SETTINGS
    # ===============================================================================================

    CD_TIME = "10M"

    TARGET_NAME = "TARGET_5CLASS"

    NUM_CLASSES = 5

    risk_percent = 0.01

    SL_RATIO = 1.0
    TP_RATIO = 1.0


    W_threshold = 0.5 # SAMPLE TO BE REMOVED LATER ON

    up_classes = [3,4]
    down_classes = [0,1]

    MODEL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"ALL_MODELS")
    MODEL = load_model(CD_TIME=CD_TIME, SYMBOL=SYMBOL, MODEL_FOLDER=MODEL_FOLDER, 
                       TARGET_NAME=TARGET_NAME, NUM_CLASSES=NUM_CLASSES)

    try:
        while True:

            df_info = MULTICLASS_DATA_PREDICTION(mt5=mt5, SYMBOL=SYMBOL, MODEL=MODEL, NUM_CLASSES=NUM_CLASSES)

            class_probability = df_info["predicted_probability"]
            median_movement_pct = df_info["median_movement_pct"]/100
            predicted_class = df_info["predicted_class"]
            predicted_direction = df_info["predicted_direction"]
            df = df_info["Data_Frame"]

            all_class_probabilities = df_info["class_probabilities"]

            down_probability = (all_class_probabilities[0] + all_class_probabilities[1])
            up_probability = ( all_class_probabilities[3] + all_class_probabilities[4])

            logger.info(f'CURRENTLY PERFORMING SYMBOL : {SYMBOL} \n')

            pip_info = get_pip_info(mt5, SYMBOL)
            pip_size = pip_info["pip_size"]
            pip_value_per_lot = pip_info["pip_value_per_lot"]

            tick = mt5.symbol_info_tick(SYMBOL)
            ask_price = tick.ask
            bid_price = tick.bid
            spread = ask_price - bid_price  # real-time spread

            symbol_info = mt5.symbol_info(SYMBOL)
            spread_points = spread / symbol_info.point
            max_spread_points = 70   # example threshold

            if spread_points > max_spread_points:
                logger.info(f"[[WARNING]] Spread too high: {spread_points:.2f} points, skipping trade")
                continue
            
            positions = mt5.positions_get(symbol=SYMBOL)

            if positions is None:
                logger.info("[[WARNING]] Error fetching positions")
                has_open_trade = False
                has_buy = False
                has_sell = False

            elif len(positions) == 0:
                has_open_trade = False
                has_buy = False
                has_sell = False

            else:
                buy_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_BUY]
                sell_positions = [p for p in positions if p.type == mt5.ORDER_TYPE_SELL]

                has_buy = len(buy_positions) > 0
                has_sell = len(sell_positions) > 0
                has_open_trade = has_buy or has_sell   # [[GOOD]] FIX

                if has_buy:
                    logger.info("[[WARNING]] BUY exists")

                if has_sell:
                    logger.info("[[WARNING]] SELL exists")

                    
            account_info = mt5.account_info()
            balance = account_info.balance

            logger.info(f'''
            ===================== ACCOUNT INFORMATIONS ==============================

            SYMBOL : {SYMBOL}

            Account Number : {account_info.login}
            Balance        : {account_info.balance}
            Equity         : {account_info.equity}
            Free Margin    : {account_info.margin_free}
            Leverage       : {account_info.leverage}

            pip_size           : {pip_size}
            pip_value_per_lot  : {pip_value_per_lot}

            SL_RATIO   : {SL_RATIO}
            TP_RATIO   : {TP_RATIO}

            ask_price : {ask_price}
            bid_price : {bid_price}
            spread    : {spread}

            PREDICTED CLASS  : {predicted_class}
            PREDICTED DIRECTION : {predicted_direction}
            CLASS PROBABILITY : {class_probability*100}%

            ACTUAL MEDIAN MOVEMENT PERCENTAGE : {df_info["median_movement_pct"]:.4f}%
            USED MEDIAN MOVEMENT FRACTION : {median_movement_pct:.8f}

            up_probability   : {up_probability:.2%}
            down_probability : {down_probability:.2%}

            ----------------------------------------------------------------------------------
            all_class_probabilities : {all_class_probabilities}
            ----------------------------------------------------------------------------------

            ========================================================================
            ''')


            if not has_open_trade:
                ##================================= MAIN DIRECTION TRADING ==================================================
                if up_probability >= W_threshold and int(predicted_class) in up_classes:

                    logger.info('CHECKING FOR BUY ENTRY LOGIC')

                    corr_allowed = correlation_check(mt5=mt5,new_symbol=SYMBOL,new_trade_type="BUY")

                    if corr_allowed:

                        logger.info("\n >>>>>>>>> BUY CORRELATION PASSED <<<<<<<<<< \n ")

                        signal_entry_buy = MULTICLASS_Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
                            direction="BUY",median_movement_pct=median_movement_pct)

                        if signal_entry_buy is None:
                            logger.info("[[BAD]] BUY skipped (no confirmation)")
                            continue

                        else:
                            logger.info("[[UP]] FINAL SIGNAL: BUY")

                            tick = mt5.symbol_info_tick(SYMBOL)
                            entry_buy = tick.ask   # REAL execution price

                            median_movement_price = entry_buy * median_movement_pct

                            SL_distance = median_movement_price * SL_RATIO
                            TP_distance = median_movement_price * TP_RATIO
                            SL_pips = SL_distance/ pip_size 

                            spread_ok = spread_filter(mt5=mt5,SYMBOL=SYMBOL,SL_distance=SL_distance)

                            if not spread_ok:
                                continue

                            SL_buy = entry_buy - SL_distance
                            TP_buy = entry_buy + TP_distance

                            lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
                            logger.info(f'LOTS SIZE USED: {lot_size}')

                            result = place_buy(mt5, SYMBOL, lot_size, entry_buy, SL_buy, TP_buy)
                            logger.info(f"BUY ORDER PLACED & RESULT: {result} ")

                            MULTICLASS_LOG_TRADE(mt5=mt5,symbol=SYMBOL,direction="BUY",entry_price=entry_buy,SL=SL_buy,TP=TP_buy,lot_size=lot_size,
                                model_class=predicted_class,class_direction=predicted_direction,class_proba=class_probability,order_result=result)
                    
                    else:
                        logger.info(f'\n BUY TRADE BLOCKED DUE TO CURRENCY EXPOSER FOR : {SYMBOL} \n')


                elif down_probability >= W_threshold  and int(predicted_class) in down_classes:

                    logger.info('CHECKING FOR SELL ENTRY LOGIC')

                    corr_allowed = correlation_check(mt5=mt5,new_symbol=SYMBOL,new_trade_type="SELL")

                    if corr_allowed:

                        logger.info(">>>>>>>>> SELL CORRELATION PASSED <<<<<<<<<< ")
                        
                        signal_entry_sell = MULTICLASS_Entry_Filtering(mt5=mt5,SYMBOL=SYMBOL,df=df,
                            direction="SELL",median_movement_pct=median_movement_pct)

                        if signal_entry_sell is None:
                            logger.info("[[BAD]] SELL skipped (no confirmation)")
                        else:
                            logger.info("[[DOWN]] FINAL SIGNAL: SELL")

                            tick = mt5.symbol_info_tick(SYMBOL)
                            entry_sell = tick.bid
                            median_movement_price = entry_sell * median_movement_pct

                            SL_distance = median_movement_price * SL_RATIO
                            TP_distance = median_movement_price * TP_RATIO
                            SL_pips = SL_distance/ pip_size 


                            # CHECKING FOR SPREAD IMPACT
                            spread_ok = spread_filter(mt5=mt5,SYMBOL=SYMBOL,SL_distance=SL_distance)

                            if not spread_ok:
                                continue

                            SL_sell = entry_sell + SL_distance
                            TP_sell = entry_sell - TP_distance

                            lot_size = calc_lot_size(mt5=mt5,balance=balance,risk_percent=risk_percent,sl_pips=SL_pips,pip_value_per_lot=pip_value_per_lot,SYMBOL=SYMBOL)
                            logger.info(f'LOTS SIZE USED: {lot_size}')

                            result = place_sell(mt5, SYMBOL, lot_size, entry_sell, SL_sell, TP_sell)
                            logger.info(f"SELL ORDER RESULT: {result}" )

                            MULTICLASS_LOG_TRADE(mt5=mt5,symbol=SYMBOL,direction="SELL",entry_price=entry_sell,SL=SL_sell,TP=TP_sell,lot_size=lot_size,
                                model_class=predicted_class,class_direction=predicted_direction,class_proba=class_probability,order_result=result)
                    
                    else:
                        logger.info(F'SELL TRADE BLOCKED DUE TO CURRENCY EXPOSER FOR : {SYMBOL}')
                    
                else:
                    logger.info("\n [[LOADING]] FINAL SIGNAL: NO TRADE \n")
                    logger.info(
                        f"NO TRADE: UP={up_probability:.2%}, "
                        f"DOWN={down_probability:.2%}, "
                        f"threshold={W_threshold:.0%}"
                    )



                while True:
                    logger.info(f'CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION FOR {SYMBOL}')
                    output = MULTICLASS_MOVE_SL_AND_PARTIAL_CLOSE(mt5=mt5, SYMBOL=SYMBOL, MODEL=MODEL, NUM_CLASSES=NUM_CLASSES)
                    if output == 1:
                        logger.info('TP/SL + PARTIAL CONDITION MET')
                        break
                    time.sleep(2)

                    
            elif has_open_trade:
                while True:
                    logger.info(f'CURRENTLY HANDLING TP/SL + PARTIAL PROFIT EXECUTION FOR {SYMBOL}')
                    output = MULTICLASS_MOVE_SL_AND_PARTIAL_CLOSE(mt5=mt5, SYMBOL=SYMBOL, MODEL=MODEL, NUM_CLASSES=NUM_CLASSES)
                    if output == 1:
                        logger.info('TP/SL + PARTIAL CONDITION MET')
                        break
                    time.sleep(2)



    finally:
        # Shutdown MT5 only once, when the bot stops
        mt5.shutdown()
        logger.info("[[GOOD]] MT5 shutdown successfully")
