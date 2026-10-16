from multiprocessing import Process
# from ATR_TRADE_HIGH_NXT import FOREX_TRADING
from MULTI_CLASS_LIVE_TRADE import FOREX_TRADING

import logging


def setup_logger(symbol):

    logger = logging.getLogger(symbol)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(f'\n %(asctime)s | [{symbol}] | %(message)s \n')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def run_symbol(symbol):

    logger = setup_logger(symbol)

    logger.info(f'\n CURRENTLY RUNNING THIS SYMBOL [[ {symbol} ]] \n')

    FOREX_TRADING(SYMBOL=symbol,logger=logger)


if __name__ == "__main__":

    symbols = [
        "AUDCAD","AUDJPY","AUDUSD","BTCUSD","CADJPY",
        "ETHUSD","EURGBP","EURJPY","EURUSD","GBPCAD",
        "GBPJPY","GBPUSD","NZDJPY","NZDUSD","USDCAD",
        "USDCHF","USDJPY","XAGUSD","XAUUSD",
    ]

    symbols = [symbol  for symbol in symbols]

    processes = []

    for symbol in symbols:

        p = Process(target=run_symbol,args=(symbol,))

        p.start()

        processes.append(p)

    for p in processes:
        p.join()


