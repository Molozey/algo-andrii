from utils.connectorInterface import SaxoOrderInterface
import time
import pandas as pd
import pprint


saxo = SaxoOrderInterface()
while True:
    # print(saxo.portfolio_open_positions())
    print(pd.DataFrame(saxo.get_asset_data_hist(ticker='CHFJPY', density=1, amount_intervals=100)).iloc[-2])
    print('-------')
    # print(saxo.get_actual_data(['CHFJPY'], mode='all'))
    time.sleep(60)

