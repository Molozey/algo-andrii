from abc import ABC, abstractmethod
from saxo_openapi import API
from saxo_openapi.contrib.orders import (tie_account_to_order, MarketOrderFxSpot, StopOrderFxSpot)
from saxo_openapi.contrib.orders.onfill import TakeProfitDetails, StopLossDetails
from saxo_openapi.contrib.util import InstrumentToUic
from saxo_openapi.contrib.session import account_info
import saxo_openapi.endpoints.trading as tr
# import saxo_openapi.endpoints.portfolio as pf
from pprint import pprint
import time
import json

class AbstractOrderInterface:
    def __init__(self):
        token = "eyJhbGciOiJFUzI1NiIsIng1dCI6IkRFNDc0QUQ1Q0NGRUFFRTlDRThCRDQ3ODlFRTZDOTEyRjVCM0UzOTQifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiVG1XWGlqam1ZdFk0ZmF0MkIwZDdYdz09IiwiY2lkIjoiVG1XWGlqam1ZdFk0ZmF0MkIwZDdYdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiNmRkMjBkMGYwYWI3NDZjZWE1NThiMzc1NWM3MzI4ZDkiLCJkZ2kiOiI4NCIsImV4cCI6IjE2NDk0MjEzOTMiLCJvYWwiOiIxRiJ9.Q0Y6xfW_2Hetns3SsRhV3LrGyddDW2TLeM5NUiGdGHw-TeXPa0M4SYWVU8FcifzmYLVm8PNv07n6TDZ2_T6wCg"
            
    @abstractmethod
    def get_fx_quote(list_tickers):
        '''
        return a dict with elements like (where ticker_n is ticker and a key for the dictinary):
            ticker_n : {'AssetType': 'FxSpot',
                        'LastUpdated': '2022-04-07T17:25:09.946000Z',
                        'PriceSource': 'SBFX',
                        'Quote': {'Amount': 100000,
                        'Ask': 132.797,
                        'Bid': 132.757,
                        'DelayedByMinutes': 0,
                        'ErrorCode': 'None',
                        'MarketState': 'Open',
                        'Mid': 132.777,
                        'PriceSource': 'SBFX',
                        'PriceSourceType': 'Firm',
                        'PriceTypeAsk': 'Tradable',
                        'PriceTypeBid': 'Tradable'},
                        'Uic': 8}
        '''
        # transfer tickers into Uids:
        str_uics = ''
        for ticker in list_tickers:
            try:
                uic = str(list(InstrumentToUic(client, AccountKey, spec={'Instrument': ticker}).values())[0]) + ','
                str_uics += uic
            except Exception as error:
                print(f'{ticker}: {error}')
        str_uics = str_uics[:-1]

        params = {
            "Uics": str_uics,
            "AccountKey": AccountKey,
            "AssetType": 'FxSpot'
            }
        # ask quotes:
        r = tr.infoprices.InfoPrices(params)
        # combine two lists in one dict:
        return dict(zip(list_tickers, client.request(r)['Data']))

    @abstractmethod
    def place_open_order(client, AccountKey, dict_orders):
        '''
        dict_orders = {fx_ticker: Amount}
        fx_ticker: text
        Amount: -int, +int 
        '''
        for ticker, amount in dict_orders.items():
            try:
                # find the Uic for Instrument
                uic = list(InstrumentToUic(client, AccountKey, spec={'Instrument': ticker}).values())[0]
                order = tie_account_to_order(AccountKey, MarketOrderFxSpot(Uic=uic, Amount=amount))
                r = tr.orders.Order(data=order)
                rv = client.request(r)
                print(f'{ticket} amount {amount}: {rv}')
            except Exception as error:
                print(f'{ticket}: {error}')
            time.sleep(1)
            
    @abstractmethod
    def validate_open_order(self):
        pass

    @abstractmethod
    def place_close_order(self):
        pass

    @abstractmethod
    def validate_close_order(self):
        pass

# WHERE WE WILL PUT NEXT ? :
    # client to requests  process
    client = API(access_token=token)
    AccountKey = account_info(client).AccountKey
    
    
