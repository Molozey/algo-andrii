from abc import ABC, abstractmethod
from saxo_openapi import API
from saxo_openapi.contrib.orders import (tie_account_to_order, MarketOrderFxSpot, StopOrderFxSpot)
from saxo_openapi.contrib.orders.onfill import TakeProfitDetails, StopLossDetails
from saxo_openapi.contrib.util import InstrumentToUic
from saxo_openapi.contrib.session import account_info
import saxo_openapi.endpoints.trading as tr
import saxo_openapi.endpoints.rootservices as rs
import saxo_openapi.endpoints.referencedata as referenceData
import saxo_openapi.definitions.accounthistory as ah
# import saxo_openapi.endpoints.portfolio as pf
from pprint import pprint
import time
import json
import datetime
import urllib
import requests
from saxoToolKit import *

class AbstractOrderInterface:
    def __init__(self):
        pass

    @abstractmethod
    def get_actual_data(self, *args):
        pass

    @abstractmethod
    def place_order(self, dict_orders):
        pass

    @abstractmethod
    def get_asset_data_hist(self, symbol, interval=None, from_='1000-01-01', to=str(datetime.date.today()), api_token='5f75d2d79bbbb4.84214003'):
        pass

    @abstractmethod
    def validate_open_order(self):
        pass

    @abstractmethod
    def validate_close_order(self):
        pass


class SaxoOrderInterface(AbstractOrderInterface):
    def __init__(self):
        self._token = "eyJhbGciOiJFUzI1NiIsIng1dCI6IkRFNDc0QUQ1Q0NGRUFFRTlDRThCRDQ3ODlFRTZDOTEyRjVCM0UzOTQifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiNGFOUTNBSE41TnRDUmJ8UjNkNy1hdz09IiwiY2lkIjoiNGFOUTNBSE41TnRDUmJ8UjNkNy1hdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiMjY4ZDk5N2Y3NWY5NDdhZGIzOGYzN2NhNmNmMzhiMzIiLCJkZ2kiOiI4NCIsImV4cCI6IjE2NDk0NjAwOTkiLCJvYWwiOiIxRiJ9.6pj_qgmVs0xQc_W60hdg6LO8NamQorpvXdFFkjIBFGc-RkyNLIk-QemyaB_YoCo3ZjTqQ1QIhQDubXOEL7Ddig"

        self._client = API(access_token=self._token)
        self._AccountKey = account_info(self._client).AccountKey
        self._ClientKey = account_info(self._client).ClientKey
        self._apiToken = '5f75d2d79bbbb4.84214003'
        r = rs.diagnostics.Get()
        rv = self._client.request(r)
        assert rv is None and r.status_code == 200
        print('connection created')
        del r, rv

        #   Sample of OwnRequest
        # OwnRequest = json.loads(self._client.OWNREQUEST(method="get", url=f'https://gateway.saxobank.com/sim/openapi/ref/v1/currencypairs/?AccountKey={self._AccountKey}&ClientKey={self._ClientKey}',
        #                               request_args={'params': None}))
        # print(OwnRequest['Data'])

    def get_actual_data(self, list_tickers):
        '''
        list_tickers = ['CHFJPY']
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
                uic = str(list(InstrumentToUic(self._client, self._AccountKey, spec={'Instrument': ticker}).values())[0]) + ','
                str_uics += uic
            except Exception as error:
                print(f'{ticker}: {error}')
        str_uics = str_uics[:-1]

        params = {
            "Uics": str_uics,
            "AccountKey": self._AccountKey,
            "AssetType": 'FxSpot'
            }
        # ask quotes:
        r = tr.infoprices.InfoPrices(params)
        # combine two lists in one dict:
        answer = dict(zip(list_tickers, self._client.request(r)['Data']))
        return answer[list_tickers[0]]['Quote']['Mid']

    def place_order(self, dict_orders):
        '''
        dict_orders = {fx_ticker: Amount}
        fx_ticker: text
        Amount: -int, +int
        '''

        # Если amount -1 <=> закрытию позиции
        for ticker, amount in dict_orders.items():
            try:
                # find the Uic for Instrument
                uic = list(InstrumentToUic(self._client, self._AccountKey, spec={'Instrument': ticker}).values())[0]
                order = tie_account_to_order(self._AccountKey, MarketOrderFxSpot(Uic=uic, Amount=amount))
                r = tr.orders.Order(data=order)
                rv = self._client.request(r)
                print(f'{ticker} amount {amount}: {rv}')
                return True
            except Exception as error:
                print(f'{ticker}: {error}')
                return False




