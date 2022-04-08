from abc import ABC, abstractmethod
from saxo_openapi import API
from saxo_openapi.contrib.orders import tie_account_to_order, MarketOrderFxSpot, StopOrderFxSpot, LimitOrderFxSpot
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
        self._token = "eyJhbGciOiJFUzI1NiIsIng1dCI6IkRFNDc0QUQ1Q0NGRUFFRTlDRThCRDQ3ODlFRTZDOTEyRjVCM0UzOTQifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiVG1XWGlqam1ZdFk0ZmF0MkIwZDdYdz09IiwiY2lkIjoiVG1XWGlqam1ZdFk0ZmF0MkIwZDdYdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiMTgyYTgyYzZhYjVlNGVlN2FlOWRhNzQ4NTk3ODVlOTAiLCJkZ2kiOiI4NCIsImV4cCI6IjE2NDk1MDQ3OTEiLCJvYWwiOiIxRiJ9.NG7Wf_iLNaTjDWKYkWVT0ZCSbBSjbO8ExDLBOKAncsnuAEZNNAczWFqIbEHLNXw0qIavAVi8Zhgjlr8C3J0mBg"

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
        list_tickers = ['CHFJPY', ...]
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

    def place_order(self, dict_orders, order_type, order_price=None):
    '''
    order_type: "market" or "limit"
    order_price: if order_type == "limit", then order_price is nesessary
    dict_orders = {fx_ticket: Amount}
    fx_ticket: text
    Amount: -int, +int 
    '''
    for ticket, amount in dict_orders.items():
        try:
            # find the Uic for Instrument
            uic = list(InstrumentToUic(client, AccountKey, spec={'Instrument': ticket}).values())[0]
            if order_type == "market":
                order = tie_account_to_order(AccountKey, MarketOrderFxSpot(Uic=uic, Amount=amount))
            elif order_type == "limit":
                order = tie_account_to_order(AccountKey, LimitOrderFxSpot(Uic=uic, Amount=amount, OrderPrice=order_price))
            r = tr.orders.Order(data=order)
            rv = client.request(r)
            print(f'{ticket} amount {amount}: {rv}')
        except Exception as error:
            print(f'{ticket}: {error}')
        time.sleep(1)

    def get_asset_data_hist(self, symbol, interval=None, from_='1000-01-01', to=str(datetime.date.today()), api_token='5f75d2d79bbbb4.84214003'):
        # idditioal information: https://eodhistoricaldata.com/financial-apis/list-supported-forex-currencies/
        # for indexes use {symbol}{.FOREX}
        # for indexes use {symbol}{.INDX}
        api_token = self._apiToken
        if interval == None:
            url = f'https://eodhistoricaldata.com/api/eod/{symbol}?api_token={api_token}&fmt=json&from={from_}&to={to}'
        else:
            # change type of 'from_' & 'to' to calculate amount days between it
            from_ = time.mktime(datetime.datetime.strptime(from_, "%Y-%m-%d %H:%M:%S").timetuple())
            to = time.mktime(datetime.datetime.strptime(to, "%Y-%m-%d %H:%M:%S").timetuple())
            days_in_term = int((to - from_)/60/60/24)
            if (interval == '1h') & (days_in_term <= 7200) | (interval == '5m') & (days_in_term <= 600) | (interval == '1m') & (days_in_term <= 120):
                url = f'https://eodhistoricaldata.com/api/intraday/{symbol}?api_token={api_token}&fmt=json&interval={interval}&from={from_}&to={to}'
            else:
                return "You must use intervals '1h'/'5m'/'1m' and all of them can contain maximum 7200/600/120 days accordingly."
        print(url)
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        if bool(data) == False:
            return "The data with the parameters does not exist on the 'eodhistoricaldata.com' server."
        return data


