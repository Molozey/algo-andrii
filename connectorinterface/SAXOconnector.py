from abc import ABC, abstractmethod
from saxo_openapi import API
from saxo_openapi.contrib.orders import tie_account_to_order, MarketOrderFxSpot, StopOrderFxSpot, LimitOrderFxSpot
from saxo_openapi.contrib.orders.onfill import TakeProfitDetails, StopLossDetails
import saxo_openapi.endpoints.chart as chart
from saxo_openapi.contrib.util import InstrumentToUic
from saxo_openapi.contrib.session import account_info
import saxo_openapi.endpoints.trading as tr
import saxo_openapi.endpoints.rootservices as rs
import saxo_openapi.endpoints.referencedata as referenceData
import saxo_openapi.definitions.accounthistory as ah
import saxo_openapi.endpoints.portfolio as pf
from pprint import pprint
import time
import json
import datetime
import urllib
import requests

class AbstractOrderInterface:
    def __init__(self):
        pass

    @abstractmethod
    def get_actual_data(self, *args):
        pass

    @abstractmethod
    def place_order(self, *args):
        pass

    @abstractmethod
    def get_asset_data_hist(self, *args):
        pass

    @abstractmethod
    def validate_open_order(self, *args):
        pass

    @abstractmethod
    def validate_close_order(self, *args):
        pass


class SaxoOrderInterface(AbstractOrderInterface):
    def __init__(self, token):
        self._token = token

        self._client = API(access_token=self._token)
        self._AccountKey = account_info(self._client).AccountKey
        self._ClientKey = account_info(self._client).ClientKey
        r = rs.diagnostics.Get()
        rv = self._client.request(r)
        assert rv is None and r.status_code == 200
        print('connection created')
        del r, rv

        #   Sample of OwnRequest
        # OwnRequest = json.loads(self._client.OWNREQUEST(method="get", url=f'https://gateway.saxobank.com/sim/openapi/ref/v1/currencypairs/?AccountKey={self._AccountKey}&ClientKey={self._ClientKey}',
        #                               request_args={'params': None}))
        # print(OwnRequest['Data'])

    def get_actual_data(self, list_tickers, mode='bidPrice'):
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
        if mode == 'midPrice':
            return answer[list_tickers[0]]['Quote']['Mid']
        if mode == 'bidPrice':
            return answer[list_tickers[0]]['Quote']['Bid']
        else:
            return answer[list_tickers[0]]['Quote']

    def place_order(self, dict_orders, order_type="market", order_price=None):
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
                uic = list(InstrumentToUic(self._client, self._AccountKey, spec={'Instrument': ticket}).values())[0]
                if order_type == "market":
                    order = tie_account_to_order(self._AccountKey, MarketOrderFxSpot(Uic=uic, Amount=amount))
                elif order_type == "limit":
                    order = tie_account_to_order(self._AccountKey, LimitOrderFxSpot(Uic=uic, Amount=amount, OrderPrice=order_price))
                r = tr.orders.Order(data=order)
                rv = self._client.request(r)
                print(f'{ticket} amount {amount}: {rv}')
                return rv
            except Exception as error:
                print(f'{ticket}: {error}')
            time.sleep(1)

    def get_fx_quote(self, list_tickers):
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
        return dict(zip(list_tickers, self._client.request(r)['Data']))

    def stop_order(self, ticker, amount, order_type='market', order_price=None):
        '''
        ticker: {text}, ('CHFJPY')
        amount: {-int, +int}
        order_type: 'market', 'limit'
        order_price: None, {+int}
            if order_type='market': order_price=None
            if order_type='limit': order_price={+int}
                (!) if long (amount > 0): order_price ≥ bid_price
                (!) if short (amount < 0): order_price ≤ ask_price
        "ask", "bid" corrections exist for "opder_type = 'market'" as saxo open api request only limits orders which
        lower than "ask" (if amount < 0) and higher than "bid" (if amount > 0)
        '''
        ask_correction = 0.9995 # price must be lower then ask
        bid_correction = 1.0005 # price must be higher then bid

        ticker_data = self.get_fx_quote([ticker])[ticker]
        # print(ticker_data)
        ask = ticker_data['Quote']['Ask']
        # print(f"ask_1: {ask}")
        bid = ticker_data['Quote']['Bid']
        # print(f"bid_1: {bid}")
        uic = ticker_data['Uic']

        if order_type == 'market':
            if amount > 0:
                order = tie_account_to_order(self._AccountKey, StopOrderFxSpot(Uic=uic, Amount=amount, OrderPrice=bid*bid_correction))
            elif amount < 0:
                order = tie_account_to_order(self._AccountKey, StopOrderFxSpot(Uic=uic, Amount=amount, OrderPrice=ask*ask_correction))
        elif order_type == 'limit':
            order = tie_account_to_order(self._AccountKey, StopOrderFxSpot(Uic=uic, Amount=amount, OrderPrice=order_price))
        r = tr.orders.Order(data=order)
        try:
            rv = self._client.request(r)
        except Exception as e:
            print('May be price was changed a lot')
            print(e)

    def get_asset_data_hist(self, ticker, density, amount_intervals):
        '''
        ticker: text ("EURUSD")
        density: int, in seconds (min 60)
        amount_intervals: int, how many historical intervals with the density (max 1200)

        return: a list of dicts, where one element like the next
            {'CloseAsk': 1.64488,
            'CloseBid': 1.64408,
            'HighAsk': 1.64499,
            'HighBid': 1.64419,
            'LowAsk': 1.64472,
            'LowBid': 1.64392,
            'OpenAsk': 1.64493,
            'OpenBid': 1.64413,
            'Time': '2022-04-07T16:17:00.000000Z'}
        '''
        density = density // 60
        uic = list(InstrumentToUic(self._client, self._AccountKey, spec={'Instrument': ticker}).values())[0]
        params = {
                "AssetType": "FxSpot",
                "Horizon": density, # 1 muinte density (min 1 minute)
                "Count": amount_intervals, # how many historical intervals with the "Horizont" (max 1200)
                "Uic": uic
                }
        r = chart.charts.GetChartData(params=params)
        rv = self._client.request(r)
        return rv['Data']
    
    def portfolio_open_positions(self):
        '''
        return a dict with one pair "key: int" and with dicts.
        It looks like:
            'EURUSD': {
                'amount_long': 75000.0,
                'amount_short': 0.0,
                'type_asset': 'FxSpot',
                'uic': 21},
            'amount_positions': 6

        where:
            "amount_positions" is number positions in account
        '''

        r = pf.netpositions.NetPositionsMe(params={})
        self._client.request(r)
        rv = r.response
        dict_positions = {}
        for n in range(len(rv['Data'])):
            ticker_type_asset = rv['Data'][n]['NetPositionId'].split('__')
            potision_info = {'amount_long': rv['Data'][n]['NetPositionBase']['AmountLong'],
                            'amount_short': rv['Data'][n]['NetPositionBase']['AmountShort'],
                            'uic': rv['Data'][n]['NetPositionBase']['Uic'],
                            'type_asset': ticker_type_asset[1]}
            dict_positions[ticker_type_asset[0]] = potision_info
        dict_positions['amount_positions'] = rv['__count']
        return dict_positions
    
    def check_order(self, order_id):
        '''
        Checking order existing. If exist then return True, else False
        '''
        r = pf.orders.GetOpenOrder(ClientKey=self._ClientKey, OrderId=order_id, params={})
        self._client.request(r)
        rv = r.response['Data']
        if len(rv) == 0:
            return False
        else:
            return True

    def saxoToolKit(self):
        url = f"https://gateway.saxobank.com/sim/openapi/ref/v1/currencypairs/?AccountKey={self._AccountKey}&ClientKey={self._ClientKey}"
        print('saxoDEV', self._client.OWNREQUEST(method='get', url=url, request_args={}))

    def cancelOrder(self, orderId):
        url = f"https://gateway.saxobank.com/sim/openapi/trade/v2/orders/{orderId}/?AccountKey={self._AccountKey}"
        self._client.OWNREQUEST(method='delete', url=url, request_args={})

    def update_token(self):
        self._client = API(access_token=self._token)
        self._AccountKey = account_info(self._client).AccountKey
        self._ClientKey = account_info(self._client).ClientKey
