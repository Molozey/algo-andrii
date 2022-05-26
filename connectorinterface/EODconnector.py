from abc import ABC, abstractmethod
from pprint import pprint
import time
import json
import datetime
import urllib
import requests

class AbstractDataInterface:
    def __init__(self):
        pass

    @abstractmethod
    def get_asset_data_hist(self, *args):
        pass

class EodDataInterface(AbstractDataInterface):
    def __init__(self, token):
        self._token = token
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
