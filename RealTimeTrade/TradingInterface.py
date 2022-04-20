import warnings
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from RealTimeTrade.utils.connectorInterface import *
from RealTimeTrade.utils.statCollectorModule import *
from RealTimeTrade.utils.timerModule import *
from RealTimeTrade.utils.utils import *
from RealTimeTrade.utils.TelegramNotificator import *
from RealTimeTrade.strategiesPool.MeanReversionDualMode import *

import pandas as pd
import numpy as np
from typing import Union
import datetime
from saxo_openapi.exceptions import OpenAPIError

global DEBUG_MODE
global Update_log
global Error_log
global Open_log

class TradingInterface:
    @classmethod
    def _time_converter(cls, stringTime):
        if stringTime[-1] == 'Z':
            return datetime.datetime.fromisoformat(stringTime[:-1])
        if len(stringTime) == 19:
            return datetime.datetime.strptime(stringTime, '%Y-%m-%d %H:%M:%S')

    availableBrokerInterface = Union[None, SaxoOrderInterface, AbstractOrderInterface]
    availableNotificator = Union[None, TelegramNotification]
    availableStrategies = Union[None, MeanReversionDual]
    availableStatisticsCollectors = Union[None, PandasStatCollector]

    brokerInterface: availableBrokerInterface
    notificator: availableNotificator
    strategy: availableStrategies
    statistics_collector: availableStatisticsCollectors
    updatableToken: bool
    _cachedCollectedTime: Union[None, datetime.datetime]

    def __init__(self, name: str, robotConfig: str,
                 ticker: str, requireTokenUpdate: bool = True):
        """
        Initialize Interface
        :param name: Robot's Name
        :param robotConfig: txt with next structure
        ...
        updateDataTime,60
        apiToken,eyJhbGciOiJFUzI1NiIsIng1dCI6IkRFNDc0QUQ1Q0NGRUFFRTlDRThCRDQ3ODlFRTZDOTEyRjVCM0UzOTQifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiNGFOUTNBSE41TnRDUmJ8UjNkNy1hdz09IiwiY2lkIjoiNGFOUTNBSE41TnRDUmJ8UjNkNy1hdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiYmZiZGM1OTE2ZDE5NDcwYTg1OTY1NDg5MTJiZGU1MWYiLCJkZ2kiOiI4NCIsImV4cCI6IjE2NTA0MTEzOTAiLCJvYWwiOiIxRiJ9.6usTiTNKk-YJTM9wenTKOV_NbeRngbouPCuzgPfW9drpGZUFeoywcybfS1_q0vVhH0m1QM4rFSsYKLFc5OTm2w
        apiTokenLifeTime,86400
        ...
        :param ticker: which instrument we trade?
        :param requireTokenUpdate should token for broker Interface be updatable
        """
        # Robot name and strategy ticker block
        self.globalTimer = None
        self.debug = DEBUG_MODE
        self.name = name
        self.ticker = ticker
        # Config block
        self._configPath = robotConfig
        self._robotConfig = pd.read_csv(robotConfig, header=None).T
        self._robotConfig = pd.Series(data=self._robotConfig.iloc[1, :].values,
                                      index=self._robotConfig.iloc[0, :])
        self.updatableDataTime = int(self._robotConfig['updateDataTime'])
        self._token = str(self._robotConfig['apiToken'])
        # Statistics and fast notificator block
        self.statistics_collector = None
        self.notificator = None
        # Updatable Token block
        self.updatableToken = requireTokenUpdate
        self.brokerInterface = None
        # Plugged strategy
        self.strategy = None
        # Saver block
        self.CloseAsk = list()
        self.CloseBid = list()
        self.HighAsk = list()
        self.LowAsk = list()
        self.LowBid = list()
        self.HighBid = list()
        self.OpenAsk = list()
        self.OpenBid = list()
        self.OpenMiddle = list()
        self.CloseMiddle = list()
        self.LowMiddle = list()
        self.HighMiddle = list()
        self.Time = list()
        # Utils block
        self._cachedCollectedTime = None

        # Inside settings
        self._refresherFreshDataTimer = 1

        # Inside config logic
        if self.updatableToken:
            self.lastTokenUpdate = self._time_converter(self._robotConfig['lastApiTokenUpdate'])
            self.tokenLife = int(self._robotConfig['apiTokenLifeTime'])
            self.threadWithTokenUpdatable = None

    @property
    def get_token(self):
        return self._token

    def add_broker_interface(self, broker: availableBrokerInterface):
        self.brokerInterface = broker

    def add_statistics_collector(self, collector: availableStatisticsCollectors):
        self.statistics_collector = collector

    def add_fast_notificator(self, notificator: availableNotificator):
        self.notificator = notificator
        if self.updatableToken:
            self.threadWithTokenUpdatable = Thread(target=self.notificator.token_request_echo,
                                                   args=(self.notificator, self)).start()

    def update_token_information(self, token: str, updateDate: datetime.datetime):
        self._robotConfig['apiToken'] = token
        self._robotConfig['lastApiTokenUpdate'] = updateDate
        self._robotConfig.to_csv(self._configPath, header=None)
        self.brokerInterface._token = token

    def add_strategy(self, strategy: availableStrategies):
        self.strategy = strategy

    def download_history_data(self, density, lookBack: int) -> str:
        if (self.updatableDataTime // density != 1) or (self.updatableDataTime % density != 0):
            warnings.warn('TradingInterface updatable time miss-matched with history density request')

        if self.brokerInterface is not None:
            history = self.brokerInterface.get_asset_data_hist(ticker=self.ticker, density=density,
                                                               amount_intervals=lookBack)
            history = pd.DataFrame(history)
            history['Time'] = history['Time'].apply(lambda x: self._time_converter(x))

            self.CloseAsk = list(history['CloseAsk'].values)
            self.CloseBid = list(history['CloseBid'].values)
            self.HighAsk = list(history['HighAsk'].values)
            self.LowAsk = list(history['LowAsk'].values)
            self.LowBid = list(history['LowBid'].values)
            self.HighBid = list(history['HighBid'].values)
            self.OpenAsk = list(history['OpenAsk'].values)
            self.OpenBid = list(history['OpenBid'].values)
            self.OpenMiddle = list(history.apply(lambda x: (x['OpenBid'] + x['OpenAsk']) / 2, axis=1).values)
            self.CloseMiddle = list(history.apply(lambda x: (x['CloseBid'] + x['CloseAsk']) / 2, axis=1).values)
            self.LowMiddle = list(history.apply(lambda x: (x['LowBid'] + x['LowAsk']) / 2, axis=1).values)
            self.HighMiddle = list(history.apply(lambda x: (x['HighBid'] + x['HighAsk']) / 2, axis=1).values)
            self.Time = list(history['Time'].values)

            self._cachedCollectedTime = self.Time[-1]
            del history
            return f'{Update_log}Successfully downloaded last {lookBack} dotes with last time {self._cachedCollectedTime}'
        else:
            warnings.warn('No brokerInterface plugged')
            return f"{Error_log}No brokerInterface"

    def download_actual_dot(self, density, lookBack: int = 1) -> str:
        if (self.updatableDataTime // density != 1) or (self.updatableDataTime % density != 0):
            warnings.warn('TradingInterface updatable time miss-matched with history density request')

        if self.brokerInterface is not None:
            _cachedCollectedTime = self._cachedCollectedTime
            while True:
                try:
                    history = self.brokerInterface.get_asset_data_hist(ticker=self.ticker, density=density,
                                                                       amount_intervals=lookBack)
                    history = pd.DataFrame(history)
                    history['Time'] = history['Time'].apply(lambda x: self._time_converter(x))
                    _cachedCollectedTime = history['Time'].values[0]
                    if abs((_cachedCollectedTime - self._cachedCollectedTime) / np.timedelta64(1, 's')) >= self.updatableDataTime:
                        break
                    time.sleep(self._refresherFreshDataTimer)
                # Exception list of api requests limit need to be expand while adding new broker providers
                except OpenAPIError:
                    if self.notificator is not None:
                        self.notificator.send_message_to_user(message="FATAL: So many requests per minute")
                        raise ConnectionError('Many Requests')
                    else:
                        raise ConnectionError('Many Requests')

            self.CloseAsk.append(history['CloseAsk'].values[0])
            self.CloseBid.append(history['CloseBid'].values[0])
            self.HighAsk.append(history['HighAsk'].values[0])
            self.LowAsk.append(history['LowAsk'].values[0])
            self.LowBid.append(history['LowBid'].values[0])
            self.HighBid.append(history['HighBid'].values[0])
            self.OpenAsk.append(history['OpenAsk'].values[0])
            self.OpenBid.append(history['OpenBid'].values[0])
            self.Time.append(list(history['Time'].values)[0])
            self.OpenMiddle.append(history.apply(lambda x: (x['OpenBid'] + x['OpenAsk']) / 2, axis=1).values[0])
            self.CloseMiddle.append(history.apply(lambda x: (x['CloseBid'] + x['CloseAsk']) / 2, axis=1).values[0])
            self.LowMiddle.append(history.apply(lambda x: (x['LowBid'] + x['LowAsk']) / 2, axis=1).values[0])
            self.HighMiddle.append(history.apply(lambda x: (x['HighBid'] + x['HighAsk']) / 2, axis=1).values[0])
            self._cachedCollectedTime = self.Time[-1]

            del _cachedCollectedTime, history
            return f'{Update_log}Successfully downloaded fresh dot with time {self.Time[-1]}'
        else:
            raise ModuleNotFoundError('No brokerInterface plugged')

    def search_for_open(self):
        answer = None
        freshData = self.download_actual_dot(self.updatableDataTime)
        if self.debug:
            print(freshData)
        while not isinstance(answer, dict):
            if self.debug:
                print(f"{Open_log}{answer}")
            answer = self.strategy.open_trade_ability()
            freshData = self.download_actual_dot(self.updatableDataTime)
            if self.debug:
                print(freshData)

        print(answer)

    def start_execution(self):
        historical = self.download_history_data(self.updatableDataTime,
                                                min(int(self.strategy.strategyParams['scanHalfTime']),
                                                    int(self._robotConfig['maxLookBack'])))
        if self.debug:
            print(historical)

        self.search_for_open()


if __name__ == '__main__':
    DEBUG_MODE = True
    Update_log = "LOG | UPDATE: "
    Error_log = "LOG | ERROR: "
    Open_log = "LOG | Cannot open: "

    # initialize
    monkey = TradingInterface(name='monkey', robotConfig='robotConfig.txt', ticker='CHFJPY',
                              requireTokenUpdate=True)
    # add saxo interface
    monkey.add_broker_interface(SaxoOrderInterface(monkey.get_token))
    # add telegram notificator
    monkey.add_fast_notificator(TelegramNotification())
    # add strategy rules
    monkey.add_strategy(MeanReversionDual(strategyConfigPath='strategiesPool/MeanReversionStrategyParameters.txt',
                                          strategyModePath='strategiesPool/DualMeanConfig.txt'))

    monkey.strategy.add_trading_interface(monkey)
    monkey.start_execution()

