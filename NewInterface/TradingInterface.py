import warnings
import os
import sys
import inspect
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from RealTimeTrade.utils.connectorInterface import *
from RealTimeTrade.utils.statCollectorModule import *
from RealTimeTrade.utils.timerModule import *
from RealTimeTrade.utils.utils import *
from RealTimeTrade.utils.TelegramNotificator import *
from RealTimeTrade.strategiesPool.MeanReversionDualMode import *
from RealTimeTrade.utils.statCollectorModule import PandasStatCollector

import pandas as pd
import numpy as np
from typing import Union
import datetime
from saxo_openapi.exceptions import OpenAPIError
from NewInterface.structures.asset_structure import assetInformation

global DEBUG_MODE
global Update_log
global Error_log
global Open_error_log
global Open_position_log
global Close_position_log
global Close_error_log


class TradingInterface:
    @classmethod
    def _time_converter(cls, stringTime: str) -> datetime.datetime:
        if stringTime[-1] == 'Z':
            return datetime.datetime.fromisoformat(stringTime[:-1])
        if len(stringTime) == 19:
            return datetime.datetime.strptime(stringTime, '%Y-%m-%d %H:%M:%S')

    availableBrokerInterface = Union[None, SaxoOrderInterface, AbstractOrderInterface]
    availableNotificator = Union[None, TelegramNotification]
    availableStrategies = Union[None, MeanReversionDual, EmptyDebugStrategy]
    availableStatisticsCollectors = Union[None, PandasStatCollector]

    brokerInterface: availableBrokerInterface
    notificator: availableNotificator
    strategy: availableStrategies
    statistics_collector: availableStatisticsCollectors
    updatableToken: bool
    _cachedCollectedTime: Union[None, datetime.datetime]
    ordersType: Union[str]
    AvailableToOpen: bool

    def __init__(self, name: str, robotConfig: str, ticker: Union[List[str], str], requireTokenUpdate: bool = True):
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
        self.AvailableToOpen = True
        self.globalTimer = Timer()
        self.tradingTimer = Timer()
        self.debug = DEBUG_MODE
        self.name = name
        self.ticker = ticker

        self.InPosition = False
        # Config block
        self._configPath = robotConfig
        self._robotConfig = pd.read_csv(robotConfig, header=None).T
        self._robotConfig = pd.Series(data=self._robotConfig.iloc[1, :].values,
                                      index=self._robotConfig.iloc[0, :])
        self.updatableDataTime = int(self._robotConfig['updateDataTime'])
        self._token = str(self._robotConfig['apiToken'])
        self.ordersType = str(self._robotConfig['ordersType'])
        # Statistics and fast notificator block
        self.statistics_collector = None
        self.notificator = None
        # Updatable Token block
        self.updatableToken = requireTokenUpdate
        self.brokerInterface = None
        # Plugged strategy
        self.strategy = None
        # Saver block
        self.historical = list()
        if type(ticker) == list:
            self.historical = [assetInformation(ticker_name=ticker_name) for ticker_name in ticker]
        if type(ticker) == str:
            self.historical = [assetInformation(ticker_name=ticker)]
        # Utils block
        self._cachedCollectedTime = None

        # Inside settings
        self._refresherFreshDataTimer = 1
        self._refresherOrderTimer = 5

        # Inside config logic
        if self.updatableToken:
            self.lastTokenUpdate = self._time_converter(self._robotConfig['lastApiTokenUpdate'])
            self.tokenLife = int(self._robotConfig['apiTokenLifeTime'])
            self.threadWithTokenUpdatable = None

    @property
    def get_token(self):
        return self._token

    def add_strategy(self, strategy: availableStrategies):
        self.strategy = strategy

    def add_broker_interface(self, broker: availableBrokerInterface):
        self.brokerInterface = broker

    def add_statistics_collector(self, collector: availableStatisticsCollectors):
        self.statistics_collector = collector

    def add_fast_notificator(self, notificator: availableNotificator):
        self.notificator = notificator
        if self.updatableToken:
            self.threadWithTokenUpdatable = Thread(target=self.notificator.token_request_echo,
                                                   args=(self.notificator, self)).start()

    def download_history_data(self, density, lookBack: int) -> str:
        if (self.updatableDataTime // density != 1) or (self.updatableDataTime % density != 0):
            warnings.warn('TradingInterface updatable time miss-matched with history density request')

        if self.brokerInterface is not None:
            for basicAsset in self.historical:

                history = self.brokerInterface.get_asset_data_hist(ticker=basicAsset.Name, density=density,
                                                                   amount_intervals=lookBack)
                history = pd.DataFrame(history)
                history['Time'] = history['Time'].apply(lambda x: self._time_converter(x))
                basicAsset.CloseAsk = list(history['CloseAsk'].values)
                basicAsset.CloseBid = list(history['CloseBid'].values)
                basicAsset.HighAsk = list(history['HighAsk'].values)
                basicAsset.LowAsk = list(history['LowAsk'].values)
                basicAsset.LowBid = list(history['LowBid'].values)
                basicAsset.HighBid = list(history['HighBid'].values)
                basicAsset.OpenAsk = list(history['OpenAsk'].values)
                basicAsset.OpenBid = list(history['OpenBid'].values)
                basicAsset.OpenMiddle = list(history.apply(lambda x: (x['OpenBid'] + x['OpenAsk']) / 2, axis=1).values)
                basicAsset.CloseMiddle = list(history.apply(lambda x: (x['CloseBid'] + x['CloseAsk']) / 2, axis=1).values)
                basicAsset.LowMiddle = list(history.apply(lambda x: (x['LowBid'] + x['LowAsk']) / 2, axis=1).values)
                basicAsset.HighMiddle = list(history.apply(lambda x: (x['HighBid'] + x['HighAsk']) / 2, axis=1).values)
                basicAsset.Time = list(history['Time'].values)

                basicAsset._cachedCollectedTime = basicAsset.Time[-1]
                del history
                return f'{Update_log}Successfully downloaded last {lookBack} dotes with last time {basicAsset._cachedCollectedTime}'
            else:
                warnings.warn('No brokerInterface plugged')
                return f"{Error_log}No brokerInterface"

    def start_execution(self):
        self.globalTimer.start()
        historical = self.download_history_data(self.updatableDataTime,
                                                min(int(self.strategy.strategyParams['scanHalfTime']),
                                                    int(self._robotConfig['maxLookBack'])))
        if self.debug:
            print(self.historical[0].Name)

        # while True:
        #     self.search_for_trade()


if __name__ == '__main__':
    DEBUG_MODE = True
    Update_log = "LOG | UPDATE: "
    Error_log = "LOG | ERROR: "
    Open_error_log = "LOG | Cannot open: "
    Open_position_log = "LOG | OpenPosition: "
    Close_position_log = "LOG | ClosePosition: "
    Close_error_log = "LOG | Cannot close: "
    # initialize
    monkey = TradingInterface(name='monkey', robotConfig='robotConfig.txt', ticker='CHFJPY',
                              requireTokenUpdate=True)
    # add collector
    monkey.add_statistics_collector(PandasStatCollector(fileToSave='stat.csv', detailsPath='details.csv'))
    # add saxo interface
    monkey.add_broker_interface(SaxoOrderInterface(monkey.get_token))
    # add telegram notificator
    # monkey.add_fast_notificator(TelegramNotification())
    # add strategy rules
    monkey.add_strategy(MeanReversionDual(strategyConfigPath='strategiesPool/MeanReversionStrategyParameters.txt',
                                          strategyModePath='strategiesPool/DualMeanConfig.txt'))
    # monkey.add_strategy(EmptyDebugStrategy(strategyConfigPath='strategiesPool/MeanReversionStrategyParameters.txt',
    #                                        strategyModePath='strategiesPool/DualMeanConfig.txt'))
    # monkey.strategy.add_trading_interface(monkey)
    monkey.start_execution()
    # print(monkey.make_order(orderDetails={"position": 100_000, "openPrice": 134.425}, typePos="open", openDetails=None))


