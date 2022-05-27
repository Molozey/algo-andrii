import warnings
import os
import sys
import inspect
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core.connectors.SAXO import *
from core.utils.timerModule import *
from core.utils.TelegramNotificator import *
from core.statCollectors.pandasSaver import PandasStatCollector
from core.structures.asset_structure import assetInformation
from core.strategiesPool.emptyStrategy.EmptyStrategy import *
from core.strategiesPool.meanReversionPrediction.MeanReversionPredict import *
from core.warnings.loggingPresets import *

import pandas as pd
import numpy as np
from typing import Union, Dict, List
import datetime
from saxo_openapi.exceptions import OpenAPIError

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
    availableStrategies = Union[None, EmptyStratergy, MeanReversionLightWithPrediction]
    availableStatisticsCollectors = Union[None, PandasStatCollector]

    brokerInterface: availableBrokerInterface
    notificator: availableNotificator
    strategy: availableStrategies
    statistics_collector: availableStatisticsCollectors
    updatableToken: bool
    _cachedCollectedTime: Union[None, datetime.datetime]
    ordersType: Union[str]
    AvailableToOpen: bool

    def __init__(self, name: str, robotConfig: str, assets: Dict,
                 requireTokenUpdate: bool = True, debug: bool = False):
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
        self.debug = debug
        self.name = name
        self.ticker = assets

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
        self.assets = list()
        for ticker in assets.items():
            self.assets.append(assetInformation(ticker_name=ticker[0], ticker_parameters=ticker[1]))
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

        for basicAsset in self.assets:
            if basicAsset.Supplier is not None:
                history = basicAsset.Supplier.get_asset_data_hist(ticker=basicAsset.Name, density=density,
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
                print(f'{Update_log}Successfully downloaded last {lookBack} dotes for {basicAsset.Name} with last time {basicAsset._cachedCollectedTime}')
            else:
                warnings.warn('No brokerInterface plugged')
                return f"{Error_log}No brokerInterface"

    def start_execution(self):
        self.globalTimer.start()
        historical = self.download_history_data(self.updatableDataTime,
                                                min(int(self.strategy.['scanHalfTime']),
                                                    int(self._robotConfig['maxLookBack'])))
        if self.debug:
            print([asset.Name for asset in self.assets])

        # while True:
        #     self.search_for_trade()


