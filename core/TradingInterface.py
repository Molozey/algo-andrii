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
        # Robot information
        self.globalTimer = Timer()
        self.tradingTimer = Timer()
        self.debug = debug
        self.name = name

        # Config block
        self._configPath = robotConfig
        self._robotConfig = pd.read_csv(robotConfig, header=None).T
        self._robotConfig = pd.Series(data=self._robotConfig.iloc[1, :].values,
                                      index=self._robotConfig.iloc[0, :])
        self._token = str(self._robotConfig['apiToken'])

        # Statistics and fast notificator block
        self.statistics_collector = None
        self.notificator = None
        # Updatable Token block
        self.updatableToken = requireTokenUpdate
        self.brokerInterface = None
        # Plugged strategy
        self.strategy = None

        # Market data storage
        self.assets = list()
        # Threads of assetsUpdatable
        self.assetsThreads = list()

        # Initialize of assets classes
        for ticker in assets.items():
            self.assets.append(assetInformation(ticker_name=ticker[0], ticker_parameters=ticker[1]))

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

    def start_execution(self):
        if not self.strategy:
            raise Exception('No strategy plugged!')

        for asset in self.assets:
            asset.add_interface(self)
            if issubclass(asset.Supplier, self.brokerInterface.__class__):
                asset.Supplier = self.brokerInterface
            else:
                # TODO: How to create dataProvider object with only information of its type.
                # One of idea is to ping user by input request. Input == parameters of provider
                asset.Supplier = asset.Supplier()
        for asset in self.assets:
            self.assetsThreads.append(Thread(target=asset.start_cycle).start())

        while True:
            self.globalTimer.start()
            self.search_for_trade()

    def search_for_trade(self):
        while not self.InPosition:
            answer = None
            while not isinstance(answer, dict):
                answer = self.strategy.open_trade_ability()
                if not isinstance(answer, dict):
                    # TODO: wait on frequency
                    pass
            if self.debug:
                print(f'Try {Open_position_log}{answer}')

            openOrderInfo = self.make_order(orderDetails=answer, typePos='open', openDetails=None)
            if Open_error_log not in openOrderInfo:
                if self.debug:
                    print(f'{Open_position_log} Success with open trade at execution time: {self.globalTimer.elapsed()}')
                self.InPosition = True
                self.tradingTimer.start()

        if self.notificator is not None:
            self.notificator.send_message_to_user(f"Opening:\n{json.dumps(answer)}")

        # TODO fix multi-opening
        while self.InPosition:
            answerHold = None
            freshData = self.download_actual_dot(self.updatableDataTime)
            if self.debug:
                print(freshData)
            while not isinstance(answerHold, dict):
                answerHold = self.strategy.close_trade_ability(openDetails=answer)
                if not isinstance(answerHold, dict):
                    freshData = self.download_actual_dot(self.updatableDataTime)
                    if self.debug:
                        print(f"{freshData}")
            if self.debug:
                print(f"{Close_position_log}{answerHold}")

            closeOrderInfo = self.make_order(orderDetails=answerHold, typePos='close', openDetails=answer)
            if Close_error_log not in closeOrderInfo:
                if self.debug:
                    print(f"{Close_position_log} Success with close trade at execution time: {self.globalTimer.elapsed()}")
                self.InPosition = False
                self.tradingTimer.stop()

        TradeDetails = {**answer, **answerHold}

        if answer['typeOperation'] == 'BUY':
            TradeDetails['pct_change'] = (TradeDetails['closePrice'] - TradeDetails['openPrice'])\
                                         / TradeDetails['openPrice']
        if answer['typeOperation'] == 'SELL':
            TradeDetails['pct_change'] = (TradeDetails['openPrice'] - TradeDetails['closePrice'])\
                                         / TradeDetails['openPrice']

        if self.notificator is not None:
            if TradeDetails['pct_change'] > 0:
                self.notificator.send_message_to_user(f"✅ Close:\n{json.dumps(TradeDetails)}")
            else:
                self.notificator.send_message_to_user(f"🔴 Close:\n{json.dumps(TradeDetails)}")

        if self.statistics_collector is not None:
            self.statistics_collector.add_trade_line(TradeDetails)


