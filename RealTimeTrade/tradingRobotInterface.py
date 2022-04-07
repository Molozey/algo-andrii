import pandas as pd
import numpy as np
import time
from utils import get_half_time, reverse_variance_ratio, variance_ratio, create_strategy_config


class ImRobot:
    def __init__(self, name, config_file_way, strategyParameters_file_way):
        """

        :param name: Robot Name
        :param config_file_way: Path to config txt
        :param strategyParameters_file_way: Path to strategy hyperparams file
        """
        self.name = name
        config = pd.read_csv(str(config_file_way), header=0, sep=' ')
        self.time_interval = float(config.iloc[list(config.index).index('updateTime: '), 0])
        del config

        self._initStrategyParams = pd.read_csv(strategyParameters_file_way, header=0, sep=',')
        self.strategyParams = create_strategy_config(self._initStrategyParams)

        self.connector = None
        self.statCollector = None
        self.timer = None
        self.tradingTimer = None

        self._inPosition = False

        self._PastPricesArray = list()

        self.tradeCapital = 10_000

    def _collect_past_prices(self):
        # We need at least self._initStrategyParams.scanHalfTime
        self._PastPricesArray = self.connector.collect_past_multiple_prices(self._initStrategyParams.scanHalfTime)
        pass

    def _collect_new_price(self):
        newPrice = self.connector.get_actual_data()
        self._PastPricesArray.append(newPrice)

    def add_timer(self, timer, tradingTimer):
        self.timer = timer
        self.tradingTimer = tradingTimer

    def add_statistics_collector(self, collector):
        self.statCollector = collector

    def add_connector(self, connector):
        self.connector = connector

    def _open_trade_ability(self):
        openDict = {
            'typeOperation': None,
            'position': None,
            'openPrice': None,
            'openIndex': None,
            'stopLossBorder': None,
            'takeProfitBorder': None
        }
        half_time = int(get_half_time(self._PastPricesArray[-int(self.strategyParams.scanHalfTime):]))
        if (half_time > self.strategyParams['scanHalfTime']) or (half_time < 0):
            return False
        self.strategyParams["rollingMean"] = int(half_time * self.strategyParams['halfToLight'])
        self.strategyParams["fatRollingMean"] = int(self.strategyParams['halfToFat'] * half_time)
        self.strategyParams["timeBarrier"] = int(half_time * self.strategyParams['halfToTime'])
        if self.strategyParams["timeBarrier"] <= 0:
            self.strategyParams["timeBarrier"] = 1

        self.strategyParams["varianceLookBack"] = int(half_time * self.strategyParams['halfToFat'])
        self.strategyParams["varianceRatioCarrete"] = int((half_time *
                                                           self.strategyParams['halfToFat']) // self.strategyParams['varianceRatioCarreteParameter']) + 1

        workingArray = self._PastPricesArray[-int(self.strategyParams.scanHalfTime):]
        bandMean = np.mean(workingArray)
        bandStd = np.std(workingArray)

        lowBand = round(bandMean - bandStd * self.strategyParams['yThreshold'], 3)
        highBand = round(bandMean + bandStd * self.strategyParams['yThreshold'], 3)

        if workingArray[-1] < lowBand:
            logTuple = self._PastPricesArray[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)

            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict['typeOperation'] = 'BUY'
                openDict['position'] = round(self.tradeCapital / lowBand, 3)
                openDict['openPrice'] = lowBand
                openDict['openTime'] = self.tradingTimer.elapsed()
                openDict['stopLossBorder'] = round(lowBand - self.strategyParams['stopLossStdMultiplier'] * bandStd, 3)
                openDict['takeProfitBorder'] = round(lowBand +
                                                     self.strategyParams['takeProfitStdMultiplier'] * bandStd, 3)

                return openDict

        if workingArray[-1] > highBand:
            logTuple = self._PastPricesArray[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)

            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict['typeOperation'] = 'SELL'
                openDict['position'] = -1 * round(self.tradeCapital / highBand, 3)
                openDict['openPrice'] = highBand
                openDict['openTime'] = self.tradingTimer.elapsed()
                openDict['stopLossBorder'] = round(highBand - self.strategyParams['stopLossStdMultiplier'] * bandStd, 3)
                openDict['takeProfitBorder'] = round(highBand +
                                                     self.strategyParams['takeProfitStdMultiplier'] * bandStd, 3)

                return openDict

        return False

    def _close_trade_ability(self):
        pass

    def _trading_loop(self):
        # Waiting until we can open a trade
        while not self._inPosition:
            self._collect_new_price()
            openAbility = self._open_trade_ability()
            if isinstance(openAbility, dict):
                self.connector.place_open_order()
                self._inPosition = True
                self.tradingTimer.start()
            time.sleep(self.time_interval)

        while self._inPosition:
            self._collect_new_price()
            closeAbility = self._close_trade_ability()
            if isinstance(closeAbility, dict):
                self.connector.place_close_order()
                self._inPosition = False
                self.tradingTimer.stop()
            time.sleep(self.time_interval)

        _stat = {**openAbility, **closeAbility}
        _stat['StrategyWorkingTime'] = self.timer.elapsed()
        self.statCollector.add_trade_line(_stat)


    def start_tradingCycle(self):
        if (self.timer is None) or (self.tradingTimer is None):
            raise ModuleNotFoundError('Timer not plugged')
        if self.connector is None:
            raise ModuleNotFoundError('Connector not plugged')
        if self.statCollector is None:
            print('Warning no statistic collector plugged')
        pass

        self.timer.start()
        while True:
            self.strategyParams = create_strategy_config(self._initStrategyParams)
            self._trading_loop()
