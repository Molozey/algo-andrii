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
        self._tradeCapital = 10_000
        config = pd.read_csv(str(config_file_way), header=None, index_col=0, sep=',')
        self.time_interval = float(config.loc['updateTime'])
        del config

        self._initStrategyParams = pd.read_csv(strategyParameters_file_way, header=None).T
        self._initStrategyParams = pd.Series(data=self._initStrategyParams.iloc[1, :].values,
                                             index=self._initStrategyParams.iloc[0, :])

        self.strategyParams = create_strategy_config(self._initStrategyParams, CAP=self._tradeCapital)

        self.connector = None
        self.statCollector = None
        self.timer = None
        self.tradingTimer = None

        self._inPosition = False
        self._positionDetails = None
        self.waitingToFatMean = False

        self._PastPricesArray = list()


    def _collect_past_prices(self):
        # We need at least self._initStrategyParams.scanHalfTime
        self._PastPricesArray = self.connector.collect_past_multiple_prices(int(self._initStrategyParams['scanHalfTime']))
        print(f'Successfully downloaded last {self.strategyParams["scanHalfTime"]} dotes')

    def _collect_new_price(self):
        newPrice = self.connector.get_actual_data()
        self._PastPricesArray.append(newPrice)

    def add_timer(self, timer, tradingTimer):
        self.timer = timer
        self.tradingTimer = tradingTimer

    def add_statistics_collector(self, collector):
        self.statCollector = collector

    def add_connector(self, connectorInterface):
        self.connector = connectorInterface

    def _open_trade_ability(self):
        openDict = {
            'typeOperation': None,
            'position': None,
            'openPrice': None,
            'openIndex': None,
            'stopLossBorder': None,
            'takeProfitBorder': None
        }
        half_time = int(get_half_time(pd.Series(self._PastPricesArray[-int(self.strategyParams['scanHalfTime']):])))
        if (half_time > self.strategyParams['scanHalfTime']) or (half_time < 0):
            return False
        self.strategyParams["rollingMean"] = int(half_time * self.strategyParams['halfToLight'])
        self.strategyParams["fatRollingMean"] = int(self.strategyParams['halfToFat'] * half_time)
        self.strategyParams["timeBarrier"] = int(half_time * self.strategyParams['halfToTime'])
        if self.strategyParams["timeBarrier"] <= 0:
            self.strategyParams["timeBarrier"] = 1

        self.strategyParams["varianceLookBack"] = int(half_time * self.strategyParams['halfToFat'])
        self.strategyParams["varianceRatioCarrete"] = int((half_time *
                                                           self.strategyParams['halfToFat']) //
                                                          self.strategyParams['varianceRatioCarreteParameter']) + 1

        workingArray = self._PastPricesArray[-int(self.strategyParams['scanHalfTime']):]
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

        if (self.tradingTimer.elapsed() // 60) > self.strategyParams['timeBarrier']:
            return {'typeHolding': 'endPeriod', 'closePrice': None,
                    'closeIndex': None}

        if self._positionDetails['typeOperation'] == 'BUY':
            if self._PastPricesArray[-1] < self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArray[-1]}
            # Block with Trailing StopLoss. This realization is not good. Need to change
            delta = self._PastPricesArray[-1] - self._PastPricesArray[-2]
            if delta > 0:
                self._positionDetails['stopLossBorder'] = round(self._positionDetails['stopLossBorder'] + delta, 3)

            if self._PastPricesArray[-1] < self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArray[-1]}

            if not self.waitingToFatMean:
                workingArray = self._PastPricesArray[-int(self.strategyParams['rollingMean']):]
                bandMean = np.mean(workingArray)
                MeanFat = np.mean(self._PastPricesArray[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArray[-1] > bandMean:
                    _log = self._PastPricesArray[-(int(max(self.strategyParams['varianceLookBack'], self.strategyParams['fatRollingMean']))+1):]
                    compute = {
                        "retOpenPrice": np.diff(_log),
                        "logOpenPrice": _log[1:]
                    }
                    assert len(compute['retOpenPrice']) == len(compute['logOpenPrice'])
                    if MeanFat > bandMean:
                        if reverse_variance_ratio(preComputed=compute, params=self.strategyParams, timeBorderCounter=self.tradingTimer.elapsed() // 60, VRstatement=self.waitingToFatMean):
                            self.waitingToFatMean = True
                            return False
                        else:
                            return {'typeHolding': 'lightCross', 'closePrice': bandMean}
                    else:
                        return {'typeHolding': 'lightCrossEmergent', 'closePrice': bandMean}

            if self.waitingToFatMean:
                MeanFat = np.mean(self._PastPricesArray[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArray[-1] > MeanFat:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': MeanFat}

                _log = self._PastPricesArray[
                       -(int(max(self.strategyParams['varianceLookBack'], self.strategyParams['fatRollingMean'])) + 1):]
                compute = {
                    "retOpenPrice": np.diff(_log),
                    "logOpenPrice": _log[1:]
                }
                if not reverse_variance_ratio(preComputed=compute, params=self.strategyParams, timeBorderCounter=self.tradingTimer.elapsed() // 60, VRstatement=self.waitingToFatMean):
                    self.waitingToFatMean = False
                    return False

        if self._positionDetails['typeOperation'] == 'SELL':
            if self._PastPricesArray[-1] > self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArray[-1]}
            # Block with Trailing StopLoss. This realization is not good. Need to change
            delta = self._PastPricesArray[-1] - self._PastPricesArray[-2]
            if delta < 0:
                self._positionDetails['stopLossBorder'] = round(self._positionDetails['stopLossBorder'] - delta, 3)

            if self._PastPricesArray[-1] > self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArray[-1]}

            if not self.waitingToFatMean:
                workingArray = self._PastPricesArray[-int(self.strategyParams['rollingMean']):]
                bandMean = np.mean(workingArray)
                MeanFat = np.mean(self._PastPricesArray[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArray[-1] < bandMean:
                    _log = self._PastPricesArray[-(int(
                        max(self.strategyParams['varianceLookBack'], self.strategyParams['fatRollingMean'])) + 1):]
                    compute = {
                        "retOpenPrice": np.diff(_log),
                        "logOpenPrice": _log[1:]
                    }
                    assert len(compute['retOpenPrice']) == len(compute['logOpenPrice'])
                    if MeanFat < bandMean:
                        if reverse_variance_ratio(preComputed=compute, params=self.strategyParams,
                                                  timeBorderCounter=self.tradingTimer.elapsed() // 60,
                                                  VRstatement=self.waitingToFatMean):
                            self.waitingToFatMean = True
                            return False
                        else:
                            return {'typeHolding': 'lightCross', 'closePrice': bandMean}
                    else:
                        return {'typeHolding': 'lightCrossEmergent', 'closePrice': bandMean}

            if self.waitingToFatMean:
                MeanFat = np.mean(self._PastPricesArray[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArray[-1] < MeanFat:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': MeanFat}

                _log = self._PastPricesArray[
                       -(int(max(self.strategyParams['varianceLookBack'], self.strategyParams['fatRollingMean'])) + 1):]
                compute = {
                    "retOpenPrice": np.diff(_log),
                    "logOpenPrice": _log[1:]
                }
                if not reverse_variance_ratio(preComputed=compute, params=self.strategyParams,
                                              timeBorderCounter=self.tradingTimer.elapsed() // 60,
                                              VRstatement=self.waitingToFatMean):
                    self.waitingToFatMean = False
                    return False

        return False

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

        self._positionDetails = openAbility

        self.waitingToFatMean = False
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
        self._collect_past_prices()
        if (self.timer is None) or (self.tradingTimer is None):
            raise ModuleNotFoundError('Timer not plugged')
        if self.connector is None:
            raise ModuleNotFoundError('Connector not plugged')
        if self.statCollector is None:
            print('Warning no statistic collector plugged')
        pass

        self.timer.start()
        while True:
            self.strategyParams = create_strategy_config(self._initStrategyParams, CAP=self._tradeCapital)
            self._trading_loop()


from timerModule import Timer
from statCollectorModule import PandasStatCollector
from historicalSimulateCollector import *

monkeyRobot = ImRobot('MNKY', config_file_way="robotConfig.txt", strategyParameters_file_way="strategyParameters.txt")

timerGlobal = Timer()
timerTrade = Timer()

connector = SimulatedOrderGenerator("dataForGenerator.csv")

pandasCollector = PandasStatCollector("stat.csv")


monkeyRobot.add_timer(timerGlobal, timerTrade)
monkeyRobot.add_statistics_collector(pandasCollector)
monkeyRobot.add_connector(connector)

monkeyRobot.start_tradingCycle()
