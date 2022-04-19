import pandas as pd
import numpy as np
import time

import datetime
import argparse


from utils.utils import get_half_time, reverse_variance_ratio, variance_ratio, create_strategy_config
from utils.timerModule import Timer
from utils.statCollectorModule import PandasStatCollector
from utils.historicalSimulateCollector import *
from utils.connectorInterface import SaxoOrderInterface

global DEBUG

class ImRobot:
    def __init__(self, name, config_file_way, strategyParameters_file_way, tickerSaxo, tickerEOD):
        """

        :param name: Robot Name
        :param config_file_way: Path to config txt
        :param strategyParameters_file_way: Path to strategy hyperparams file
        """
        self.timerToken = None

        self.name = name
        self._tradeCapital = 100_000
        try:
            config = pd.read_csv(config_file_way, header=None).T
            config = pd.Series(data=config.iloc[1, :].values,
                                          index=config.iloc[0, :])
            self.time_interval = int(config['updateDataTime'])
            self.crossingMaxTime = int(config['waitingParameter'])
            del config
        except KeyError:
            print('Unknown keys in config')

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

        self._PastPricesArrayAsk = list()
        self._PastPricesArrayBid = list()
        self._PastPricesArrayMiddle = list()
        self.SAXO = tickerSaxo
        self.EOD = tickerEOD

        self._introWaitingTimer = None
        self._mode = "multiCrossing"
        # self._mode = "singleCrossing"
        self._BBandsMode = "Ask&Bid"
        # self._BBandsMode="OnlyOne"
        print("Using mode of searching in POINT:", self._mode)
        print("Using BBandsMode", self._BBandsMode)

    def add_timer(self, timer, tradingTimer):
        self.timer = timer
        self.tradingTimer = tradingTimer

    def add_statistics_collector(self, collector):
        self.statCollector = collector

    def add_connector(self, connectorInterface):
        self.connector = connectorInterface

    def _collect_past_prices(self):
        # We need at least self._initStrategyParams.scanHalfTime
        historicalData = self.connector.get_asset_data_hist(self.SAXO, 60,
                                                            int(min(self._initStrategyParams["scanHalfTime"], 1200)))
        historicalData = pd.DataFrame(historicalData)
        self._PastPricesArrayAsk = list(historicalData.apply(lambda x: x["OpenAsk"], axis=1).values)
        self._PastPricesArrayBid = list(historicalData.apply(lambda x: x["OpenBid"], axis=1).values)
        self._PastPricesArrayMiddle = list(historicalData.apply(lambda x: (x["OpenBid"] + x["OpenAsk"]) / 2, axis=1).values)
        # self._PastPricesArray = pd.read_csv('TESTINGprices.csv')
        # self._PastPricesArray = list(self._PastPricesArray.open.values)
        print(f'Successfully downloaded last {self.strategyParams["scanHalfTime"]} dotes')
        if DEBUG:
            print(f"Last time in historical:", historicalData.iloc[-1]['Time'])

    def _collect_new_price(self):
        newPrice = self.connector.get_asset_data_hist(self.SAXO, 60, 1)
        newPrice = pd.DataFrame(newPrice)
        self._PastPricesArrayAsk.append(newPrice.apply(lambda x: x["OpenAsk"], axis=1).values[0])
        self._PastPricesArrayBid.append(newPrice.apply(lambda x: x["OpenBid"], axis=1).values[0])
        self._PastPricesArrayMiddle.append(newPrice.apply(lambda x: (x["OpenBid"] + x["OpenAsk"]) / 2, axis=1).values[0])
        if DEBUG:
            print('Downloaded past price at time:', newPrice['Time'].values[0])

    def _calculate_bands(self):
        # We calculate meanReversionHalftime by middlePrice
        if self._BBandsMode == 'Ask&Bid':
            half_time = int(
                get_half_time(pd.Series(self._PastPricesArrayMiddle[-int(self.strategyParams['scanHalfTime']):])))

            #
            # half_time = 80
            #

            self.strategyParams["rollingMean"] = int(half_time * self.strategyParams['halfToLight'])
            self.strategyParams["fatRollingMean"] = int(self.strategyParams['halfToFat'] * half_time)
            self.strategyParams["timeBarrier"] = int(half_time * self.strategyParams['halfToTime'])
            #
            self.strategyParams["timeBarrier"] = 2
            #
            if self.strategyParams["timeBarrier"] <= 0:
                self.strategyParams["timeBarrier"] = 1

            self.strategyParams["varianceLookBack"] = int(half_time * self.strategyParams['halfToFat'])
            self.strategyParams["varianceRatioCarrete"] = int((half_time *
                                                               self.strategyParams['halfToFat']) //
                                                              self.strategyParams['varianceRatioCarreteParameter']) + 1

            if (half_time > self.strategyParams['scanHalfTime']) or (half_time < 2):
                if DEBUG:
                    print('Wrong HalfTime: ', half_time)
                return False

            workingAsk = self._PastPricesArrayAsk[-self.strategyParams["rollingMean"]:]
            workingBid = self._PastPricesArrayBid[-self.strategyParams["rollingMean"]:]

            meanAsk = np.mean(workingAsk)
            meanBid = np.mean(workingBid)

            stdAsk = np.std(workingAsk)
            stdBid = np.std(workingBid)

            lowAskBand = round(meanAsk - stdAsk * self.strategyParams['yThreshold'], 3)
            highAskBand = round(meanAsk + stdAsk * self.strategyParams['yThreshold'], 3)

            lowBidBand = round(meanBid - stdBid * self.strategyParams['yThreshold'], 3)
            highBidBand = round(meanBid + stdBid * self.strategyParams['yThreshold'], 3)

            dictRet = {'AskStd': stdAsk, 'BidStd': stdBid, 'lowBid': lowBidBand, 'highBid': highBidBand, 'lowAsk': lowAskBand, 'highAsk': highAskBand, 'halfTime': half_time}

            return dictRet

        if self._mode == 'OnlyOne':
            half_time = int(
                get_half_time(pd.Series(self._PastPricesArrayMiddle[-int(self.strategyParams['scanHalfTime']):])))

            #
            # half_time = 80
            #

            self.strategyParams["rollingMean"] = int(half_time * self.strategyParams['halfToLight'])
            self.strategyParams["fatRollingMean"] = int(self.strategyParams['halfToFat'] * half_time)
            self.strategyParams["timeBarrier"] = int(half_time * self.strategyParams['halfToTime'])
            #
            self.strategyParams["timeBarrier"] = 2
            #
            if self.strategyParams["timeBarrier"] <= 0:
                self.strategyParams["timeBarrier"] = 1

            self.strategyParams["varianceLookBack"] = int(half_time * self.strategyParams['halfToFat'])
            self.strategyParams["varianceRatioCarrete"] = int((half_time *
                                                               self.strategyParams['halfToFat']) //
                                                              self.strategyParams['varianceRatioCarreteParameter']) + 1

            if (half_time > self.strategyParams['scanHalfTime']) or (half_time < 2):
                if DEBUG:
                    print('Wrong HalfTime: ', half_time)
                return False

            workingAsk = self._PastPricesArrayMiddle[-self.strategyParams["rollingMean"]:]
            workingBid = self._PastPricesArrayMiddle[-self.strategyParams["rollingMean"]:]

            meanAsk = np.mean(workingAsk)
            meanBid = np.mean(workingBid)

            stdAsk = np.std(workingAsk)
            stdBid = np.std(workingBid)

            lowAskBand = round(meanAsk - stdAsk * self.strategyParams['yThreshold'], 3)
            highAskBand = round(meanAsk + stdAsk * self.strategyParams['yThreshold'], 3)

            lowBidBand = round(meanBid - stdBid * self.strategyParams['yThreshold'], 3)
            highBidBand = round(meanBid + stdBid * self.strategyParams['yThreshold'], 3)

            dictRet = {'AskStd': stdAsk, 'BidStd': stdBid, 'lowBid': lowBidBand, 'highBid': highBidBand,
                       'lowAsk': lowAskBand, 'highAsk': highAskBand, 'halfTime': half_time}

            return dictRet

    def _open_trade_ability(self):
        self._collect_new_price()
        Bands = self._calculate_bands()
        while not isinstance(Bands, dict):
            time.sleep(60)
            self._collect_new_price()
            Bands = self._calculate_bands()

        if (self._mode == 'multiCrossing') and (self._BBandsMode == 'Ask&Bid'):
            if self._PastPricesArrayBid[-1] > Bands['highAsk']:
                if DEBUG:
                    print(f"SELL Start searching sec cross with Bid higher to highASK")
                self._introWaitingTimer.start()
                while (self._introWaitingTimer.elapsed() // 60) < self.crossingMaxTime:
                    time.sleep(60.5)
                    self._collect_new_price()
                    Bands = self._calculate_bands()
                    if self._PastPricesArrayBid[-1] < Bands['highAsk']:
                        logTuple = self._PastPricesArrayBid[-(int(self.strategyParams['varianceLookBack']) + 1):]
                        retTuple = np.diff(logTuple)
                        logTuple = logTuple[1:]
                        assert len(retTuple) == len(logTuple)
                        if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                            openDict = dict()
                            openDict['typeOperation'] = 'SELL'
                            # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                            openDict['position'] = - int(round(self._tradeCapital))
                            openDict['openPrice'] = Bands['highAsk']
                            openDict['openTime'] = self.timer.elapsed()
                            openDict['stopLossBorder'] = round(Bands['highAsk'] + self.strategyParams['stopLossStdMultiplier'] * Bands['AskStd'], 3)
                            self._introWaitingTimer.stop()
                            return openDict
                if (self._introWaitingTimer.elapsed() // 60) >= self.crossingMaxTime:
                    self._introWaitingTimer.stop()
                    return 'CantOpenCrossing'

            if self._PastPricesArrayAsk[-1] < Bands['lowBid']:
                if DEBUG:
                    print(f"BUY Start searching sec cross with Ask higher to lowBid")
                self._introWaitingTimer.start()
                while (self._introWaitingTimer.elapsed() // 60) < self.crossingMaxTime:
                    time.sleep(60.5)
                    self._collect_new_price()
                    Bands = self._calculate_bands()
                    if self._PastPricesArrayAsk[-1] > Bands['lowBid']:
                        logTuple = self._PastPricesArrayAsk[-(int(self.strategyParams['varianceLookBack']) + 1):]
                        retTuple = np.diff(logTuple)
                        logTuple = logTuple[1:]
                        assert len(retTuple) == len(logTuple)
                        if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                            openDict = dict()
                            openDict['typeOperation'] = 'BUY'
                            # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                            openDict['position'] = int(round(self._tradeCapital))
                            openDict['openPrice'] = Bands['lowBid']
                            openDict['openTime'] = self.timer.elapsed()
                            openDict['stopLossBorder'] = round(Bands['lowBid'] - self.strategyParams['stopLossStdMultiplier'] * Bands['BidStd'], 3)
                            self._introWaitingTimer.stop()
                            return openDict
                if (self._introWaitingTimer.elapsed() // 60) >= self.crossingMaxTime:
                    self._introWaitingTimer.stop()
                    return 'CantOpenCrossing'

            # time.sleep(60.5)
            return 'NoFirstMatch'

        if (self._mode == 'singleCrossing') and (self._BBandsMode == 'Ask&Bid'):
            if self._PastPricesArrayBid[-1] > Bands['highAsk']:
                if DEBUG:
                    print(f"SELL Start searching sec cross with Bid higher to highASK")
                logTuple = self._PastPricesArrayBid[-(int(self.strategyParams['varianceLookBack']) + 1):]
                retTuple = np.diff(logTuple)
                logTuple = logTuple[1:]
                assert len(retTuple) == len(logTuple)
                if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                    openDict = dict()
                    openDict['typeOperation'] = 'SELL'
                    # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                    openDict['position'] = - int(round(self._tradeCapital))
                    openDict['openPrice'] = Bands['highAsk']
                    openDict['openTime'] = self.timer.elapsed()
                    openDict['stopLossBorder'] = round(Bands['highAsk'] + self.strategyParams['stopLossStdMultiplier'] * Bands['AskStd'], 3)
                    self._introWaitingTimer.stop()
                    return openDict

            if self._PastPricesArrayAsk[-1] < Bands['lowBid']:
                if DEBUG:
                    print(f"BUY Start searching sec cross with Ask higher to lowBid")

                logTuple = self._PastPricesArrayAsk[-(int(self.strategyParams['varianceLookBack']) + 1):]
                retTuple = np.diff(logTuple)
                logTuple = logTuple[1:]
                assert len(retTuple) == len(logTuple)
                if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                    openDict = dict()
                    openDict['typeOperation'] = 'BUY'
                    # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                    openDict['position'] = int(round(self._tradeCapital))
                    openDict['openPrice'] = Bands['lowBid']
                    openDict['openTime'] = self.timer.elapsed()
                    openDict['stopLossBorder'] = round(Bands['lowBid'] - self.strategyParams['stopLossStdMultiplier'] * Bands['BidStd'], 3)
                    self._introWaitingTimer.stop()
                    return openDict

            # time.sleep(60.5)
            return 'NoFirstMatch'

        if (self._mode == 'singleCrossing') and (self._BBandsMode == 'OnlyOne'):
            if self._PastPricesArrayMiddle[-1] > Bands['highAsk']:
                if DEBUG:
                    print(f"SELL Start searching sec cross with Bid higher to highASK")
                logTuple = self._PastPricesArrayMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
                retTuple = np.diff(logTuple)
                logTuple = logTuple[1:]
                assert len(retTuple) == len(logTuple)
                if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                    openDict = dict()
                    openDict['typeOperation'] = 'SELL'
                    # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                    openDict['position'] = - int(round(self._tradeCapital))
                    openDict['openPrice'] = Bands['highAsk']
                    openDict['openTime'] = self.timer.elapsed()
                    openDict['stopLossBorder'] = round(Bands['highAsk'] + self.strategyParams['stopLossStdMultiplier'] * Bands['AskStd'], 3)
                    self._introWaitingTimer.stop()
                    return openDict

            if self._PastPricesArrayMiddle[-1] < Bands['lowBid']:
                if DEBUG:
                    print(f"BUY Start searching sec cross with Ask higher to lowBid")

                logTuple = self._PastPricesArrayMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
                retTuple = np.diff(logTuple)
                logTuple = logTuple[1:]
                assert len(retTuple) == len(logTuple)
                if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                    openDict = dict()
                    openDict['typeOperation'] = 'BUY'
                    # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                    openDict['position'] = int(round(self._tradeCapital))
                    openDict['openPrice'] = Bands['lowBid']
                    openDict['openTime'] = self.timer.elapsed()
                    openDict['stopLossBorder'] = round(Bands['lowBid'] - self.strategyParams['stopLossStdMultiplier'] * Bands['BidStd'], 3)
                    self._introWaitingTimer.stop()
                    return openDict

            # time.sleep(60.5)
            return 'NoFirstMatch'

        if (self._mode == 'multiCrossing') and (self._BBandsMode == 'OnlyOne'):
            if self._PastPricesArrayMiddle[-1] > Bands['highAsk']:
                if DEBUG:
                    print(f"SELL Start searching sec cross with Bid higher to highASK")
                self._introWaitingTimer.start()
                while (self._introWaitingTimer.elapsed() // 60) < self.crossingMaxTime:
                    time.sleep(60.5)
                    self._collect_new_price()
                    Bands = self._calculate_bands()
                    if self._PastPricesArrayMiddle[-1] < Bands['highAsk']:
                        logTuple = self._PastPricesArrayMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
                        retTuple = np.diff(logTuple)
                        logTuple = logTuple[1:]
                        assert len(retTuple) == len(logTuple)
                        if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                            openDict = dict()
                            openDict['typeOperation'] = 'SELL'
                            # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                            openDict['position'] = - int(round(self._tradeCapital))
                            openDict['openPrice'] = Bands['highAsk']
                            openDict['openTime'] = self.timer.elapsed()
                            openDict['stopLossBorder'] = round(
                                Bands['highAsk'] + self.strategyParams['stopLossStdMultiplier'] * Bands['AskStd'], 3)
                            self._introWaitingTimer.stop()
                            return openDict
                if (self._introWaitingTimer.elapsed() // 60) >= self.crossingMaxTime:
                    self._introWaitingTimer.stop()
                    return 'CantOpenCrossing'

            if self._PastPricesArrayMiddle[-1] < Bands['lowBid']:
                if DEBUG:
                    print(f"BUY Start searching sec cross with Ask higher to lowBid")
                self._introWaitingTimer.start()
                while (self._introWaitingTimer.elapsed() // 60) < self.crossingMaxTime:
                    time.sleep(60.5)
                    self._collect_new_price()
                    Bands = self._calculate_bands()
                    if self._PastPricesArrayMiddle[-1] > Bands['lowBid']:
                        logTuple = self._PastPricesArrayMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
                        retTuple = np.diff(logTuple)
                        logTuple = logTuple[1:]
                        assert len(retTuple) == len(logTuple)
                        if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                            openDict = dict()
                            openDict['typeOperation'] = 'BUY'
                            # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                            openDict['position'] = int(round(self._tradeCapital))
                            openDict['openPrice'] = Bands['lowBid']
                            openDict['openTime'] = self.timer.elapsed()
                            openDict['stopLossBorder'] = round(
                                Bands['lowBid'] - self.strategyParams['stopLossStdMultiplier'] * Bands['BidStd'], 3)
                            self._introWaitingTimer.stop()
                            return openDict
                if (self._introWaitingTimer.elapsed() // 60) >= self.crossingMaxTime:
                    self._introWaitingTimer.stop()
                    return 'CantOpenCrossing'

            # time.sleep(60.5)
            return 'NoFirstMatch'

    def _buyStop(self):
        if self._BBandsMode == 'Ask&Bid':
            self._collect_new_price()
            if (self.tradingTimer.elapsed() // 60) > self.strategyParams['timeBarrier']:
                return {'typeHolding': 'endPeriod', 'closePrice': self._PastPricesArrayBid[-1]}

            if self._PastPricesArrayBid[-1] < self._positionDetails['stopLossBorder']:
                if DEBUG:
                    print(f"Stop loss, price={self._PastPricesArrayBid[-1]}, stopLoss = {self._positionDetails['stopLossBorder']}")
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayBid[-1]}
            # Block with Trailing StopLoss. This realization is not good. Need to change
            delta = self._PastPricesArrayBid[-1] - self._PastPricesArrayBid[-2]
            if delta > 0:
                self._positionDetails['stopLossBorder'] = round(self._positionDetails['stopLossBorder'] + delta, 3)

            if self._PastPricesArrayBid[-1] < self._positionDetails['stopLossBorder']:
                if DEBUG:
                    print(f"Stop loss after upper level, price={self._PastPricesArrayBid[-1]}, stopLoss = {self._positionDetails['stopLossBorder']}")
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayBid[-1]}

            if not self.waitingToFatMean:
                workingArray = self._PastPricesArrayBid[-int(self.strategyParams['rollingMean']):]
                bandMean = np.mean(workingArray)
                MeanFat = np.mean(self._PastPricesArrayBid[-int(self.strategyParams['fatRollingMean']):])
                if (self._PastPricesArrayBid[-1] > bandMean) and (self._PastPricesArrayBid[-2] < bandMean):
                    _log = self._PastPricesArrayBid[-(int(max(self.strategyParams['varianceLookBack'], self.strategyParams['fatRollingMean']))+1):]
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
                MeanFat = np.mean(self._PastPricesArrayBid[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArrayBid[-1] > MeanFat:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': MeanFat}

                _log = self._PastPricesArrayBid[
                       -(int(max(self.strategyParams['varianceLookBack'], self.strategyParams['fatRollingMean'])) + 1):]
                compute = {
                    "retOpenPrice": np.diff(_log),
                    "logOpenPrice": _log[1:]
                }
                if not reverse_variance_ratio(preComputed=compute, params=self.strategyParams, timeBorderCounter=self.tradingTimer.elapsed() // 60, VRstatement=self.waitingToFatMean):
                    self.waitingToFatMean = False
                    return False

            return False

        if self._BBandsMode == 'OnlyOne':
            self._collect_new_price()
            if (self.tradingTimer.elapsed() // 60) > self.strategyParams['timeBarrier']:
                return {'typeHolding': 'endPeriod', 'closePrice': self._PastPricesArrayMiddle[-1]}

            if self._PastPricesArrayMiddle[-1] < self._positionDetails['stopLossBorder']:
                if DEBUG:
                    print(
                        f"Stop loss, price={self._PastPricesArrayMiddle[-1]}, stopLoss = {self._positionDetails['stopLossBorder']}")
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayMiddle[-1]}
            # Block with Trailing StopLoss. This realization is not good. Need to change
            delta = self._PastPricesArrayMiddle[-1] - self._PastPricesArrayMiddle[-2]
            if delta > 0:
                self._positionDetails['stopLossBorder'] = round(self._positionDetails['stopLossBorder'] + delta, 3)

            if self._PastPricesArrayMiddle[-1] < self._positionDetails['stopLossBorder']:
                if DEBUG:
                    print(
                        f"Stop loss after upper level, price={self._PastPricesArrayMiddle[-1]}, stopLoss = {self._positionDetails['stopLossBorder']}")
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayMiddle[-1]}

            if not self.waitingToFatMean:
                workingArray = self._PastPricesArrayMiddle[-int(self.strategyParams['rollingMean']):]
                bandMean = np.mean(workingArray)
                MeanFat = np.mean(self._PastPricesArrayMiddle[-int(self.strategyParams['fatRollingMean']):])
                if (self._PastPricesArrayMiddle[-1] > bandMean) and (self._PastPricesArrayMiddle[-2] < bandMean):
                    _log = self._PastPricesArrayMiddle[-(int(
                        max(self.strategyParams['varianceLookBack'], self.strategyParams['fatRollingMean'])) + 1):]
                    compute = {
                        "retOpenPrice": np.diff(_log),
                        "logOpenPrice": _log[1:]
                    }
                    assert len(compute['retOpenPrice']) == len(compute['logOpenPrice'])
                    if MeanFat > bandMean:
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
                MeanFat = np.mean(self._PastPricesArrayMiddle[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArrayMiddle[-1] > MeanFat:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': MeanFat}

                _log = self._PastPricesArrayMiddle[
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

    def _shortStop(self):
        if self._BBandsMode == "Ask&Bid":
            if self._PastPricesArrayAsk[-1] > self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayAsk[-1]}
            # Block with Trailing StopLoss. This realization is not good. Need to change
            delta = self._PastPricesArrayAsk[-1] - self._PastPricesArrayAsk[-2]
            if delta < 0:
                self._positionDetails['stopLossBorder'] = round(self._positionDetails['stopLossBorder'] - delta, 3)

            if self._PastPricesArrayAsk[-1] > self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayAsk[-1]}

            if not self.waitingToFatMean:
                workingArray = self._PastPricesArrayAsk[-int(self.strategyParams['rollingMean']):]
                bandMean = np.mean(workingArray)
                MeanFat = np.mean(self._PastPricesArrayAsk[-int(self.strategyParams['fatRollingMean']):])
                if (self._PastPricesArrayAsk[-1] < bandMean) and (self._PastPricesArrayAsk[-2] > bandMean):
                    _log = self._PastPricesArrayAsk[-(int(
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
                MeanFat = np.mean(self._PastPricesArrayAsk[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArrayAsk[-1] < MeanFat:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': MeanFat}

                _log = self._PastPricesArrayAsk[
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
        if self._BBandsMode == 'OnlyOne':
            if self._PastPricesArrayMiddle[-1] > self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayMiddle[-1]}
            # Block with Trailing StopLoss. This realization is not good. Need to change
            delta = self._PastPricesArrayMiddle[-1] - self._PastPricesArrayMiddle[-2]
            if delta < 0:
                self._positionDetails['stopLossBorder'] = round(self._positionDetails['stopLossBorder'] - delta, 3)

            if self._PastPricesArrayMiddle[-1] > self._positionDetails['stopLossBorder']:
                return {'typeHolding': 'stopLoss', 'closePrice': self._PastPricesArrayMiddle[-1]}

            if not self.waitingToFatMean:
                workingArray = self._PastPricesArrayMiddle[-int(self.strategyParams['rollingMean']):]
                bandMean = np.mean(workingArray)
                MeanFat = np.mean(self._PastPricesArrayMiddle[-int(self.strategyParams['fatRollingMean']):])
                if (self._PastPricesArrayMiddle[-1] < bandMean) and (self._PastPricesArrayMiddle[-2] > bandMean):
                    _log = self._PastPricesArrayMiddle[-(int(
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
                MeanFat = np.mean(self._PastPricesArrayMiddle[-int(self.strategyParams['fatRollingMean']):])
                if self._PastPricesArrayMiddle[-1] < MeanFat:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': MeanFat}

                _log = self._PastPricesArrayMiddle[
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

    def _trading_loop(self, typeOrder='market'):
        # Waiting until we can open a trade
        while (not self._inPosition):
            answer = self._open_trade_ability()
            while not isinstance(answer, dict):
                print('Sleep answer 60')
                time.sleep(60)
                answer = self._open_trade_ability()
                print('ANSWER', answer)
            if typeOrder == 'limit':
                orderID = self.connector.place_order({self.SAXO: answer['position']}, order_type='limit',
                                                     order_price=answer['openPrice'])['OrderId']
                # print(self.connector.get_actual_data([self.SAXO], mode='all'))
                orderIDminute = datetime.datetime.now().minute
                while (orderIDminute == datetime.datetime.now().minute) and (not self._inPosition):
                    time.sleep(1)
                    orderStatus = self.connector.check_order(orderID)
                    if not orderStatus:
                        self._inPosition = True
                        self.tradingTimer.start()
                if not self._inPosition:
                    self.connector.cancelOrder(orderID)
            if typeOrder == 'market':
                orderID = self.connector.place_order({self.SAXO: answer['position']}, order_type='market')
                self._inPosition = True
                self.tradingTimer.start()
                print('Complete trade')
        self._positionDetails = answer
        time.sleep(60)
        while self._inPosition:
            if answer['typeOperation'] == 'BUY':
                hold = self._buyStop()
                while not isinstance(hold, dict):
                    time.sleep(60)
                    hold = self._buyStop()
                    if DEBUG:
                        print('Hold', hold)
                if typeOrder == 'limit':
                    orderID = self.connector.place_order({self.SAXO: -1 * answer['position']}, order_type='limit',
                                                         order_price=answer['openPrice'])['OrderId']
                    # print(self.connector.get_actual_data([self.SAXO], mode='all'))
                    orderIDminute = datetime.datetime.now().minute
                    while (orderIDminute == datetime.datetime.now().minute) and (not self._inPosition):
                        time.sleep(1)
                        orderStatus = self.connector.check_order(orderID)
                        if not orderStatus:
                            self._inPosition = False
                            self.tradingTimer.stop()
                    if not self._inPosition:
                        self.connector.cancelOrder(orderID)
                if typeOrder == 'market':
                    orderID = self.connector.place_order({self.SAXO: -1 * answer['position']}, order_type='market')
                    self._inPosition = False
                    self.tradingTimer.stop()
                    print('Complete stop trade')

            if answer['typeOperation'] == 'SELL':
                hold = self._shortStop()
                while not isinstance(hold, dict):
                    time.sleep(60)
                    hold = self._shortStop()
                    if DEBUG:
                        print('Hold', hold)
                if typeOrder == 'limit':
                    orderID = self.connector.place_order({self.SAXO: -1 * answer['position']}, order_type='limit',
                                                         order_price=answer['openPrice'])['OrderId']
                    # print(self.connector.get_actual_data([self.SAXO], mode='all'))
                    orderIDminute = datetime.datetime.now().minute
                    while (orderIDminute == datetime.datetime.now().minute) and (not self._inPosition):
                        time.sleep(1)
                        orderStatus = self.connector.check_order(orderID)
                        if not orderStatus:
                            self._inPosition = False
                            self.tradingTimer.stop()
                    if not self._inPosition:
                        self.connector.cancelOrder(orderID)
                if typeOrder == 'market':
                    orderID = self.connector.place_order({self.SAXO: -1 * answer['position']}, order_type='market')
                    self._inPosition = False
                    self.tradingTimer.stop()
                    print('Complete stop trade')

        _stat = {**answer, **hold}
        _stat['StrategyWorkingTime'] = self.timer.elapsed()
        _stat['DateTime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.statCollector.add_trade_line(_stat)
        time.sleep(60)


    def start_tradingCycle(self):
        self._inPosition = False
        self._collect_past_prices()
        initMinute = datetime.datetime.now().minute
        if (self.timer is None) or (self.tradingTimer is None):
            raise ModuleNotFoundError('Timer not plugged')
        if self.connector is None:
            raise ModuleNotFoundError('Connector not plugged')
        if self.statCollector is None:
            print('Warning no statistic collector plugged')
        pass

        self.timer.start()
        while initMinute == datetime.datetime.now().minute:
            time.sleep(0.05)

        time.sleep(6.5)
        print('waited')
        while True:
            # print('Last Price in Slicer:', self._PastPricesArray[-1])
            if (self.timerToken.elapsed() // 3600) > 3:
                with open("token.txt", 'r') as f:
                    tokenContent = f.readline()[:-1]
                    print(tokenContent)
                    tokenLife = f.readline()
                    tokenLife = datetime.datetime.fromisoformat(tokenLife)
                    if ((datetime.datetime.now() - tokenLife).total_seconds() // 60 // 60) > 22:
                        print('Token will expire soon')
                self.timerToken.stop()
                self.timerToken.start()

            self.strategyParams = create_strategy_config(self._initStrategyParams, CAP=self._tradeCapital)
            self._trading_loop()


parser = argparse.ArgumentParser(description='Creates csv into backTest dir')
parser.add_argument('--currencyPair', help='for example CHFJPY', default='CHFJPY')
args = parser.parse_args()


monkeyRobot = ImRobot('MNKY', config_file_way="robotConfig.txt", strategyParameters_file_way="strategyParameters.txt",
                      tickerSaxo=f'{args.currencyPair}', tickerEOD=f'{args.currencyPair}.FOREX')

timerGlobal = Timer()
timerTrade = Timer()
tokenTimer = Timer()
waitCrossing = Timer()


monkeyRobot.timerToken = tokenTimer
monkeyRobot._introWaitingTimer = waitCrossing
# connector = SimulatedOrderGenerator("dataForGenerator.csv")
with open("token.txt", 'r') as f:
    tokenContent = f.readline()[:-1]
    print(tokenContent)
    tokenLife = f.readline()
    tokenLife = datetime.datetime.fromisoformat(tokenLife)
    if ((datetime.datetime.now() - tokenLife).total_seconds() // 60 // 60) > 22:
        print('Token will expire soon')


connector = SaxoOrderInterface(token=tokenContent)
pandasCollector = PandasStatCollector("stat.csv")
#
#
monkeyRobot.add_timer(timerGlobal, timerTrade)
monkeyRobot.add_statistics_collector(pandasCollector)
monkeyRobot.add_connector(connector)

DEBUG = True

print(pd.DataFrame(monkeyRobot.connector.get_asset_data_hist(ticker='CHFJPY', density=60, amount_intervals=1000)).columns)
# monkeyRobot.timerToken.start()
# monkeyRobot.start_tradingCycle()
# monkeyRobot.connector.place_order({'CHFJPY': 100_000}, order_type='limit', order_price=134.312)


# monkeyRobot.connector.place_order({'CHFJPY': -100_000}, order_type='market')
# monkeyRobot.connector.place_order({'CHFJPY': monkeyRobot._tradeCapital})

# monkeyRobot.connector.place_order({monkeyRobot.SAXO: 100_000}, order_type='market')
# monkeyRobot.connector.get_actual_data(['CHFJPY'])
# monkeyRobot.connector.stop_order(ticker='CHFJPY', amount=400000)
