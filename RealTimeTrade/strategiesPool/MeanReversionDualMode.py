from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import time


class MeanReversionDual:
    pass


from RealTimeTrade.TradingInterface import TradingInterface
from RealTimeTrade.utils.utils import *
from RealTimeTrade.utils.timerModule import Timer


class AbstractStrategy(ABC):
    pass


class MeanReversionDual(AbstractStrategy):
    UnableToOpenLog = "LOG | UnableOpen: "
    UnableToOpenLogCross = UnableToOpenLog + " no second crossing: "

    availableTradingInterface = Union[None, TradingInterface]
    availableBBandsModes = Union['Ask&Bid', 'OnlyOne']
    availableOpenCrossingModes = Union['multiCrossing', 'singleCrossing']

    tradingInterface: availableTradingInterface
    BBandsMode: availableBBandsModes
    openMode: availableOpenCrossingModes

    def __init__(self, strategyConfigPath: str, strategyModePath: str):
        super(MeanReversionDual, self).__init__()
        self._tradeCapital = 100_000

        self.tradingInterface = None

        mode = pd.read_csv(strategyModePath, header=None).T
        mode = pd.Series(data=mode.iloc[1, :].values,
                                             index=mode.iloc[0, :])
        self.openMode = mode['OpenCrossingMode']
        self.BBandsMode = mode['BBandsMode']
        self.maxCrossingParameter = mode['waitingParameter']
        del mode

        self._initStrategyParams = pd.read_csv(strategyConfigPath, header=None).T
        self._initStrategyParams = pd.Series(data=self._initStrategyParams.iloc[1, :].values,
                                             index=self._initStrategyParams.iloc[0, :])

        self.strategyParams = create_strategy_config(self._initStrategyParams, CAP=self._tradeCapital)

        self.Bands = None
        pass

    def add_trading_interface(self, tradingInterface: availableTradingInterface):
        self.tradingInterface = tradingInterface

    def _make_bollinger_bands(self):
        if self.BBandsMode == 'Ask&Bid':
            half_time = int(
                get_half_time(pd.Series(self.tradingInterface.OpenMiddle[-int(self.strategyParams['scanHalfTime']):])))

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
                return f'LOG | Unable to calculate BBands because half-time have error value: {half_time}'

            workingAsk = self.tradingInterface.OpenAsk[-self.strategyParams["rollingMean"]:]
            workingBid = self.tradingInterface.OpenBid[-self.strategyParams["rollingMean"]:]

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

        if self.BBandsMode == 'OnlyOne':
            half_time = int(
                get_half_time(pd.Series(self.tradingInterface.OpenMiddle[-int(self.strategyParams['scanHalfTime']):])))

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
                return f'LOG | Unable to calculate BBands because half-time have error value: {half_time}'

            workingMiddle = self.tradingInterface.OpenMiddle[-self.strategyParams["rollingMean"]:]

            meanMiddle = np.mean(workingMiddle)

            stdMiddle = np.std(workingMiddle)

            lowMiddleBand = round(meanMiddle - stdMiddle * self.strategyParams['yThreshold'], 3)
            highMiddleBand = round(meanMiddle + stdMiddle * self.strategyParams['yThreshold'], 3)

            dictRet = {'AskStd': stdMiddle, 'BidStd': stdMiddle, 'lowBid': lowMiddleBand, 'highBid': highMiddleBand,
                       'lowAsk': lowMiddleBand, 'highAsk': highMiddleBand, 'halfTime': half_time}

            return dictRet

    def _multi_cross_ask_and_bid(self):
        if self.tradingInterface.OpenBid[-1] > self.Bands['highAsk']:
            WaitingTimer = Timer()
            WaitingTimer.start()
            while (WaitingTimer.elapsed() // 60) < self.maxCrossingParameter:
                time.sleep(self.tradingInterface.updatableDataTime)
                fresh_data = self.tradingInterface.download_actual_dot(density=self.tradingInterface.updatableDataTime)
                if self.tradingInterface.debug:
                    print(fresh_data)
                self.Bands = self._make_bollinger_bands()
                if self.tradingInterface.OpenBid[-1] < self.Bands['highAsk']:
                    logTuple = self.tradingInterface.OpenBid[-(int(self.strategyParams['varianceLookBack']) + 1):]
                    retTuple = np.diff(logTuple)
                    logTuple = logTuple[1:]
                    assert len(retTuple) == len(logTuple)
                    if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                        openDict = dict()
                        openDict['typeOperation'] = 'SELL'
                        # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                        openDict['position'] = - int(round(self._tradeCapital))
                        openDict['openPrice'] = self.Bands['highAsk']
                        openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                        openDict['stopLossBorder'] = round(self.Bands['highAsk'] +
                                                           self.strategyParams['stopLossStdMultiplier']
                                                           * self.Bands['AskStd'], 3)
                        WaitingTimer.stop()
                        return openDict
                if (WaitingTimer.elapsed() // 60) >= self.maxCrossingParameter:
                    WaitingTimer.stop()
                    del WaitingTimer
                    return f'{self.UnableToOpenLogCross}CantOpenCrossing'

        if self.tradingInterface.OpenAsk[-1] < self.Bands['lowBid']:
            WaitingTimer = Timer()
            WaitingTimer.start()
            while (WaitingTimer.elapsed() // 60) < self.maxCrossingParameter:
                time.sleep(self.tradingInterface.updatableDataTime)
                fresh_data = self.tradingInterface.download_actual_dot(density=self.tradingInterface.updatableDataTime)
                if self.tradingInterface.debug:
                    print(fresh_data)
                self.Bands = self._make_bollinger_bands()
                if self.tradingInterface.OpenAsk[-1] > self.Bands['lowBid']:
                    logTuple = self.tradingInterface.OpenAsk[-(int(self.strategyParams['varianceLookBack']) + 1):]
                    retTuple = np.diff(logTuple)
                    logTuple = logTuple[1:]
                    assert len(retTuple) == len(logTuple)
                    if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                        openDict = dict()
                        openDict['typeOperation'] = 'BUY'
                        # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                        openDict['position'] = int(round(self._tradeCapital))
                        openDict['openPrice'] = self.Bands['lowBid']
                        openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                        openDict['stopLossBorder'] = round(self.Bands['lowBid'] -
                                                           self.strategyParams['stopLossStdMultiplier']
                                                           * self.Bands['BidStd'], 3)
                        WaitingTimer.stop()
                        return openDict
            if (WaitingTimer.elapsed() // 60) >= self.maxCrossingParameter:
                WaitingTimer.stop()
                del WaitingTimer
                return f'{self.UnableToOpenLogCross}CantOpenCrossing'

        return f"{self.UnableToOpenLog}no match"

    def _multi_cross_only_middle(self):
        if self.tradingInterface.OpenMiddle[-1] > self.Bands['highAsk']:
            WaitingTimer = Timer()
            WaitingTimer.start()
            while (WaitingTimer.elapsed() // 60) < self.maxCrossingParameter:
                time.sleep(self.tradingInterface.updatableDataTime)
                fresh_data = self.tradingInterface.download_actual_dot(density=self.tradingInterface.updatableDataTime)
                if self.tradingInterface.debug:
                    print(fresh_data)
                self.Bands = self._make_bollinger_bands()
                if self.tradingInterface.OpenMiddle[-1] < self.Bands['highAsk']:
                    logTuple = self.tradingInterface.OpenMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
                    retTuple = np.diff(logTuple)
                    logTuple = logTuple[1:]
                    assert len(retTuple) == len(logTuple)
                    if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                        openDict = dict()
                        openDict['typeOperation'] = 'SELL'
                        # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                        openDict['position'] = - int(round(self._tradeCapital))
                        openDict['openPrice'] = self.Bands['highAsk']
                        openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                        openDict['stopLossBorder'] = round(self.Bands['highAsk'] +
                                                           self.strategyParams['stopLossStdMultiplier']
                                                           * self.Bands['AskStd'], 3)
                        WaitingTimer.stop()
                        return openDict
                if (WaitingTimer.elapsed() // 60) >= self.maxCrossingParameter:
                    WaitingTimer.stop()
                    del WaitingTimer
                    return f'{self.UnableToOpenLogCross}CantOpenCrossing'

        if self.tradingInterface.OpenMiddle[-1] < self.Bands['lowBid']:
            WaitingTimer = Timer()
            WaitingTimer.start()
            while (WaitingTimer.elapsed() // 60) < self.maxCrossingParameter:
                time.sleep(self.tradingInterface.updatableDataTime)
                fresh_data = self.tradingInterface.download_actual_dot(density=self.tradingInterface.updatableDataTime)
                if self.tradingInterface.debug:
                    print(fresh_data)
                self.Bands = self._make_bollinger_bands()
                if self.tradingInterface.OpenMiddle[-1] > self.Bands['lowBid']:
                    logTuple = self.tradingInterface.OpenMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
                    retTuple = np.diff(logTuple)
                    logTuple = logTuple[1:]
                    assert len(retTuple) == len(logTuple)
                    if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                        openDict = dict()
                        openDict['typeOperation'] = 'BUY'
                        # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                        openDict['position'] = int(round(self._tradeCapital))
                        openDict['openPrice'] = self.Bands['lowBid']
                        openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                        openDict['stopLossBorder'] = round(self.Bands['lowBid'] -
                                                           self.strategyParams['stopLossStdMultiplier']
                                                           * self.Bands['BidStd'], 3)
                        WaitingTimer.stop()
                        return openDict
            if (WaitingTimer.elapsed() // 60) >= self.maxCrossingParameter:
                WaitingTimer.stop()
                del WaitingTimer
                return f'{self.UnableToOpenLogCross}CantOpenCrossing'

        return f"{self.UnableToOpenLog}no crossing b bands"

    def _single_cross_ask_and_bid(self):
        if self.tradingInterface.OpenBid[-1] > self.Bands['highAsk']:
            logTuple = self.tradingInterface.OpenBid[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)
            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict = dict()
                openDict['typeOperation'] = 'SELL'
                # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                openDict['position'] = - int(round(self._tradeCapital))
                openDict['openPrice'] = self.Bands['highAsk']
                openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                openDict['stopLossBorder'] = round(
                    self.Bands['highAsk'] + self.strategyParams['stopLossStdMultiplier'] * self.Bands['AskStd'], 3)
                return openDict

        if self.tradingInterface.OpenAsk[-1] < self.Bands['lowBid']:
            logTuple = self.tradingInterface.OpenAsk[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)
            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict = dict()
                openDict['typeOperation'] = 'BUY'
                # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                openDict['position'] = int(round(self._tradeCapital))
                openDict['openPrice'] = self.Bands['lowBid']
                openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                openDict['stopLossBorder'] = round(
                    self.Bands['lowBid'] - self.strategyParams['stopLossStdMultiplier'] * self.Bands['BidStd'], 3)
                return openDict

        return f"{self.UnableToOpenLog}no crossing b bands"

    def _single_cross_only_middle(self):
        if self.tradingInterface.OpenMiddle[-1] > self.Bands['highAsk']:
            logTuple = self.tradingInterface.OpenMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)
            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict = dict()
                openDict['typeOperation'] = 'SELL'
                # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                openDict['position'] = - int(round(self._tradeCapital))
                openDict['openPrice'] = self.Bands['highAsk']
                openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                openDict['stopLossBorder'] = round(
                    self.Bands['highAsk'] + self.strategyParams['stopLossStdMultiplier'] * self.Bands['AskStd'], 3)
                return openDict

        if self.tradingInterface.OpenMiddle[-1] < self.Bands['lowBid']:
            logTuple = self.tradingInterface.OpenMiddle[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)
            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict = dict()
                openDict['typeOperation'] = 'BUY'
                # openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                openDict['position'] = int(round(self._tradeCapital))
                openDict['openPrice'] = self.Bands['lowBid']
                openDict['openTime'] = self.tradingInterface.globalTimer.elapsed()
                openDict['stopLossBorder'] = round(
                    self.Bands['lowBid'] - self.strategyParams['stopLossStdMultiplier'] * self.Bands['BidStd'], 3)
                return openDict

        return f"{self.UnableToOpenLog}no crossing b bands"

    def open_trade_ability(self):
        self.Bands = self._make_bollinger_bands()
        while not isinstance(self.Bands, dict):
            time.sleep(self.tradingInterface.updatableDataTime)
            fresh_data = self.tradingInterface.download_actual_dot(density=self.tradingInterface.updatableDataTime)
            if self.tradingInterface.debug:
                print(fresh_data)
            self.Bands = self._make_bollinger_bands()

        if (self.openMode == 'multiCrossing') and (self.BBandsMode == 'Ask&Bid'):
            self._multi_cross_ask_and_bid()

        if (self.openMode == 'multiCrossing') and (self.BBandsMode == 'OnlyOne'):
            self._multi_cross_only_middle()

        if (self.openMode == 'singleCrossing') and (self.BBandsMode == 'Ask&Bid'):
            self._single_cross_ask_and_bid()

        if (self.openMode == 'singleCrossing') and (self.BBandsMode == 'OnlyOne'):
            self._single_cross_only_middle()