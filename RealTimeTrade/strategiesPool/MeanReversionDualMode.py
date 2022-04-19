from abc import ABC, abstractmethod
from typing import Union
import pandas as pd


class MeanReversionDual:
    pass


from RealTimeTrade.TradingInterface import TradingInterface
from RealTimeTrade.utils.utils import *


class AbstractStrategy(ABC):
    pass


class MeanReversionDual(AbstractStrategy):
    availableTradingInterface = Union[None, TradingInterface]
    availableBBandsModes = Union['Ask&Bid', 'OnlyOne']
    availableOpenCrossingModes = Union['multiCrossing', 'singleCrossing']

    tradingInterface: availableTradingInterface
    BBandsMode: availableBBandsModes
    openMode: availableOpenCrossingModes

    def __init__(self, strategyConfigPath: str, BBandsMode: availableBBandsModes,
                 openCrossMode: availableOpenCrossingModes):
        super(MeanReversionDual, self).__init__()
        self._tradeCapital = 100_000

        self.tradingInterface = None
        self.openMode = openCrossMode
        self.BBandsMode = BBandsMode

        self._initStrategyParams = pd.read_csv(strategyConfigPath, header=None).T
        self._initStrategyParams = pd.Series(data=self._initStrategyParams.iloc[1, :].values,
                                             index=self._initStrategyParams.iloc[0, :])

        self.strategyParams = create_strategy_config(self._initStrategyParams, CAP=self._tradeCapital)
        pass

    def add_trading_interface(self, tradingInterface: availableTradingInterface):
        self.tradingInterface = tradingInterface

    def _make_bollinger_bands(self):
        if self.BBandsMode == 'Ask&Bid':
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

            dictRet = {'AskStd': stdAsk, 'BidStd': stdBid, 'lowBid': lowBidBand, 'highBid': highBidBand,
                       'lowAsk': lowAskBand, 'highAsk': highAskBand, 'halfTime': half_time}

        if self.BBandsMode == 'OnlyOne':
            pass

