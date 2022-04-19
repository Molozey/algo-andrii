from abc import ABC, abstractmethod
from typing import Union

from RealTimeTrade.TradingInterface import TradingInterface


class AbstractStrategy(ABC):
    pass


class MeanReversionDual(AbstractStrategy):
    availableTradingInterface = Union[None, TradingInterface]
    availableBBandsModes = Union['Ask&Bid', 'OnlyOne']
    availableOpenCrossingModes = Union['multiCrossing', 'singleCrossing']


    tradingInterface: availableTradingInterface
    BBandsMode: availableBBandsModes
    openMode: availableOpenCrossingModes

    def __init__(self, BBandsMode, openCrossMode):
        super(MeanReversionDual, self).__init__()

        self.tradingInterface = None
        self.openMode = openCrossMode
        self.BBandsMode = BBandsMode
        pass

    def add_trading_interface(self, tradingInterface: availableTradingInterface):
        self.tradingInterface = tradingInterface

    def _make_bollinger_bands(self):
