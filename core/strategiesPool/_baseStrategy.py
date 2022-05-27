from abc import ABC, abstractmethod
from pandas import read_csv, Series


class AbstractStrategy(ABC):
    def __init__(self, strategyConfigPath: str, strategyModePath: str):
        self.mode = read_csv('../' + strategyModePath, header=None).T
        self.mode = Series(data=self.mode.iloc[1, :].values,
                           index=self.mode.iloc[0, :])

        self._initStrategyParams = read_csv('../' + strategyConfigPath, header=None).T
        self._initStrategyParams = Series(data=self._initStrategyParams.iloc[1, :].values,
                                          index=self._initStrategyParams.iloc[0, :])

    @abstractmethod
    def open_trade_ability(self) -> dict:
        pass

    @abstractmethod
    def close_trade_ability(self, openDetails) -> dict:
        pass

    @abstractmethod
    def add_trading_interface(self, tradingInterface):
        pass
