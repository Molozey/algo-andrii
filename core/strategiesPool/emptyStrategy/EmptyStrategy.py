from core.strategiesPool._baseStrategy import AbstractStrategy
from pandas import read_csv, Series
from core.strategiesPool.meanReversionPrediction.requiredInfo import required


class EmptyStratergy(AbstractStrategy):
    def __init__(self, strategyConfigPath, strategyModePath):
        super(EmptyStratergy, self).__init__(strategyConfigPath=strategyConfigPath,
                                             strategyModePath=strategyModePath)

        self.required_assets = required

    def open_trade_ability(self):
        return True

    def close_trade_ability(self, openDetails):
        return True

    def add_trading_interface(self, tradingInterface):
        pass
