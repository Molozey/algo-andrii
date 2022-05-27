from core.connectors.SAXO import SaxoOrderInterface
from core.statCollectors.pandasSaver import PandasStatCollector
from core.TradingInterface import TradingInterface
from core.utils.TelegramNotificator import TelegramNotification
from core.strategiesPool.meanReversionPrediction.MeanReversionPredict import MeanReversionLightWithPrediction
from core.warnings.loggingPresets import *


if __name__ == '__main__':
    DEBUG_MODE = True
    #initialize strategy
    _path = "core/strategiesPool/meanReversionOld/"
    strategy = MeanReversionLightWithPrediction(strategyConfigPath=_path + 'MeanReversionStrategyParameters.txt',
                                                strategyModePath=_path + 'DualMeanConfig.txt')
    del _path
    required = strategy.required_assets
    # initialize
    monkey = TradingInterface(name='monkey', robotConfig='robotConfig.txt', assets=required,
                              requireTokenUpdate=True, debug=DEBUG_MODE)
    # add collector
    monkey.add_statistics_collector(PandasStatCollector(fileToSave='stat.csv', detailsPath='details.csv'))
    # add saxo interface
    monkey.add_broker_interface(SaxoOrderInterface(monkey.get_token))
    # add telegram notificator
    # monkey.add_fast_notificator(TelegramNotification())
    monkey.add_strategy(strategy)
    monkey.strategy.add_trading_interface(monkey)
    # monkey.start_execution()
    # print(monkey.make_order(orderDetails={"position": 100_000, "openPrice": 134.425}, typePos="open", openDetails=None))


