import pandas as pd
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
        half_time = int(get_half_time(self._PastPricesArray[-int(self.strategyParams.scanHalfTime):]))
        if (half_time > self.strategyParams['scanHalfTime']) or (half_time < 0):
            return False
        pass

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
            self._trading_loop()

a = [2,3,4,5,5,6,6,6,6,6,6,6,10,6]
print(a[-5:])