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
        self._tradeCapital = 100_000_0
        config = pd.read_csv(str(config_file_way), header=None, index_col=0, sep=':')
        self.time_interval = int(config.loc['updateTime'])
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
        self.SAXO = tickerSaxo
        self.EOD = tickerEOD

    def _collect_past_prices(self):
        # We need at least self._initStrategyParams.scanHalfTime
        shiftedTime = (datetime.datetime.now() - pd.Timedelta(f'{self._initStrategyParams["scanHalfTime"]}T')).strftime('%Y-%m-%d %H:%M:%S')
        actualTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # self._PastPricesArray = self.connector.get_asset_data_hist(symbol=self.EOD, interval='1m', from_=shiftedTime, to=actualTime)
        # self._PastPricesArray = pd.DataFrame(self._PastPricesArray)
        # self._PastPricesArray.to_csv('TESTINGprices.csv')
        self._PastPricesArray = self.connector.get_asset_data_hist(self.SAXO, 1, int(min(self._initStrategyParams["scanHalfTime"], 1200)))
        self._PastPricesArray = pd.DataFrame(self._PastPricesArray)
        self._PastPricesArray.to_csv('TESTINGprices.csv')
        # self._PastPricesArray = list(self._PastPricesArray.apply(lambda x: round((x.OpenBid + x["OpenAsk"]) / 2, 3), axis=1).values)
        self._PastPricesArray = list(self._PastPricesArray.apply(lambda x: x["OpenBid"], axis=1).values)
        # self._PastPricesArray = pd.read_csv('TESTINGprices.csv')
        # self._PastPricesArray = list(self._PastPricesArray.open.values)
        print(f'Successfully downloaded last {self.strategyParams["scanHalfTime"]} dotes')
        del shiftedTime, actualTime

    def _collect_new_price(self):
        newPrice = self.connector.get_actual_data([self.SAXO], mode='bidPrice')
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
        if (half_time > self.strategyParams['scanHalfTime']) or (half_time < 2):
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

        workingArray = self._PastPricesArray[-int(half_time):]
        bandMean = np.mean(workingArray)
        bandStd = np.std(workingArray)

        lowBand = round(bandMean - bandStd * self.strategyParams['yThreshold'], 3)
        highBand = round(bandMean + bandStd * self.strategyParams['yThreshold'], 3)
        if DEBUG:
            # print(self._PastPricesArray[-half_time:])
            # print(f"actualPrice:{workingArray[-1]} highBand:{highBand} lowBand:{lowBand} scanHalfTime:{half_time}")
            pass
        lastPrice = pd.Series(self.connector.get_asset_data_hist(self.SAXO, 1, 1)[0])
        # if (workingArray[-2] > lowBand) and (workingArray[-1] < lowBand):
        if (lastPrice['OpenBid'] > lowBand) and (lastPrice['CloseBid'] < lowBand) and (workingArray[-1] < lowBand):
            logTuple = self._PastPricesArray[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)

            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict['typeOperation'] = 'BUY'
                openDict['position'] = int(round(self._tradeCapital / lowBand, 3))
                openDict['openPrice'] = lowBand
                openDict['openTime'] = self.timer.elapsed()
                openDict['stopLossBorder'] = round(lowBand - self.strategyParams['stopLossStdMultiplier'] * bandStd, 3)
                openDict['takeProfitBorder'] = round(lowBand +
                                                     self.strategyParams['takeProfitStdMultiplier'] * bandStd, 3)

                return openDict

        # if (workingArray[-2] < highBand) and (workingArray[-1] > highBand):
        if (lastPrice['CloseBid'] > highBand) and (lastPrice['OpenBid'] < highBand) and (workingArray[-1] > highBand):
            logTuple = self._PastPricesArray[-(int(self.strategyParams['varianceLookBack']) + 1):]
            retTuple = np.diff(logTuple)
            logTuple = logTuple[1:]
            assert len(retTuple) == len(logTuple)

            if variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=self.strategyParams):
                openDict['typeOperation'] = 'SELL'
                openDict['position'] = int(-1 * round(self._tradeCapital / highBand, 3))
                openDict['openPrice'] = highBand
                openDict['openTime'] = self.timer.elapsed()
                openDict['stopLossBorder'] = round(highBand - self.strategyParams['stopLossStdMultiplier'] * bandStd, 3)
                openDict['takeProfitBorder'] = round(highBand +
                                                     self.strategyParams['takeProfitStdMultiplier'] * bandStd, 3)

                return openDict

        return False

    def _close_trade_ability(self):

        if (self.tradingTimer.elapsed() // 60) > self.strategyParams['timeBarrier']:
            return {'typeHolding': 'endPeriod', 'closePrice': self._PastPricesArray[-1]}

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
                if (self._PastPricesArray[-1] > bandMean) and (self._PastPricesArray[-2] < bandMean):
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
                if (self._PastPricesArray[-1] < bandMean) and (self._PastPricesArray[-2] > bandMean):
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
        COMPLETE = False
        while (not self._inPosition) and (not COMPLETE):
            self._collect_new_price()
            if DEBUG:
                # print('WaitOpenLast:', self._PastPricesArray[-1])
                # print('WaitOpenLast:', self._PastPricesArray[-10:])
                pass
            openAbility = self._open_trade_ability()
            if isinstance(openAbility, dict):
                if DEBUG:
                    print('=====')
                    print(openAbility)
                    print('=====')
                # self.connector.place_order({self.SAXO: openAbility['position']})
                orderID = self.connector.place_order({self.SAXO: openAbility['position']}, order_type='limit',
                                                     order_price=openAbility['openPrice'])['OrderId']
                # print(self.connector.get_actual_data([self.SAXO], mode='all'))
                orderIDminute = datetime.datetime.now().minute
                while (orderIDminute == datetime.datetime.now().minute) and (not self._inPosition):
                    time.sleep(1)
                    orderStatus = self.connector.check_order(orderID)
                    if not orderStatus:
                        self._inPosition = True
                        COMPLETE = True
                        self.tradingTimer.start()
                if not COMPLETE:
                    self.connector.cancelOrder(orderID)
            time.sleep(self.time_interval)

        self._positionDetails = openAbility
        # REPLACE
        self.waitingToFatMean = False
        COMPLETE = False
        while (self._inPosition) and (not COMPLETE):
            self._collect_new_price()
            if DEBUG:
                # print('WaitCloseLast:', self._PastPricesArray[-1])
                pass
            closeAbility = self._close_trade_ability()
            if isinstance(closeAbility, dict):
                if DEBUG:
                    print('====')
                    print(closeAbility)
                    print('====')
                # self.connector.place_order({self.SAXO: -1 * openAbility['position']})
                orderID = self.connector.place_order({self.SAXO: -1 * openAbility['position']}, order_type='limit',
                                                     order_price=closeAbility['closePrice'])['OrderId']
                orderIDminute = datetime.datetime.now().minute
                while (orderIDminute == datetime.datetime.now().minute) and (self._inPosition is True):
                    time.sleep(1)
                    orderStatus = self.connector.check_order(orderID)
                    if not orderStatus:
                        self._inPosition = False
                        COMPLETE = True
                        self.tradingTimer.stop()
                        break
                if not COMPLETE:
                    self.connector.cancelOrder(orderID)
            time.sleep(self.time_interval)

        _stat = {**openAbility, **closeAbility}
        _stat['StrategyWorkingTime'] = self.timer.elapsed()
        _stat['DateTime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.statCollector.add_trade_line(_stat)
        time.sleep(self.time_interval)


    def start_tradingCycle(self):
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
            continue

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

monkeyRobot.timerToken = tokenTimer

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

DEBUG = False

monkeyRobot.start_tradingCycle()

# monkeyRobot.connector.place_order({monkeyRobot.SAXO: 100_000}, order_type='market')
# monkeyRobot.connector.get_actual_data(['CHFJPY'])
# monkeyRobot.connector.stop_order(ticker='CHFJPY', amount=400000)
