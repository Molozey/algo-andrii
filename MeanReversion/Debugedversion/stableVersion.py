import platform
import sys
import os
import threading
import matplotlib.pylab as plt


import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid
from sklearn import linear_model


global systemDivide
global DEBUGMODE
global ESTIMATORSIMPLIFIER
global OPTIMIZESIMPLIFIER

global RecursionBorder

if platform.platform().split('-')[0] == 'macOS':
    systemDivide = '/'
else:
    systemDivide = '\\'


threading.stack_size(2**27)
sys.setrecursionlimit(10 ** 5)

pairName = 'CHFJPY.csv'

inpData = pd.read_csv(f"../testData{systemDivide}{pairName}", index_col=1)
# Какие колонки нужны для работы
columns = ['open', 'high', 'low', 'close']
inpData = inpData[columns]
inpData.index = pd.to_datetime(inpData.index)


def create_strategy_config(params):
    """
    Создает удобную сетку для дальнейших расчетов
    :param params: начальные init параметры
    :return: словарь из параметров использующийся везде
    """
    capital = 20_000
    slippage = 2
    retParams = {
        # Капитал
        'capital': capital,
        'slippage': slippage,
        # Можно использовать для стоп лоссов и тейков с учетом слипэджа
        'slippagePerCapital': slippage / capital,
        # То какой размах мы будем брать для построения полос Боллинджера. Это и есть X Threshold из файла Евгения
        'yThreshold': round(params['yThreshold'] / 100, 2),
        # Период за который мы строим малую скользяшку
        'rollingMean': None,
        # Период за который мы строим большую скользяшку
        'fatRollingMean': None,
        # Временной барьер, Максимальное время сколько мы можем держать позицию
        'timeBarrier': None,
        # Параметр для определения того что данные MeanReversion/TrendFollowing
        # Используется в VRratio тестах для открытия/удержания позиции
        'varianceRatioFilter': params['varianceRatioFilter'],
        'reverseVarianceRatioFilter': params['reverseVarianceRatioFilter'],
        # Сколько времени мы не торгуем после срабатывания стоп лосса
        'restAfterLoss': params['restAfterLoss'],
        # Сколько времени мы не торгуем после закрытия позиции о большую скользяшку
        'restAfterFatProfit': params['restAfterFatProfit'],
        # Процент стоп лосса
        'stopLossStdMultiplier': round(params['stopLossStdMultiplier'] / 100, 3),
        # Процент тэйк профита
        'takeProfitStdMultiplier': round(params['takeProfitStdMultiplier'] / 100, 3),
        # Нужно чтобы пересчитывать VR границы в автоматическом режиме
        'varianceRatioCarreteParameter': params['varianceRatioCarreteParameter'],
        # Тот период за который мы будем считать Variance Ratio. Те ставя тут к примеру 1500,
        # мы должны будем передавать в функцию
        # VR Ratio 1500 точек данных. Сейчас этот гипермараметр связан с гиперпараметров периода малой скользяшки
        'varianceLookBack': None,
        # Чему будет равен временной лаг Q; Q = varianceLookBack // PARAM + 1
        'varianceRatioCarrete': None,
        # Параметр по которому мы будем искать период полураспада
        'scanHalfTime': int(params['scanHalfTime']),
        #
        'halfToFat': params['halfToFat'],
        #
        'halfToLight': params['halfToLight'],
        #
        'halfToTime': params['halfToTime'],

    }
    return retParams


def cook_data(df: pd.DataFrame, params) -> pd.DataFrame:
    """
    Подготавливает данные для дальнейшего рассчета стратегии.
    Передается на одну точку влева данных больше
    :param df: должен быть open, close, low, high
    :param params: список параметров из create_grid
    :return: данные с полосами Болинджера + логарифмы + возвраты
    """
    df['rollMean'] = df['open'].rolling(window=params['rollingMean']).mean()
    df['rollingStd'] = df['open'].rolling(window=params['rollingMean']).std()
    df['fatMean'] = df['open'].rolling(window=params['fatRollingMean']).mean()
    df['logOpenPrice'] = np.log(df['open'])
    df['retOpenPrice'] = df['logOpenPrice'].diff()
    return df.iloc[1:]


# [1, 2, 3, 4, 5, 6]
# [6, 1, 2, 3, 4, 5]
#
# DELTA
# [-, 1, 1, 1, 1, 1] = f(x)
# x = [6, 1, 2, 3, 4, 5]
# f(x) = ax + b
# a = ?
def get_half_time(openTuple: pd.Series) -> float:
    """
    Функция отдающая период полураспада
    :param openTuple:
    :return:
    [MEAN REVERSION | MEAN REVERSION | TREND FOLLOOWING | MEAB REVERSIOn | TREND FOLLOWNING]
    [MEAN REVERSION | TREND FOLLOOWING | MEAB REVERSIOn | TREND FOLLOWNING]
    [OOWING | MEAB REVERSIOn | TREND FOLLOWNING]
    [IOn | TREND FOLLOWNING]
    """
    df_open = openTuple.to_frame()
    df_lag = df_open.shift(1)
    df_delta = df_open - df_lag
    linear_regression_model = linear_model.LinearRegression()
    df_delta = df_delta.values.reshape(len(df_delta), 1)
    df_lag = df_lag.values.reshape(len(df_lag), 1)
    linear_regression_model.fit(df_lag[1:], df_delta[1:])
    half_life = -np.log(2) / linear_regression_model.coef_.item()
    return half_life


def variance_ratio(logTuple: tuple, retTuple: tuple, params: dict) -> bool:
    """
    Функция для open. Здесь лаг q зависит только от гиперпараметра
    Возвращает значение variacne ratio. Необходимо для понимания того, можно ли открывать сделку
    :param logTuple: tuple из цен открытия включая проверяемую точку
    :param retTuple: tuple из цен открытия включая проверяемую точку
    :param params: список параметров из create_grid
    :return: Можно ли открывать сделку. Фактически является фильтром
    """
    buffer_size = len(retTuple)
    means = (1 / buffer_size) * np.sum(retTuple)
    # сдвиг во времени q
    m = params['varianceRatioCarrete'] * (buffer_size - params['varianceRatioCarrete'] + 1) \
        * (1 - (params['varianceRatioCarrete'] / buffer_size))
    sigma_a = (1 / (buffer_size - 1)) * np.sum(np.square(np.subtract(retTuple, means)))
    subtract_returns = np.subtract(logTuple,
                                   np.roll(logTuple, params['varianceRatioCarrete']))[params['varianceRatioCarrete']:]
    _buff_ = np.sum(np.square(subtract_returns - params['varianceRatioCarrete'] * means))
    try:
        sigma_b = (1 / m) * _buff_
    except ZeroDivisionError:
        print('Warning at variance ratio. Division on zero')
        return False

    result = (sigma_b / sigma_a)
    if result < params['varianceRatioFilter']:
        return True
    else:
        return False


def reverse_variance_ratio(preComputed, params: dict, timeBorderCounter: int, VRstatement=False) -> bool:
    """
    Возвращает значение variance ratio. Необходимо для понимания того, можно ли открывать сделку
    :param preComputed: Заранее просчитанные логарифмы и возвраты
    :param params: список параметров из create_grid
    :param timeBorderCounter: Штука показывающая сколько мы находимся в сделке
    :param VRstatement: Тип того для чего мы проверяем варианс рейшу, для входа в режим поиска
    False большой скользяшке, или True для возврата к малой скользяшке

    :return: Можно ли открывать сделку. Фактически является фильтром
    """
    try:
        retTuple = preComputed["retOpenPrice"]
        logTuple = preComputed["logOpenPrice"]
        if timeBorderCounter < params["varianceLookBack"]:
            buffer_size = len(retTuple)
            means = (1 / buffer_size) * np.sum(retTuple)
            # сдвиг во времени q
            m = timeBorderCounter * (buffer_size - timeBorderCounter + 1) * (1 - (timeBorderCounter / buffer_size))
            sigma_a = (1 / (buffer_size - 1)) * np.sum(np.square(np.subtract(retTuple, means)))
            subtract_returns = np.subtract(logTuple, np.roll(logTuple, timeBorderCounter))[timeBorderCounter:]
            _buff_ = np.sum(np.square(subtract_returns - timeBorderCounter * means))
            sigma_b = (1 / m) * _buff_
            result = (sigma_b / sigma_a)
            if not VRstatement:
                if result > params['reverseVarianceRatioFilter']:
                    return True
                else:
                    return False
            if VRstatement:
                if result < params['varianceRatioFilter']:
                    return True
                else:
                    return False
        else:
            return False

    except ZeroDivisionError:
        print('Warning at variance ratio. Division on zero')
        return False


def calculate_max_drawdown(PNL_SERIES, dollars=True):
    """
    solution by Marco de Prado
    :param PNL_SERIES:
    :param dollars:
    :return:
    """
    dropout_df = PNL_SERIES.to_frame('pnl')
    dropout_df['hwm'] = dropout_df.expanding().max()
    df0 = dropout_df.groupby('hwm').min().reset_index()
    df0.columns = ['hwm', 'min']
    df0 = df0[df0['hwm'] > df0['min']]
    if dollars:
        dd = df0['hwm'] - df0['min']
    else:
        dd = df0['min'] / df0['hwm']

    return max(dd)


def open_position(position, dataFrame, params, reqCounter, preComputed):
    TRIGGER = False
    localParams = params.copy()
    openDict = {
        'typeOperation': None,
        'position': None,
        'openPrice': None,
        'openIndex': None,
        'stopLossBorder': None,
        'takeProfitBorder': None
    }

    if (reqCounter >= RecursionBorder) or (position == dataFrame.shape[0] - 3):
        return position

    # Получить время полураспада для генерации параметров сделки
    half_time = int(get_half_time(dataFrame.close[position-params['scanHalfTime']:position]))
    if (half_time > params['scanHalfTime']) or (half_time < 0):
        return open_position(position=position+1, dataFrame=dataFrame, params=params,
                             reqCounter=reqCounter+1, preComputed=preComputed)

    localParams["rollingMean"] = int(half_time * params['halfToLight'])
    localParams["fatRollingMean"] = int(params['halfToFat'] * half_time)
    localParams["timeBarrier"] = int(half_time * params['halfToTime'])
    if localParams["timeBarrier"] <= 0:
        localParams["timeBarrier"] = 1

    localParams["varianceLookBack"] = int(half_time * params['halfToFat'])
    localParams["varianceRatioCarrete"] = int((half_time *
                                               params['halfToFat']) // params['varianceRatioCarreteParameter']) + 1
    # Считаем локальные барьеры открытия сделки
    # print(f"HalfTime={half_time}; Position={position}, DF shape={dataFrame.shape}, RollMean={localParams['rollingMean']}")
    bands_roll = dataFrame.open.iloc[position-localParams["rollingMean"]:position+1].rolling(localParams["rollingMean"])
    bands_mean = bands_roll.mean().iloc[-1]
    bands_std = bands_roll.std().iloc[-1]
    # print(bands_roll.mean())
    # print(bands_roll.mean())
    # print(f"BandsMean={bands_mean}; BandsStd={bands_std}")
    low_band = round(bands_mean - bands_std * params['yThreshold'], 3)
    high_band = round(bands_mean + bands_std * params['yThreshold'], 3)
    # print(f"BL{low_band}-DOT{dataFrame.open[position]}-BH{high_band}")
    # print("Actual DOT:", dataFrame.iloc[position])
    if (dataFrame.open[position] > low_band) and (dataFrame.low[position] < low_band):
        # Если это так, то убеждаемся что можем открыть сделку проводя тест
        # VR_RATIO (LOOKBACK={HYPERPARAMETER}, time_laq===q={HYPERPARAMETER})
        if variance_ratio(logTuple=preComputed["logTuple"][position - localParams['varianceLookBack']:position],
                          retTuple=preComputed["retTuple"][position - localParams['varianceLookBack']:position],
                          params=localParams):
            # Формируем удобочитаемый тип return функции
            openDict['typeOperation'] = 'BUY'
            openDict['position'] = round(params['capital'] / low_band, 3)
            openDict['openPrice'] = low_band
            openDict['openIndex'] = position
            openDict['stopLossBorder'] = round(low_band - params['stopLossStdMultiplier'] * bands_std, 3)
            openDict['takeProfitBorder'] = round(low_band + params['takeProfitStdMultiplier'] * bands_std, 3)
            TRIGGER = True

            return {'openDict': openDict, 'params': localParams}

    elif (dataFrame.open[position] < high_band) and (dataFrame.high[position] > high_band):
        # Если это так, то убеждаемся что можем открыть сделку проводя тест
        # VR_RATIO (LOOKBACK={HYPERPARAMETER}, time_laq===q={HYPERPARAMETER})
        if variance_ratio(logTuple=preComputed["logTuple"][position - localParams['varianceLookBack']:position],
                          retTuple=preComputed["retTuple"][position - localParams['varianceLookBack']:position],
                          params=localParams):
            # Формируем удобочитаемый тип return функции
            openDict['typeOperation'] = 'SELL'
            openDict['position'] = round(-1 * (params['capital'] / high_band), 3)
            openDict['openPrice'] = high_band
            openDict['openIndex'] = position
            openDict['stopLossBorder'] = round(high_band + params['stopLossStdMultiplier'] * bands_std, 3)
            openDict['takeProfitBorder'] = round(high_band - params['takeProfitStdMultiplier'] * bands_std, 3)
            TRIGGER = True

            return {'openDict': openDict, 'params': localParams}

    # В случае, если сделку открыть не получилось, переходим к следующей точке.
    # Вывод - пока что сделку не получилось
    # В real-time это является аналогом ожидания до появления следующих данных и повторения
    # проверки на открытие уже на них
    if not TRIGGER:
        return open_position(position=position+1, dataFrame=dataFrame, params=params, reqCounter=reqCounter+1,
                             preComputed=preComputed)


def close_position(position, openDict, dataFrame, localParams, reqCounter, preComputed, borderCounter, indicatorVR):
    TRIGGER = False
    # print('=======')
    # print(preComputed.iloc[position])
    # print('------')
    # print(dataFrame.iloc[position])
    # print('=======')

    if (reqCounter > RecursionBorder) or (position == dataFrame.shape[0] - 3):
        return [position, localParams, borderCounter, indicatorVR, openDict]

    if borderCounter == localParams['timeBarrier']:
        return {'typeHolding': 'endPeriod', 'closePrice': dataFrame.open[position + 1],
                'closeIndex': position + 1}

    elif openDict['typeOperation'] == 'BUY':
        # Стоп лосс условие
        if (dataFrame.open[position] > openDict['stopLossBorder']) and (dataFrame.low[position] <
                                                                        openDict['stopLossBorder']):
            return {'typeHolding': 'stopLoss', 'closePrice': openDict['stopLossBorder'],
                    'closeIndex': position}

        if openDict['typeOperation'] == 'BUY':
            delta = dataFrame.open[position] - dataFrame.open[position - 1]
            if delta > 0:
                openDict['stopLossBorder'] = round(openDict['stopLossBorder'] + delta, 3)

        if dataFrame.open[position] < openDict['stopLossBorder']:
            return {'typeHolding': 'stopLoss', 'closePrice': dataFrame.open[position],
                    'closeIndex': position}
        # Smart mean crossing
        # Проверяем адекватное расположение между открытием и скользящей малой.
        # Причина этого такая же как для open_position
        # Читай в чем суть - выше!
        if (dataFrame.open[position] < preComputed["rollMean"][position]) and (not indicatorVR):
            # Проверяем что можно закрыть лонг о пересечение с малой скользящей
            if dataFrame.high[position] > preComputed["rollMean"][position]:

                # Проверяем можно ли продолжить удержание позиции. Делаем это через VariacneRatio за какой-то период
                # arrowIndex - params['varianceLookBack'], где varianceLookBack - гиперпараметр стратегии.
                # Временной лаг Q для VARIANCE_RATIO определяется как время что мы находимся в позиции
                # Если VarianceRatio показывает что данные стали TrendFollowing - мы меняем режим
                # стратегии на попытку закрыться
                # О скользящее среднее за больший период. Так мы получим большую прибыль
                if preComputed["rollMean"][position] < preComputed["fatMean"][position]:
                    if reverse_variance_ratio(preComputed=preComputed,
                                              params=localParams, timeBorderCounter=borderCounter + 1):
                        #   Local Trend Following recursion
                        return close_position(position=position + 1, dataFrame=dataFrame, reqCounter=reqCounter + 1,
                                              openDict=openDict, localParams=localParams, preComputed=preComputed,
                                              borderCounter=borderCounter + 1, indicatorVR=True)

                    else:
                        # Если VR RATIO не показал возможность попытаться
                        # закрыться о большую скользяшку, то закрываемся о малую
                        return {'typeHolding': 'lightCross', 'closePrice': preComputed["rollMean"][position],
                                'closeIndex': position}
                        pass

                else:
                    # Может быть так, что скольщяшее среднее за больший период находится выше чем
                    # скользящее за малый период
                    # Если учесть что мы торгуем на тех инструментах что показывают большую склонность
                    # к meanReversion такая ситуация говорит нам что нужно как можно скорей сбрасывать позицию.
                    return {'typeHolding': 'lightCrossEmergent', 'closePrice': preComputed["rollMean"][position],
                            'closeIndex': position}

        # Определяет режим работы в случае альтернативной стратегии на которую мы переключаеся
        # в случае выполнения каких-то условий
        if indicatorVR:
            # Аналогично тому что было раньше
            if dataFrame.open[position] < preComputed["fatMean"][position]:
                # Аналогично тому что было раньше
                if dataFrame.high[position] > preComputed["fatMean"][position]:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': preComputed["fatMean"][position],
                            'closeIndex': position}
                # Даем возможность переключить режим стратегии снова на нулевой.
                # Делаем это если данные снова начали показывать склонность к meanReversion
            if not reverse_variance_ratio(preComputed=preComputed,
                                          params=localParams, timeBorderCounter=borderCounter + 1,
                                          VRstatement=True):
                #   Local Trend Following recursion
                return close_position(position=position + 1, dataFrame=dataFrame, reqCounter=reqCounter + 1,
                                      openDict=openDict, localParams=localParams, preComputed=preComputed,
                                      borderCounter=borderCounter + 1, indicatorVR=False)

    elif openDict['typeOperation'] == 'SELL':
        # Стоп лосс условие
        if (dataFrame.open[position] < openDict['stopLossBorder']) and (dataFrame.high[position] >
                                                                        openDict['stopLossBorder']):
            return {'typeHolding': 'stopLoss', 'closePrice': openDict['stopLossBorder'],
                    'closeIndex': position}

        if openDict['typeOperation'] == 'SELL':
            delta = dataFrame.open[position] - dataFrame.open[position - 1]
            if delta < 0:
                openDict['stopLossBorder'] = round(openDict['stopLossBorder'] - delta, 3)

        if dataFrame.open[position] > openDict['stopLossBorder']:
            return {'typeHolding': 'stopLoss', 'closePrice': dataFrame.open[position],
                    'closeIndex': position}
        # Smart mean crossing
        # Проверяем адекватное расположение между открытием и скользящей малой.
        # Причина этого такая же как для open_position. Читай в чем суть - выше!

        if (dataFrame.open[position] > preComputed["rollMean"][position]) and (not indicatorVR):
            # Проверяем что можно закрыть лонг о пересечение с малой скользящей
            if dataFrame.low[position] < preComputed["rollMean"][position]:
                # Проверяем можно ли продолжить удержание позиции. Делаем это через VariacneRatio за какой-то период
                # arrowIndex - params['varianceLookBack'], где varianceLookBack - гиперпараметр стратегии.
                # Временной лаг Q для VARIANCE_RATIO определяется как время что мы находимся в позиции
                # Если VarianceRatio показывает что данные стали TrendFollowing -
                # мы меняем режим стратегии на попытку закрыться
                # О скользящее среднее за больший период. Так мы получим большую прибыль
                if preComputed["rollMean"][position] > preComputed["fatMean"][position]:
                    if reverse_variance_ratio(preComputed=preComputed,
                                              params=localParams, timeBorderCounter=borderCounter + 1,
                                              VRstatement=False):
                        #   Local Trend Following recursion
                        return close_position(position=position + 1, dataFrame=dataFrame, reqCounter=reqCounter + 1,
                                              openDict=openDict, localParams=localParams, preComputed=preComputed,
                                              borderCounter=borderCounter + 1, indicatorVR=True)

                    else:
                        # Если VR RATIO не показал возможность попытаться закрыться о
                        # большую скользяшку, то закрываемся о малую
                        return {'typeHolding': 'lightCross', 'closePrice': preComputed["rollMean"][position],
                                'closeIndex': position}
                        pass

                else:
                    # Может быть так, что скольщяшее среднее за больший период находится
                    # выше чем скользящее за малый период
                    # Если учесть что мы торгуем на тех инструментах что показывают большую
                    # склонность к meanReversion
                    # такая ситуация говорит нам что нужно как можно скорей сбрасывать позицию.
                    return {'typeHolding': 'lightCrossEmergent', 'closePrice': preComputed["rollMean"][position],
                            'closeIndex': position}

        # Определяет режим работы в случае альтернативной стратегии на которую мы переключаеся в случае
        # выполнения каких-то условий
        if indicatorVR:
            # Аналогично тому что было раньше
            if dataFrame.open[position] > preComputed["fatMean"][position]:
                # Аналогично тому что было раньше
                if dataFrame.low[position] < preComputed["fatMean"][position]:
                    return {'typeHolding': 'fatExtraProfit', 'closePrice': preComputed["fatMean"][position],
                            'closeIndex': position}
                # Даем возможность переключить режим стратегии снова на нулевой.
                # Делаем это если данные снова начали показывать склонность к meanReversion
            if not reverse_variance_ratio(preComputed=preComputed,
                                          params=localParams, timeBorderCounter=borderCounter + 1,
                                          VRstatement=True):
                #   Local Trend Following recursion
                return close_position(position=position + 1, dataFrame=dataFrame, reqCounter=reqCounter + 1,
                                      openDict=openDict, localParams=localParams, preComputed=preComputed,
                                      borderCounter=borderCounter + 1, indicatorVR=False)

    if not TRIGGER:
        # Trailing StopLoss
        return close_position(position=position + 1, dataFrame=dataFrame, reqCounter=reqCounter + 1,
                              openDict=openDict, localParams=localParams, preComputed=preComputed,
                              borderCounter=borderCounter + 1, indicatorVR=indicatorVR)



def _collectTrades(initPOS, SL, coll_DATAFRAME, leftShift, rightShift, openShift, Collparameters):
    statistics = list()
    POS = initPOS
    # print(pd.Series(Collparameters))
    while POS < SL:
        # print(f"{POS} of {SL}")
        # print(f"===========POS={POS}==============")
        preComputed = {'logTuple': np.log(coll_DATAFRAME.open.iloc[POS - leftShift - 1: POS + rightShift].copy())}
        preComputed['retTuple'] = preComputed['logTuple'].diff()[1:]
        preComputed['logTuple'] = preComputed['logTuple'][1:]

        openPosition = 1
        while not isinstance(openPosition, dict):
            if POS > SL:
                return statistics
            # ИЗМЕНЕНИЕ НЕПРОВЕРЕНО: В ПРАВУЮ ГРАНИЦУ ДОБАВЛЕНА + 1
            # !!!!!!!!!!!!!!!!!
            openPosition = open_position(position=openShift,
                                         dataFrame=coll_DATAFRAME.iloc[POS - leftShift: POS + rightShift + 1],
                                         params=Collparameters, reqCounter=0, preComputed=preComputed)
            if not isinstance(openPosition, dict):
                delta = openPosition - openShift
                POS += delta
                if POS > SL:
                    return statistics
                preComputed = {'logTuple': np.log(coll_DATAFRAME.open.iloc[POS - leftShift - 1:
                                                                       POS + rightShift].copy())}
                preComputed['retTuple'] = preComputed['logTuple'].diff()[1:]
                preComputed['logTuple'] = preComputed['logTuple'][1:]


        openDict = openPosition['openDict']
        # order = tie_account_to_order(AccountKey, MarketOrderFxSpot(Uic=19, Amount=openDict['position']))

        openDict['openIndex'] = openDict['openIndex'] - openShift + POS
        POS = openDict['openIndex'] + 1

        localParameters = openPosition['params'].copy()
        closeShift = int(max(localParameters['varianceLookBack'], localParameters['fatRollingMean'])) + 1
        closeLeft = int(max(localParameters['varianceLookBack'], localParameters['fatRollingMean'])) + 1
        closeRight = int(min(RecursionBorder, localParameters['timeBarrier'])) + 1

        preComputedClose = cook_data(coll_DATAFRAME.iloc[POS - 1 - closeLeft: POS + closeRight].copy(),
                                     params=localParameters).copy()

        indicatorVR = False
        borderCounter = 0
        closePosition = 1
        # print(openDict)
        while not isinstance(closePosition, dict):
            if POS > SL:
                return statistics

            closePosition = close_position(position=closeShift, openDict=openDict,
                                           dataFrame=coll_DATAFRAME.iloc[POS - closeLeft: POS + closeRight].copy(),
                                           localParams=localParameters, reqCounter=0,
                                           preComputed=preComputedClose, borderCounter=borderCounter,
                                           indicatorVR=indicatorVR)

            if not isinstance(closePosition, dict):
                delta = closePosition[0] - closeShift
                POS += delta
                if POS > SL:
                    return statistics
                preComputedClose = cook_data(coll_DATAFRAME.iloc[POS - 1 - closeLeft: POS + closeRight].copy(),
                                             params=localParameters).copy()
                localParameters = closePosition[1]
                indicatorVR = closePosition[3]
                borderCounter = closePosition[2]
                openDict = closePosition[4]


        closeDict = closePosition
        # print(closeDict)
        closeDict['closeIndex'] = closeDict['closeIndex'] - closeShift + POS
        POS = closeDict['closeIndex'] + 1

        # order = tie_account_to_order(AccountKey, StopOrderFxSpot(Uic=19, Amount=openDict['position'],
        #                                                          OrderPrice=closeDict['closePrice']))

        if closeDict['typeHolding'] == 'stopLoss':
            POS += int(localParameters['restAfterLoss'])

        statistics.append({**openDict, **closeDict})

    return statistics


def _estimator(_DATAFRAME, _gridParams: dict):
    SL = int(_DATAFRAME.shape[0] // ESTIMATORSIMPLIFIER)
    Estparameters = _gridParams.copy()
    openShift = int(max(Estparameters['scanHalfTime'], int(Estparameters['scanHalfTime'] * Estparameters['halfToFat']))) + 1
    leftShift = int(max(Estparameters['scanHalfTime'], int(Estparameters['scanHalfTime'] * Estparameters['halfToFat']))) + 1
    # ИЗМЕНЕНИЕ НЕПРОВЕРЕНО: В ПРАВУЮ ГРАНИЦУ ДОБАВЛЕНА + 1
    # !!!!!!!!!!!!!!!!!
    rightShift = int(min(RecursionBorder, Estparameters['scanHalfTime'] * Estparameters['halfToTime'])) + 1
    POS = openShift + 1
    statistics = _collectTrades(initPOS=POS, SL=SL, coll_DATAFRAME=_DATAFRAME, leftShift=leftShift,
                                rightShift=rightShift, openShift=openShift, Collparameters=Estparameters)

    if len(statistics) != 0:
        retDF = pd.DataFrame(statistics)
        retDF['profit'] = (retDF["position"] * (retDF["closePrice"] -
                                                retDF["openPrice"]) - Estparameters['slippage']
                           if (retDF["typeOperation"] == 'BUY').bool
                           else abs(retDF["position"]) * (retDF["openPrice"]
                                                          - retDF["closePrice"]) - Estparameters['slippage'])
        retDF.index = retDF.openIndex
        stepDF = pd.DataFrame(index=pd.RangeIndex(min(retDF.openIndex), max(retDF.openIndex)))
        stepPnl = stepDF.merge(retDF, left_index=True, right_index=True, how='outer').profit.replace(np.nan, 0).cumsum()
        del stepDF
        TPNL = stepPnl.iloc[-1]
        try:
            PNLDD = TPNL / calculate_max_drawdown(stepPnl)
        except (ZeroDivisionError, ValueError):
            if DEBUGMODE:
                print('ValueError 585')
                print(stepPnl)
            PNLDD = -1

        totalMetric = pd.Series({**Estparameters, 'PNLDD': PNLDD, 'TotalPNL': TPNL})
        if DEBUGMODE:
            _nill = pd.DataFrame(statistics)
            # print('Operations inside:', np.unique(_nill.typeHolding, return_counts=True))

        return pd.DataFrame(statistics), totalMetric
    else:
        if DEBUGMODE:
            print('Empty Statistics')
            # print(_DATAFRAME)
            # print(pd.Series(_gridParams))
        statistics = pd.DataFrame()
        totalMetric = pd.Series({**_gridParams.copy(), 'PNLDD': -1, 'TotalPNL': -100_000})
        return statistics, totalMetric


def strategy_real_time_optimize(realTimeData, parameters, savePath: str, show=True, update=False):
    JOI_PARAMETER = 15
    SL = int(realTimeData.shape[0] // OPTIMIZESIMPLIFIER)
    paramsEvolution = list()
    RealTimeOptimizeTrades = list()

    realTimeData = realTimeData.copy()
    # Время сколько торгуем
    _UPDATE_TIME = pd.Timedelta('1w')
    # Время за сколько оптимизируемся
    _TRADE_TIME = pd.Timedelta('2w')
    _UPDATE_TIME //= '1T'
    _TRADE_TIME //= '1T'
    SL = (SL // _TRADE_TIME) * _TRADE_TIME
    POSITION = _UPDATE_TIME
    if show:
        tqdm_bar = tqdm(total=SL, leave=False)
    optimalParams = parameters.copy()

    if POSITION >= SL:
        if DEBUGMODE:
            print('POSITION > SL at first step. Try to load more data or reduce optimization period')
        return None, None, [[None, None], [None, None], [None, None]]
    while POSITION < SL:
        if POSITION > SL:
           #  Проверка на границу
           break

        paramsEvolution.append([POSITION, optimalParams])
        optimizing_grid = {
            # Оптимизировать !!!
            'yThreshold': np.unique(np.linspace(optimalParams["yThreshold"] * 100 * 0.95,
                                                optimalParams["yThreshold"] * 100 * 1.05, num=4)),
            # Оптимизировать !
            'varianceRatioFilter': np.unique([optimalParams["varianceRatioFilter"]]),
            'reverseVarianceRatioFilter': np.unique([optimalParams["reverseVarianceRatioFilter"]]),
            # Оптимизировать !!
            'restAfterLoss': [optimalParams["restAfterLoss"]],
            # Оптимизировать !
            'restAfterFatProfit': [optimalParams['restAfterFatProfit']],
            # Оптимизировать !!
            'stopLossStdMultiplier': np.unique(
                [int(x) for x in np.linspace(optimalParams["stopLossStdMultiplier"] * 100 * 0.9,
                                             optimalParams["stopLossStdMultiplier"] * 100 * 1.1, num=5)]),

            'takeProfitStdMultiplier': [optimalParams["takeProfitStdMultiplier"] * 100],
            # Оптимизировать !!
            'varianceRatioCarreteParameter': np.unique([z if z != 0 else 1 for z in [int(x) for x in np.linspace(optimalParams["varianceRatioCarreteParameter"] * 0.8,
                                                                                                            optimalParams["varianceRatioCarreteParameter"] * 1.2, num=3)]]),
            # Оптимизировать !!!
            'scanHalfTime': np.unique(
                [z if z != 0 else 1 for z in [int(x) for x in np.linspace(optimalParams["scanHalfTime"] * 0.9,
                                                                          optimalParams["scanHalfTime"] * 1.1,
                                                                          num=3)]]),
            # Оптимизировать 0!
            'halfToFat': np.unique(
                [round(z, 3) if z != 0 else 1 for z in [float(x) for x in np.linspace(optimalParams["halfToFat"] * 0.9,
                                                                                      optimalParams["halfToFat"] * 1.1,
                                                                                      num=3)]]),
            # Оптимизировать 0!
            'halfToLight': np.unique([round(z, 3) if z != 0 else 1 for z in
                                      [float(x) for x in np.linspace(optimalParams["halfToLight"] * 0.9,
                                                                     optimalParams["halfToLight"] * 1.1, num=3)]]),
            # Оптимизировать 0!
            'halfToTime': np.unique([round(z, 3) if z != 0 else 1 for z in
                                     [float(x) for x in np.linspace(optimalParams["halfToTime"] * 0.9,
                                                                    optimalParams["halfToTime"] * 1.1, num=3)]]),
        }

        params_net = pd.DataFrame(ParameterGrid(optimizing_grid)).sample(frac=1, random_state=9).reset_index(drop=True)
        optimizing_step = list()
        statistics, totalMetric = _estimator(realTimeData.iloc[POSITION - _UPDATE_TIME: POSITION].copy(),
                                             _gridParams=create_strategy_config(optimalParams).copy())
        optimizing_step.append(totalMetric)
        if DEBUGMODE:
            print(f"OPTIMIZATION:{POSITION - _UPDATE_TIME}:{POSITION}")

        for _arrowParam in range(0, JOI_PARAMETER):
            if DEBUGMODE:
                print(f"{_arrowParam} of {JOI_PARAMETER - 1}")
            parameters = create_strategy_config(params_net.iloc[_arrowParam]).copy()
            if parameters['halfToFat'] <= parameters['halfToLight']:
                parameters['halfToFat'] = parameters['halfToLight']
            statistics, totalMetric = _estimator(realTimeData.iloc[POSITION - _UPDATE_TIME: POSITION].copy(),
                                                 _gridParams=parameters)

            optimizing_step.append(totalMetric)

        optimalParams = pd.DataFrame(optimizing_step)
        if optimalParams[optimalParams.PNLDD == -1].shape[0] == optimalParams.shape[0]:
            if DEBUGMODE:
                print('All parameters combinations raises Errors. Check your initCondition')
                print('Wait one day to try make new optimization. Shift POSITION.')
                # print(f"Y_threshold = {pd.Series(optimizing_step[0])}")
            optimalParams = optimalParams.sample(frac=1, random_state=9).reset_index(drop=True).iloc[0]
            POSITION += 1000
            continue

        if update:
            if not os.access(f'{savePath}{systemDivide}{POSITION - _UPDATE_TIME}_{POSITION}', os.F_OK):
                os.mkdir(f'{savePath}{systemDivide}{POSITION - _UPDATE_TIME}_{POSITION}')

        if update:
            optimalParams.to_csv(f'{savePath}{systemDivide}{POSITION - _UPDATE_TIME}_{POSITION}{systemDivide}all_joi.csv')

        optimalParams = pd.DataFrame(optimizing_step).sort_values(by='TotalPNL', ascending=False).iloc[0]

        optimalParams = {
        # Оптимизировать !!!
        'yThreshold': optimalParams['yThreshold'] * 100,
        # Оптимизировать !
        # 'varianceRatioFilter': np.linspace(parameters["varianceRatioFilter"] * 0.9, parameters["varianceRatioFilter"] * 1.1, num=3),
        'varianceRatioFilter': optimalParams['varianceRatioFilter'],
        # 'reverseVarianceRatioFilter': np.linspace(parameters["reverseVarianceRatioFilter"] * 0.9, parameters["reverseVarianceRatioFilter"] * 1.1, num=3),
        'reverseVarianceRatioFilter': optimalParams['reverseVarianceRatioFilter'],
        # Оптимизировать !!
        'restAfterLoss': int(optimalParams['restAfterLoss']),
        # Оптимизировать !
        'restAfterFatProfit' : int(optimalParams['restAfterFatProfit']),
        # Оптимизировать !!
        'stopLossStdMultiplier': int(optimalParams['stopLossStdMultiplier'] * 100),
        # Оптимизировать !
        'takeProfitStdMultiplier': int(optimalParams['takeProfitStdMultiplier'] * 100),
        # Оптимизировать !!
        'varianceRatioCarreteParameter': optimalParams['varianceRatioCarreteParameter'],
        # Оптимизировать !!!
        'scanHalfTime': int(optimalParams['scanHalfTime']),
        # Оптимизировать 0!
        'halfToFat': optimalParams['halfToFat'],
        # Оптимизировать 0!
        'halfToLight': optimalParams['halfToLight'],
        # Оптимизировать 0!
        'halfToTime': optimalParams['halfToTime'],
        }

        if update:
            pd.Series(optimalParams).to_csv(f'{savePath}{systemDivide}{POSITION - _UPDATE_TIME}_{POSITION}{systemDivide}best_param.csv')

        optimalParams = create_strategy_config(optimalParams).copy()
        if DEBUGMODE:
            print(f"TRADE:{POSITION}:{POSITION + _TRADE_TIME}")
        POSITIONSHIFTER = int(max(optimalParams['scanHalfTime'], int(optimalParams['scanHalfTime'] * optimalParams['halfToFat']))) + 2

        statistics, totalMetric = _estimator(realTimeData.iloc[POSITION - POSITIONSHIFTER: POSITION + _TRADE_TIME].copy(),
                                             _gridParams=optimalParams)
        if statistics.empty:
            if DEBUGMODE:
                print('Cannot open any trade while _Trade Period. Check parameters evolution')
            POSITION += int(_TRADE_TIME // ESTIMATORSIMPLIFIER)
            continue

        statistics = pd.DataFrame(statistics)
        statistics['openIndex'] = statistics['openIndex'] + POSITION - POSITIONSHIFTER
        statistics['closeIndex'] = statistics['closeIndex'] + POSITION - POSITIONSHIFTER
        if update:
            statistics.groupby(by='typeHolding').describe().to_csv(
                f'{savePath}{systemDivide}{POSITION - _UPDATE_TIME}_{POSITION}{systemDivide}tradingStat.csv')
        POSITION = statistics['closeIndex'].iloc[-1]
        if DEBUGMODE:
            print("Position after trade period:", POSITION)
        RealTimeOptimizeTrades.append(statistics)

    paramsEvolution[0][1]['yThreshold'] *= 100
    paramsEvolution[0][1]['stopLossStdMultiplier'] *= 100
    paramsEvolution[0][1]['takeProfitStdMultiplier'] *= 100
    if update:
        pd.DataFrame([_[1] for _ in paramsEvolution]).to_csv(f"{savePath}{systemDivide}evolution.csv")

    if len(RealTimeOptimizeTrades) == 0:
        if DEBUGMODE:
            print('No trades with this initialConditions')
            print(pd.Series(paramsEvolution[0][1]))
        RealTimeOptimizeTrades = pd.DataFrame()
        PNLDD = -1
        TPNL = -100000
        startParams = paramsEvolution[0][1]
        totalResult = {'PNLDD': PNLDD, 'TPNL': TPNL, **startParams}
        return totalResult, RealTimeOptimizeTrades, paramsEvolution

    initDF = RealTimeOptimizeTrades[0]
    if len(RealTimeOptimizeTrades) > 1:
        for tr in RealTimeOptimizeTrades[1:]:
            initDF = initDF.append(tr, ignore_index=True)

    ret = pd.DataFrame(initDF)
    ret['profit'] = (ret["position"] * (ret["closePrice"] -
                                        ret["openPrice"]) - paramsEvolution[0][1]['slippage']
                     if (ret["typeOperation"] == 'BUY').bool
                     else abs(ret["position"]) * (ret["openPrice"] -
                                                  ret["closePrice"]) - paramsEvolution[0][1]['slippage'])
    ret.index = ret.openIndex
    stepDF = pd.DataFrame(index=pd.RangeIndex(min(ret.openIndex), max(ret.openIndex)))
    stepPnl = stepDF.merge(ret, left_index=True, right_index=True, how='outer').profit.replace(np.nan, 0).cumsum()
    del stepDF
    TPNL = stepPnl.iloc[-1]
    try:
        PNLDD = TPNL / calculate_max_drawdown(stepPnl)
    except (ZeroDivisionError, ValueError):
        if DEBUGMODE:
            print('ValueError 585')
            print(stepPnl)
        PNLDD = -1

    if show:
        print('TOTAL PNL = ', TPNL)
        print('DD DIV TOTAL PNL = ', PNLDD)
        plt.figure(figsize=(12, 5))
        plt.plot(stepPnl)
        plt.title('trade PNL')
        plt.show()
        print('==========' * 20)

    startParams = paramsEvolution[0][1]
    totalResult = {'PNLDD': PNLDD, 'TPNL': TPNL, **startParams}
    if update:
        pd.Series(totalResult).to_csv(f'{savePath}{systemDivide}Condition_result.csv')
        stepPnl.to_csv(f"{savePath}{systemDivide}ConditionPNL.csv")
    return totalResult, RealTimeOptimizeTrades, paramsEvolution


def _get_the_best_strategy(mainOptimizerData, initConditions, numberOfDotes=20, saveInfo=True):
    import shutil
    if os.access(f'backTEST', os.F_OK):
        shutil.rmtree(f'backTEST')

    if not os.access(f'backTEST', os.F_OK):
        os.mkdir(f'backTEST')

    strategyRESULTcollector = list()
    parametersEvolution = list()
    if numberOfDotes < initConditions.shape[0]:
        BORDER = numberOfDotes
    else:
        BORDER = initConditions.shape[0]

    for NUM in tqdm(range(1, BORDER)):
        PATH = f"backTEST{systemDivide}condition_{NUM}"
        if not os.access(f'{PATH}', os.F_OK):
            os.mkdir(f'{PATH}')
        parmNUM = create_strategy_config(initConditions.iloc[NUM].copy()).copy()
        if parmNUM['halfToFat'] < parmNUM['halfToLight']:
            parmNUM['halfToFat'] = parmNUM['halfToLight']
        totalRes, trades, paramsEvo = strategy_real_time_optimize(realTimeData=mainOptimizerData.iloc[:400_000].copy(),
                                                                  savePath=PATH,
                                                                  parameters=parmNUM.copy(), show=False, update=True)
        paramsEvo = [_[1] for _ in paramsEvo]
        paramsEvoSeries = pd.DataFrame(paramsEvo)
        parametersEvolution.append(paramsEvo)
        strategyRESULTcollector.append(totalRes)
    return strategyRESULTcollector, parametersEvolution


firstConditions = {
    # Оптимизировать !!!
    'yThreshold': [150, 250, 300],
    # Оптимизировать !
    'varianceRatioFilter': [1.0],
    'reverseVarianceRatioFilter': [1.0],
    # Оптимизировать !!
    'restAfterLoss': [200, 600],
    # Оптимизировать !
    'restAfterFatProfit': [1],
    # Оптимизировать !!
    'stopLossStdMultiplier': [80_00, 100_00, 120_00],
    # Оптимизировать !
    'takeProfitStdMultiplier': [4050],
    # Оптимизировать !!
    'varianceRatioCarreteParameter': [20, 40],
    # Оптимизировать !!!
    'scanHalfTime': [500, 1000, 2000],
    # Оптимизировать 0!
    'halfToFat': [2, 3, 1],
    # Оптимизировать 0!
    'halfToLight': [1, 1.5],
    # Оптимизировать 0!
    'halfToTime': [3, 2, 1],
    }

ESTIMATORSIMPLIFIER = 1.5
OPTIMIZESIMPLIFIER = 6
RecursionBorder = 1000
DEBUGMODE = True
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Creates csv into backTest dir')
    # parser.add_argument('--file', help='filename at testData without csv')
    parser.add_argument('--approx', help='number of insample strategies', default=140)
    args = parser.parse_args()

    DEBUGMODE = True
    firstConditions = ParameterGrid(firstConditions)
    firstConditions = pd.DataFrame(firstConditions).sample(frac=1, random_state=9).reset_index(drop=True)
    print(f"Init params shape = {firstConditions.shape[0]}")
    RES, EVOLUTION = _get_the_best_strategy(mainOptimizerData=inpData.copy().iloc[:400_000],
                                            initConditions=firstConditions,
                                            numberOfDotes=20, saveInfo=True)

