
import pandas as pd
from sklearn import linear_model
import numpy as np
from typing import Union, List


def get_half_time(openTuple: pd.Series) -> float:
    """
    Функция отдающая период полураспада
    :param openTuple:
    :return:
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


def create_strategy_config(params, CAP):
    """
    Создает удобную сетку для дальнейших расчетов
    :param params: начальные init параметры
    :return: словарь из параметров использующийся везде
    """
    capital = CAP
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


def variance_ratio(logTuple: tuple, retTuple: tuple, params: dict,
                   extend_info: bool = False) -> Union[bool, list[float, bool]]:
    """
    Функция для open. Здесь лаг q зависит только от гиперпараметра
    Возвращает значение variacne ratio. Необходимо для понимания того, можно ли открывать сделку
    :param extend_info: adding to return float of varianceRatio
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
        if not extend_info:
            return True
        if extend_info:
            return [result, True]
    else:
        if not extend_info:
            return False
        if extend_info:
            return [result, False]


def reverse_variance_ratio(preComputed, params: dict, timeBorderCounter: int, VRstatement=False,
                           extend_info: bool = False) -> Union[bool, list[float, bool]]:
    """
    Возвращает значение variance ratio. Необходимо для понимания того, можно ли открывать сделку
    :param extend_info: adding to return float of varianceRatio
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
                    if not extend_info:
                        return True
                    if extend_info:
                        return [result, True]
                else:
                    if not extend_info:
                        return False
                    if extend_info:
                        return [result, False]
            if VRstatement:
                if result < params['varianceRatioFilter']:
                    if not extend_info:
                        return True
                    if extend_info:
                        return [result, True]
                else:
                    if not extend_info:
                        return False
                    if extend_info:
                        return [result, False]
        else:
            return False

    except ZeroDivisionError:
        print('Warning at variance ratio. Division on zero')
        return False
    except:
        print(preComputed)
        return False


