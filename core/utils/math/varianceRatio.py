import numpy as np
from typing import Union, List


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
