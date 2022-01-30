import pandas as pd
from AvailableErrors import *
def createTimeDifference(index):
    """
    Выполняет пересчет индексов в количество дней с начала первой даты. Пример:
    DatetimeIndex(['2018-11-01 10:00:00', '2018-11-01 11:00:00',
               '2018-11-01 12:00:00', '2018-11-01 13:00:00',
               '2018-11-01 14:00:00', '2018-11-01 15:00:00',
               '2018-11-01 16:00:00', '2018-11-01 17:00:00',
               '2018-11-01 18:00:00', '2018-11-01 19:00:00',
               '2018-11-01 20:00:00', '2018-11-01 21:00:00',
               '2018-11-01 22:00:00', '2018-11-01 23:00:00',
               '2018-11-02 00:00:00', '2018-11-02 01:00:00',
               '2018-11-02 02:00:00', '2018-11-02 03:00:00',
               '2018-11-02 04:00:00', '2018-11-02 05:00:00'],
              dtype='datetime64[ns]', name='time', freq='H')
    Превращается в
    [Timedelta('0 days 10:00:00'),Timedelta('0 days 11:00:00'),Timedelta('0 days 12:00:00'), Timedelta('0 days 14:00:00'),Timedelta('0 days 15:00:00'),
     Timedelta('0 days 16:00:00'),Timedelta('0 days 17:00:00'),Timedelta('0 days 18:00:00'),Timedelta('0 days 19:00:00'),Timedelta('0 days 20:00:00'),
     Timedelta('0 days 21:00:00'),Timedelta('0 days 22:00:00'),Timedelta('0 days 23:00:00'),Timedelta('1 days 00:00:00'),Timedelta('1 days 01:00:00'),
     Timedelta('1 days 02:00:00'),Timedelta('1 days 03:00:00'),Timedelta('1 days 04:00:00'),Timedelta('1 days 05:00:00')]

    :param index:
    :return:
    """
    if type(index) != pd.core.indexes.datetimes.DatetimeIndex:
        raise StrategyErrors().WrongIndicesType()

    ZERO_TIME = index[0].to_numpy()
    ZERO_TIME = ZERO_TIME.astype('datetime64[s]').item().date()
    return index - pd.to_datetime(ZERO_TIME)