from Andrii.Strategy_Constructor.exceptions.AvailableErrors import *
from Andrii.Strategy_Constructor.extractors.TrendFollowingExtractor import StrategyExtractor
from Andrii.Strategy_Constructor.tool_module.tools_functions import createTimeDifference


from pandas.core.indexes.datetimes import DatetimeIndex
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas import to_datetime
from pandas import DataFrame
from numpy import any as nany
from numpy import isin as nisin

from abc import ABC, abstractmethod, abstractclassmethod


global _HOLIDAY_MARKERS
_HOLIDAY_MARKERS = [5, 6]


class BaseFilter(ABC):

    def __init__(self, priority):
        self.priority = priority

    @abstractclassmethod
    def FilterIndex(cls):
        pass

    @abstractmethod
    def apply_buffer_data(self, buffer_data: DataFrame) -> None:
        pass

    @abstractmethod
    def condition(self) -> bool:
        pass


class ConsistingTime(BaseFilter):
    priority = 3

    def __init__(self, *consist_time) -> None:
        self.compatible = [StrategyExtractor.ExtractorIndex]
        self.buffer_data = None

        super().__init__(self.priority)

        for _ in consist_time:
            if type(_) != Timedelta:
                raise FilterErrors().WrongShiftedType()
        self.consist_time = [_ for _ in consist_time]

    @classmethod
    def FilterIndex(cls):
        return 'ConsistingTime'

    def apply_buffer_data(self, buffer_data: DataFrame) -> None:
        if type(buffer_data.index) != DatetimeIndex:
            raise StrategyErrors().WrongIndicesType()

        ZERO_TIME = buffer_data.index[0].to_numpy()
        ZERO_TIME = ZERO_TIME.astype('datetime64[s]').item().date()

        self.buffer_data = buffer_data.copy()
        self.buffer_data.index = buffer_data.index - to_datetime(ZERO_TIME)

        # Если не использовать вызов функции - будет быстрей. Но так зависимости выглядят понятней
        # self.buffer_data = createTimeDifference(buffer_data.index)


        return None

    def condition(self) -> bool:

        for timedelta in self.consist_time:
            #if not timedelta in self.buffer_data.index:
            if not timedelta in self.buffer_data.index:
                return False

            if self.buffer_data.loc[timedelta].isna().any():
                return False

        return True


class NoHolidays(BaseFilter):
    priority = 3

    def __init__(self) -> None:
        self.compatible = [StrategyExtractor.ExtractorIndex]
        self.buffer_data = None
        super().__init__(self.priority)

    @classmethod
    def FilterIndex(cls):
        return 'NoHolidays'

    def apply_buffer_data(self, buffer_data: DataFrame) -> None:
        if type(buffer_data.index) != DatetimeIndex:
            raise StrategyErrors().WrongIndicesType()
        self.buffer_data = buffer_data.copy()
        return None

    def condition(self) -> bool:
        # print(self.buffer_data.index.dayofweek)
        if nany(nisin(self.buffer_data.index.dayofweek, _HOLIDAY_MARKERS)):
            return False
        # for holiday in self.buffer_data.index.dayofweek:
        #     if holiday in _HOLIDAY_MARKERS:
        #         return False

        return True


class LeftBorder(BaseFilter):
    priority = 0

    def __init__(self, left_border: Timedelta) -> None:
        self.compatible = [StrategyExtractor.ExtractorIndex]

        if type(left_border) != Timedelta:
            raise FilterErrors().WrongShiftedType()
        self.buffer_data = None
        self._left_border = left_border

        super().__init__(self.priority)

    @classmethod
    def FilterIndex(cls):
        return 'LeftBorder'

    def apply_buffer_data(self, buffer_data: DataFrame) -> None:

        if type(buffer_data.index) != DatetimeIndex:
            raise StrategyErrors().WrongIndicesType()

        # self.buffer_data = createTimeDifference(buffer_data.index)
        ZERO_TIME = buffer_data.index[0].to_numpy()
        ZERO_TIME = ZERO_TIME.astype('datetime64[s]').item().date()

        self.buffer_data = buffer_data.copy()
        self.buffer_data.index = buffer_data.index - to_datetime(ZERO_TIME)
        return None

    def condition(self) -> bool:
        if self.buffer_data.index[0] != self._left_border:
            return False
        if self.buffer_data.iloc[0].isna().any():
            return False

        return True


class RightBorder(BaseFilter):
    priority = 0

    def __init__(self, right_border: Timedelta):
        self.compatible = [StrategyExtractor.ExtractorIndex]

        if type(right_border) != Timedelta:
            raise FilterErrors().WrongShiftedType()
        self.buffer_data = None
        self._right_border = right_border

        super().__init__(self.priority)

    @classmethod
    def FilterIndex(cls):
        return 'RightBorder'

    def apply_buffer_data(self, buffer_data: DataFrame):
        if type(buffer_data.index) != DatetimeIndex:
            raise StrategyErrors().WrongIndicesType()

        self.buffer_data = buffer_data.copy()
        self.buffer_data.index = createTimeDifference(buffer_data.index)
        return None

    def condition(self) -> bool:
        if self.buffer_data.index[-1] != self._right_border:
            return False
        if self.buffer_data.iloc[-1].isna().any():
            return False

        return True
