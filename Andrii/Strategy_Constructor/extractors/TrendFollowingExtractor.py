from datetime import timedelta
from pandas import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from Andrii.Strategy_Constructor.exceptions.AvailableErrors import *
from Andrii.Strategy_Constructor.strategyHubs.HubTrend import *

class StrategyExtractor:
    @classmethod
    def ExtractorIndex(cls):
        return 'StrategyExtractor'

    def __init__(self) -> None:
        self.compatible = [HubTrendFollowingStrategy.StrategyIndex()]

        self._start = None
        self._end = None
        self.input_data = None

        self.filters = list()
        self.after_filters = None

        # self.after_filters = list()

        pass

    def input_params(self, start_horizon: timedelta, end_horizon: (timedelta or str)) -> None:

        self._start = start_horizon
        #   end_horizon can be ':'
        self._end = end_horizon

    def apply_filters(self, single_filter) -> None:
        if self.ExtractorIndex not in single_filter.compatible:
            raise StrategyErrors().UnCompatibleFilter(Strategy=self, Filter=single_filter)
        self.filters.append(single_filter)

    def transfer_data(self, data: DataFrame) -> None:
        if type(data.index) != DatetimeIndex:
            raise StrategyErrors().WrongIndicesType()
        self.input_data = data

    def making_horizon(self):
        for dot_on_all_data in self.input_data.index:
            logical_filter = True
            #   SELECT BORDERS
            if self._end != ':':
                buffer_data = self.input_data.loc[dot_on_all_data + self._start: dot_on_all_data + self._end]
            if self._end == ':':
                buffer_data = self.input_data.loc[dot_on_all_data + self._start:]
            for single_filter in sorted(self.filters, key=lambda fil: fil.priority):

                FILTER = single_filter
                FILTER.apply_buffer_data(buffer_data=buffer_data)
                if not FILTER.condition():
                    logical_filter = False
                    break

            if logical_filter:
                self.after_filters = buffer_data
                break   # This break scares me

