from Andrii.Strategy_Constructor.exceptions.AvailableErrors import *
from pandas import DataFrame
from tqdm import tqdm


class HubTrendFollowingStrategy:
    @classmethod
    def StrategyIndex(cls):
        return 'HubTrendFollowingStrategy'

    def __init__(self):
        self.all_data = None
        self.lookback = None
        self.potential_hold = None
        self.extractor = None
        self.RulesConstructor = None
        self.actual_data = None

        self.simulation_results = None
    def add_extractor(self, method):
        if self.StrategyIndex() not in method.compatible:
            raise StrategyErrors().UnCompatibleExtractor(HubStrategy=self, StrategyExtractor=method)
        self.extractor = method

    def transfer_init_data(self, raw_data: DataFrame):
        if type(raw_data) != DataFrame:
            raise TypeError('Input Data must have type pandas DataFrame')
        self.all_data = raw_data.copy()


    def testing_simulation(self):
        tqdm_bar = tqdm(total=self.all_data.shape[0])
        print('-----')
        RESULTS = list()
        self.actual_data = self.all_data
        horizonStopTriggerLast = None
        horizonStopTriggerCurrent = 1
        #   Очень страшное условие, не уверен что всегда будет работать корректно
        #   Нужно вырезать последний список торгов
        # while horizonStopTriggerCurrent != horizonStopTriggerLast: #  Выходит бесконечный цикл
        # (self.all_data.shape[0] - tqdm_bar.last_print_n > 100) связанно со скользящим средним
        while (not self.actual_data.empty) and (horizonStopTriggerCurrent != horizonStopTriggerLast) and (self.all_data.shape[0] - tqdm_bar.last_print_n > 1000):
            self.extractor.transfer_data(self.actual_data)

            self.extractor.making_horizon()
            #   ПРОБЛЕМА С БЫСТРОДЕЙСТВИЕМ
            potential_period = self.extractor.after_filters

            self.RulesConstructor.transfer_data(potential_period)
            self.RulesConstructor.CalculateAllParams()

            open_dot_index, operation = self.RulesConstructor.FindOpenDot()

            if open_dot_index:
                open_dot = potential_period.loc[open_dot_index]
                max_holding_period = potential_period.loc[open_dot_index:]
                self.RulesConstructor.add_holding_data(max_holding_period)
                close_dot_index, hold_type = self.RulesConstructor.FindCloseDot(open_dot=open_dot, operation_type=operation)
                self.actual_data = self.actual_data.loc[close_dot_index:]

                horizonStopTriggerLast = horizonStopTriggerCurrent
                horizonStopTriggerCurrent = open_dot_index

                RESULTS.append({'OpenTime': open_dot_index, 'OpenPrice': self.all_data.loc[open_dot_index].open,
                                'CloseTime': close_dot_index, 'ClosePrice': self.all_data.loc[close_dot_index].open,
                                'OperationType': operation,
                                'CloseType': hold_type})

                tqdm_bar.update(self.actual_data.iloc[0].line_number - tqdm_bar.last_print_n)
            if not open_dot_index:
                self.actual_data = self.actual_data.loc[potential_period.index[-1]:]
                tqdm_bar.update(self.actual_data.iloc[0].line_number - tqdm_bar.last_print_n)
        tqdm_bar.update(self.all_data.shape[0] - tqdm_bar.last_print_n)
        #tqdm_bar.close()
        self.simulation_results = RESULTS

    def add_Rules_constructor(self, rulesConstructor):
        if self.StrategyIndex not in rulesConstructor.compatible:
            raise StrategyErrors().UnCompatibleRulesConstructor(HubStrategy=self, RulesConstructor=rulesConstructor)

        self.RulesConstructor = rulesConstructor