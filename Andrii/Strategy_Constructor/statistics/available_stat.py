from Andrii.Strategy_Constructor.openingRulesComplex.CrossHighLow import OpeningCrossHighLowRule

from pandas import Series


class CalculateLow:
    def __init__(self):
        self.compatible = [OpeningCrossHighLowRule.RuleIndex]

        self.calc_data = None

        self.param = None

    @classmethod
    def WorkerIndex(self):
        return 'CalculateLow'

    def add_calculation_data(self, calc_data: Series):
        if type(calc_data) != Series:
            raise TypeError(f"Worker gets wrong type of input data")
        self.calc_data = calc_data

    def create_parameter(self):
        self.param = min(self.calc_data)


class CalculateHigh:
    def __init__(self):
        self.compatible = [OpeningCrossHighLowRule.RuleIndex]

        self.calc_data = None

        self.param = None

    @classmethod
    def WorkerIndex(self):
        return 'CalculateHigh'

    def add_calculation_data(self, calc_data: Series):
        if type(calc_data) != Series:
            raise TypeError(f"Worker gets wrong type of input data")
        self.calc_data = calc_data

    def create_parameter(self):
        self.param = max(self.calc_data)
