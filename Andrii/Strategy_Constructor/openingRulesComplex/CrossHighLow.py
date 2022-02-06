from Andrii.Strategy_Constructor.rules_constructors.TrendFollowingRules import TrendFollowingStrategyRulesConstructor
from Andrii.Strategy_Constructor.exceptions.AvailableErrors import *

from pandas import Series


class OpeningCrossHighLowRule:
    def __init__(self):
        self.compatible = [TrendFollowingStrategyRulesConstructor.StrategyRuleConstructorType]

        self.low = None
        self.high = None

    @classmethod
    def RuleIndex(cls):
        return 'OpeningCrossHighLowRule'

    def add_low_calculation(self, low):
        if self.RuleIndex not in low.compatible:
            raise SmallWorkers().UnCompatibleRuleBlock(ComplexRule=self, Worker=low)
        self.low = low

    def add_high_calculation(self, high):
        if self.RuleIndex not in high.compatible:
            raise SmallWorkers().UnCompatibleRuleBlock(ComplexRule=self, Worker=high)
        self.high = high

    # Должен быть в каждом правиле
    def CalculateRuleParams(self, calculation_data: Series):
        if type(calculation_data) != Series:
            raise TypeError('Data for parameter calculation must have pd.Series type')
        self.low.add_calculation_data(calculation_data)
        self.high.add_calculation_data(calculation_data)
        self.low.create_parameter()
        self.high.create_parameter()

    def condition(self, current_dot):
        result = {'OpenChecker': False, 'OperationType': None}
        if current_dot.open > self.high.param:
            result['OpenChecker'] = True
            result['OperationType'] = 'BUY'

        if current_dot.open < self.low.param:
            result["OpenChecker"] = True
            result['OperationType'] = 'SELL'

        return result