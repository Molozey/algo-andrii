from Andrii.Strategy_Constructor.rules_constructors.TrendFollowingRules import TrendFollowingStrategyRulesConstructor
from Andrii.Strategy_Constructor.exceptions.AvailableErrors import *
from pandas import Series


class BBandsRule:
    def __init__(self):
        self.compatible = [TrendFollowingStrategyRulesConstructor.StrategyRuleConstructorType]

        self.LBand = None
        self.HBand = None

    @classmethod
    def RuleIndex(cls):
        return 'BBandsRule'

    def add_BBand(self, BBand):
        self.BBand = BBand

    # Должен быть в каждом правиле
    def CalculateRuleParams(self, calculation_data: Series):
        if type(calculation_data) != Series:
            raise TypeError('Data for parameter calculation must have pd.Series type')
        self.BBand.add_calculation_data(calculation_data)
        self.BBand.create_parameter()

    def condition(self, current_dot):
        result = {'OpenChecker': False, 'OperationType': None}
        if current_dot.open < self.BBand.param[0].iloc[-1]:     # Low level
            result['OpenChecker'] = True
            result['OperationType'] = 'BUY'

        if current_dot.open > self.BBand.param[1].iloc[-1]:       # High level
            result["OpenChecker"] = True
            result['OperationType'] = 'SELL'

        return result