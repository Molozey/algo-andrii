from Andrii.Strategy_Constructor.rules_constructors.TrendFollowingRules import TrendFollowingStrategyRulesConstructor

from abc import ABC, abstractmethod


class HoldRuleBase(ABC):
    @abstractmethod
    def add_open_position_parameters(self):
        pass

    @abstractmethod
    def condition(self):
        pass


class HoldStopLossNoSlippage(ABC):
    def __init__(self):
        self.compatible = [TrendFollowingStrategyRulesConstructor.StrategyRuleConstructorType]

        self.open_dot = None
        self.operation_type = None

        self.threshold = None

    @classmethod
    def HoldIndex(cls):
        return 'HoldStopLoss'

    #   MUST BE IN ALL HOLD RULES
    def add_open_position_parameters(self, open_dot, operation_type):
        self.open_dot = open_dot
        self.operation_type = operation_type

    def make_threshold(self, threshold):
        self.threshold = threshold

    def condition(self, current_dot):
        result = {'HoldTrigger': False, 'HoldingRule': self.HoldIndex()}
        if self.operation_type == "BUY":
            if (current_dot.open / self.open_dot.open) - 1 < -1 * self.threshold["BUY"]:
                result["HoldTrigger"] = True
        if self.operation_type == "SELL":
            if (self.open_dot.open / current_dot.open) - 1 < -1 * self.threshold["SELL"]:
                result["HoldingTrigger"] = True
        return result
