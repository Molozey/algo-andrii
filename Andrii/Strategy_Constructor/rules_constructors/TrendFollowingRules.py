from Andrii.Strategy_Constructor.strategyHubs.HubTrend import HubTrendFollowingStrategy
from Andrii.Strategy_Constructor.exceptions.AvailableErrors import *

from pandas import to_datetime, Timedelta, DataFrame

from numpy import array as narray, unique as nunique, delete as ndelete, where as nwhere

class TrendFollowingStrategyRulesConstructor:
    def __init__(self):
        self.compatible = [HubTrendFollowingStrategy.StrategyIndex]

        self.opening_rules = list()
        self.holding_rules = list()


        # Тип правил захватывающий статистику с данных вне потенциального периода владения активом. Должен будет базироваться на специальных классах использующих timedelty
        self.soft_holding_rules = list()


        self.data = None
        self.median = None

        self.scanner_data = None
        self.potential_hold_data = None

        self.holding_data = None

        self.open_dot_for_hold = None

        self.operation = None
        self.open_operation = None


    @classmethod
    def StrategyRuleConstructorType(cls):
        return 'TrendFollowConstructor'

    def add_holding_data(self, holding_data: DataFrame):
        self.holding_data = holding_data

    def transfer_data(self, tr_data):
        self.data = tr_data.copy()

        if self.median:
            _start_datetime = to_datetime(self.data.index[0])

            self.scanner_data = self.data.loc[:_start_datetime + self.median]
            self.potential_hold_data = self.data.loc[_start_datetime + self.median:]

    def add_median_dot(self, median_dot: Timedelta):
        self.median = median_dot

    def add_opening_rule(self, opening_rule):
        if self.StrategyRuleConstructorType not in opening_rule.compatible:
            raise RulesConstructorErrors().UnCompatibleComplexRule(RuleConstructor=self, ComplexRule=opening_rule)
        self.opening_rules.append(opening_rule)

    def add_hold_rule(self, hold_rule):
        if self.StrategyRuleConstructorType not in hold_rule.compatible:
            raise RulesConstructorErrors().UnCompatibleComplexRule(RuleConstructor=self, ComplexRule=hold_rule)
        self.holding_rules.append(hold_rule)

    def CalculateAllParams(self):
        if self.scanner_data is not None:
            for single_rule in self.opening_rules:
                single_rule.CalculateRuleParams(self.scanner_data.open)
        else:
            'Strategy has no parameters to calculate from lookback period. Be careful if you strategy may have lookback, something goes wrong'

    def FindCloseDot(self, open_dot, operation_type):
        close_dot_index = None
        close_type = None


        triggered_holding_rules = list()
        if operation_type not in ['BUY', 'SELL']:
            raise ValueError(f'Operation_type has type {operation_type} instead of "BUY" or "SELL"')
        # Вызываются все правила удержания. Возвращается самый ранний.

        holdLogical = False
        for holdRule in self.holding_rules:
            holdRuleInfo = {'TriggerName': None, 'TriggerTime':None}

            holdRule.add_open_position_parameters(open_dot, operation_type)
            for index in self.holding_data.index:
                HoldResult = holdRule.condition(current_dot=self.holding_data.loc[index])
                if HoldResult["HoldTrigger"]:
                    holdRuleInfo['TriggerName'] = HoldResult['HoldingRule']
                    holdRuleInfo["TriggerTime"] = index
                    triggered_holding_rules.append(holdRuleInfo)
                    holdLogical = True
                    break

        if not holdLogical:
            close_dot_index = self.holding_data.index[-1]
            close_type = 'END PERIOD'
        if holdLogical:
            FirstTrigger = sorted(triggered_holding_rules, key=lambda x: x['TriggerTime'])[0]
            close_dot_index = FirstTrigger["TriggerTime"]
            close_type = FirstTrigger["TriggerName"]

        return close_dot_index, close_type

    def FindOpenDot(self):

        open_dot = None
        operation = None

        if self.potential_hold_data is not None:
            _buffer = self.potential_hold_data

        if self.potential_hold_data is None:
            _buffer = self.data

        for index in _buffer.index:
            OpenStatus = self.AvailableOpenStatus(_buffer.loc[index])
            if OpenStatus['logical']:
                open_dot = index
                operation = OpenStatus['operation']
                break

        if open_dot:
            self.open_dot_for_hold = _buffer.loc[index]
            self.open_operation = OpenStatus['operation']

        return open_dot, operation

    def AvailableCloseStatus(self, dot):
        logical_filter = True
        for single_rule in self.holding_rules:

            ruleCondtion = single_rule.condition(dot)
            if ruleCondtion["CloseChecker"]:
                logical_filter = True
                break
        return logical_filter

    def AvailableOpenStatus(self, dot):
        logical_filter = True
        operation = None
        rules_results = list()

        for single_rule in self.opening_rules:

            ruleCondtion = single_rule.condition(dot)
            rules_results.append(ruleCondtion)
            if not ruleCondtion["OpenChecker"]:
                logical_filter = False
                break

        if logical_filter:
            a = narray([x["OperationType"] for x in rules_results])
            if len(nunique(ndelete(a, nwhere(a == None)))) == 1:
                logical_filter = True
                operation = ndelete(a, nwhere(a == None))[0]
        return {'logical': logical_filter, 'operation': operation}

