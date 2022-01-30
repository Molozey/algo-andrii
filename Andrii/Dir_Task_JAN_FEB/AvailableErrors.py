"""================================================================================================================================================="""
class CompatibleException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f"{self.message}"

"""================================================================================================================================================="""
class StrategyErrors:
    def __init__(self):
        self.message = None

    def UnCompatibleFilter(self, Strategy, Filter):
        self.message = f"Filter ({Filter.FilterIndex()}) not compatible with Extractor ({Strategy.ExtractorIndex()})"
        raise CompatibleException(self.message)

    def WrongIndicesType(self):
        self.message = f"Change DataFrame index to pandas datetime format"
        raise TypeError(self.message)

    def UnCompatibleExtractor(self, HubStrategy, StrategyExtractor):
        self.message = f"Extractor ({StrategyExtractor.ExtractorIndex()}) not compatible with StrategyHub ({HubStrategy.StrategyIndex()})"
        raise CompatibleException(self.message)

    def UnCompatibleRulesConstructor(self, HubStrategy, RulesConstructor):
        self.message = f"RulesConstructor ({RulesConstructor.StrategyRuleConstructorType()}) not compatible with StrategyHub ({HubStrategy.StrategyIndex()})"
        raise CompatibleException(self.message)
"""================================================================================================================================================="""
class RulesConstructorErrors:
    def __init__(self):
        self.message = None

    def UnCompatibleComplexRule(self, RuleConstructor, ComplexRule):
        self.message = f"Rule ({RuleConstructor.StrategyRuleConstructorType()}) not compatible with StrategyHub ({ComplexRule.RuleIndex()})"
"""================================================================================================================================================="""
class SmallWorkers:
    def __init__(self):
        self.message = None

    def UnCompatibleRuleBlock(self, ComplexRule, Worker):
        self.message = f"Worker ({Worker.WorkerIndex()}) cannot be used with ComplexRule ({ComplexRule.RuleIndex()})"
"""================================================================================================================================================="""
class SplitDataErrors:
    def __init__(self):
        self.message = None

    def WrongCentralDotTypeError(self):
        self.message = f"Central splitter dot have wrong type. Must be int"
        raise TypeError(self.message)
"""--------------------------------------------------------------------------------------------------------------------------------------------------"""

class FilterErrors:
    def __init__(self):
        self.message = None

    def WrongTimeType(self):
        self.message = f"Type of input index must me pandas.core.indexes.datetimes.DatetimeIndex"
        raise TypeError(self.message)

    def WrongShiftedType(self):
        self.message = f"Type must be pandas._libs.tslibs.timedeltas.Timedelta"
        raise TypeError(self.message)
"""================================================================================================================================================="""