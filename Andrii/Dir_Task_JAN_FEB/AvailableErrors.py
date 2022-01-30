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
        self.message = f"Filter ({Filter.FilterIndex()}) not compatible with strategy ({Strategy.StrategyIndex()})"
        raise CompatibleException(self.message)

    def WrongIndicesType(self):
        self.message = f"Change DataFrame index to pandas datetime format"
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