from abc import ABC, abstractmethod
from pandas import DataFrame


class BaseObjectType(ABC):
    """
    Базовый класс отображающий тип передаваемой информации. От него базируются все типы данных
    """
    def __init__(self):
        self._object = None

    @property
    @abstractmethod
    def ObjectType(self) -> None:
        """
        Необходим для создания логики совместимых узлов нод
        :return: None
        """
        pass

    @abstractmethod
    def transfer(self, obj):
        """
        Определяет правила записи объекта в память
        :param obj: любой объект
        :return: None
        """
        pass

    @abstractmethod
    def receive(self):
        """
        Определяет правило передачи информации сохраненной в объекте
        :return:
        """
        pass


class Value(BaseObjectType):
    """
    Тип информации передающий числовые значения
    """
    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, value):
        if type(value) != (float or int):
            raise ValueError(f"Get {type(value)} instead of INT | FLOAT")
        self._object = value

    def receive(self):
        return self._object


class Frame(BaseObjectType):
    """
    Тип информации передающий объекты pandas.DataFrame
    """
    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, value):
        if type(value) != DataFrame:
            raise ValueError(f"Get {type(value)} instead of pd.DaraFrame")
        self._object = value

    def receive(self):
        return self._object


class Extractor(BaseObjectType):
    """
    Тип информации передающий объекты семлирования данных
    """
    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, extractor):
        self._object = extractor

    def receive(self):
        return self._object


class OpenGraph(BaseObjectType):
    """
    Тип информации передающий граф открытия позиции
    """
    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, open_graph):
        self._object = open_graph

    def receive(self):
        return self._object


class CloseGraph(BaseObjectType):
    """
    Тип информации передающий граф удержания позиции
    """
    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, hold_graph):
        self._object = hold_graph

    def receive(self):
        return self._object


class StrategyHub(BaseObjectType):
    """
    Тип информации передающий граф удержания позиции
    """
    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, strategy_hub):
        self._object = strategy_hub

    def receive(self):
        return self._object