from abc import ABC, abstractmethod
import pandas


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
        if type(value) != pandas.DataFrame:
            raise ValueError(f"Get {type(value)} instead of pd.DaraFrame")
        self._object = value

    def receive(self):
        return self._object


class NodesLinker(BaseObjectType):
    """
    Тип информации хранящий в себе информацию о связи между нодами
    """
    def ObjectType(self) -> None:
        return self.__class__.__name__

    def transfer(self, parent):
        self._object = parent

    def receive(self):
        return self._object
