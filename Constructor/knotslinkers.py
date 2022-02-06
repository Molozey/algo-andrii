from typing import List
from abc import ABC, abstractmethod
from Constructor.objectTypes import BaseObjectType


class BaseOutputKnot(ABC):
    """
    Базовая реализация узла передачи данных
    """
    def __init__(self, KnotObject: BaseObjectType):
        """

        :param KnotObject: объект объясняющий логику принятия и передачи данных из узла
        """
        self.KnotObject = KnotObject
        self._children.clear()

    _children: List = []

    def add_child_link(self, *childKnot) -> None:
        """
        Создание связи между родительским и дочерним узлом. После установления в случае изменения состояния в родителе
        дочка получит обновленное состояние
        :param childKnot: объект типа BaseInputKnot
        :return: None
        """
        for child in childKnot:
            if child.KnotObject.ObjectType() != self.KnotObject.ObjectType():
                raise KeyError('Linking Error')

            self._children.append(child)
        return None

    def remove_child_link(self, *childKnot):
        """
        Удаление связи между родительским и дочерним узлом.
        :param childKnot: объект типа BaseInputKnot
        :return: None
        """
        for child in childKnot:
            self._children.remove(child)
        return None

    def _refreshChildrenKnots(self):
        """
        Запуск оповещения потомков об изменении состояния родителя
        :return:
        """
        for observer in self._children:
            observer.update(self)

    def changed_condition(self, obj):
        """
        Функция изменения состояния данных находящихся в узле
        :param obj: любой объект
        :return: None
        """
        self.KnotObject.transfer(obj)
        self._refreshChildrenKnots()


class BaseInputKnot(ABC):
    """
    Базовая реализация узла принятия данных
    """
    def __init__(self, KnotObject: BaseObjectType):
        """

        :param KnotObject: объект объясняющий логику принятия и передачи данных в узла
        """
        self.KnotObject = KnotObject

    def update(self, parent: BaseOutputKnot):
        """
        Функция обновления состояния данных в дочернем узле
        :param parent: объект передающийся из родителя через интерфейс оповещения
        :return:
        """
        self.KnotObject.transfer(parent.KnotObject.receive())

