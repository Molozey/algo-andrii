from typing import List
from abc import ABC, abstractmethod
from Constructor.objectTypes import BaseObjectType


class BaseOutputKnot:
    """
    Базовая реализация узла передачи данных
    """
    def __init__(self, KnotObject):
        """
        :param KnotObject: объект объясняющий логику принятия и передачи данных из узла
        """
        self.KnotObject = KnotObject
        self._children = list()

    def add_child_link(self, childKnot) -> None:
        """
        Создание связи между родительским и дочерним узлом. После установления в случае изменения состояния в родителе
        дочка получит обновленное состояние
        :param childKnot: объект типа BaseInputKnot
        :return: None
        """
        if childKnot.KnotObject.ObjectType() != self.KnotObject.ObjectType():
            raise KeyError('Linking Error')
        self._children.append(childKnot)
        return None

    def remove_child_link(self, childKnot):
        """
        Удаление связи между родительским и дочерним узлом.
        :param childKnot: объект типа BaseInputKnot
        :return: None
        """
        if childKnot in self._children:
            self._children.remove(childKnot)
        return None

    def _refreshChildrenKnots(self):
        """
        Запуск оповещения потомков об изменении состояния родителя
        :return:
        """
        for observer in self._children:
            observer.Tupdate(self)

    def changed_condition(self, obj):
        """
        Функция изменения состояния данных находящихся в узле
        :param obj: любой объект
        :return: None
        """
        self.KnotObject.transfer(obj)
        self._refreshChildrenKnots()

    def getContent(self):
        return self.KnotObject.receive()


class BaseInputKnot:
    """
    Базовая реализация узла принятия данных
    """
    def __init__(self, KnotObject):
        """

        :param KnotObject: объект объясняющий логику принятия и передачи данных в узла
        """
        self.KnotObject = KnotObject

    def Tupdate(self, parent):
        """
        Функция обновления состояния данных в дочернем узле
        :param parent: объект передающийся из родителя через интерфейс оповещения
        :return:
        """
        self.KnotObject.transfer(parent.KnotObject.receive())

    def getContent(self):
        return self.KnotObject.receive()

