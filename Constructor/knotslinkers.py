from typing import List
from abc import ABC, abstractmethod


class BaseObjectType(ABC):

    def __init__(self):
        self._object = None

    @property
    @abstractmethod
    def ObjectType(self):
        pass

    @abstractmethod
    def transfer(self, obj):
        pass

    @abstractmethod
    def receive(self):
        pass


class Value(BaseObjectType):

    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, value):
        self._object = value

    def receive(self):
        return self._object


class BaseOutputKnot(ABC):
    def __init__(self, KnotObject: BaseObjectType):
        self.KnotObject = KnotObject
        self._children.clear()

    _children: List = []

    def add_child_link(self, *childKnot) -> None:
        for child in childKnot:
            if child.KnotObject.ObjectType() != self.KnotObject.ObjectType():
                raise KeyError('Linking Error')

            self._children.append(child)
        return None

    def remove_child_link(self, *childKnot):
        for child in childKnot:
            self._children.remove(child)
        return None

    def _refreshChildrenKnots(self):
        for observer in self._children:
            observer.update(self)

    def changed_condition(self, obj):
        self.KnotObject.transfer(obj)
        self._refreshChildrenKnots()


class BaseInputKnot(ABC):
    def __init__(self, KnotObject: BaseObjectType):
        self.KnotObject = KnotObject

    def update(self, parent: BaseOutputKnot):

        self.KnotObject.transfer(parent.KnotObject.receive())

