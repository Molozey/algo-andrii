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


class DataFrame(BaseObjectType):

    def ObjectType(self):
        return self.__class__.__name__

    def transfer(self, value):
        self._object = value

    def receive(self):
        return self._object
