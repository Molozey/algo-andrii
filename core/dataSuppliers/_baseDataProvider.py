from abc import ABC, abstractmethod


class AbstractDataProvider(ABC):
    @abstractmethod
    def get_actual_data(self, *params):
        pass

    @abstractmethod
    def get_historical_data(self, *params):
        pass