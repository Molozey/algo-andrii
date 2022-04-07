from abc import ABC, abstractmethod


class AbstractOrderInterface:
    def __init__(self):
        pass

    @abstractmethod
    def get_actual_data(self):
        pass

    @abstractmethod
    def place_open_order(self):
        pass

    @abstractmethod
    def validate_open_order(self):
        pass

    @abstractmethod
    def place_close_order(self):
        pass

    @abstractmethod
    def validate_close_order(self):
        pass

    @abstractmethod
    def collect_past_multiple_prices(self, NumberOfPrices):
        pass
