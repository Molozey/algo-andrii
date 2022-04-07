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

    def validate_open_order(self):
        pass

    def place_close_order(self):
        pass

    def validate_close_order(self):
        pass
