from abc import ABC, abstractmethod


class AbstractOrderInterface:

    @abstractmethod
    def get_actual_data(self, *args):
        pass

    @abstractmethod
    def place_order(self, *args):
        pass

    @abstractmethod
    def get_asset_data_hist(self, *args):
        pass

    @abstractmethod
    def validate_open_order(self, *args):
        pass

    @abstractmethod
    def validate_close_order(self, *args):
        pass


