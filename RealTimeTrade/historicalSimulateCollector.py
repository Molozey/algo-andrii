from connectorInterface import AbstractOrderInterface
import pandas as pd


class SimulatedOrderGenerator(AbstractOrderInterface):
    def __init__(self, pathToData):
        super().__init__()
        self.data = pd.read_csv(pathToData)
        self.position = 0

    def place_open_order(self):
        pass

    def place_close_order(self):
        pass

    def collect_past_multiple_prices(self, numberOfDotes):
        self.position = numberOfDotes
        return list(self.data.open.iloc[:numberOfDotes])

    def get_actual_data(self):
        self.position += 1
        return self.data.open.iloc[self.position-1]
