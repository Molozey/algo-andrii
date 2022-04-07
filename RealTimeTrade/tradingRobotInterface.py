import pandas as pd


class ImRobot:
    def __init__(self, name, config_file_way):
        """

        :param name: Robot Name
        :param config_file_way: Path to config txt
        """
        self.name = name
        config = pd.read_csv(str(config_file_way), header=0, sep=' ')
        self.time_interval = float(config.iloc[list(config.index).index('updateTime: '), 0])

        self.connector = None

    def add_connector(self, connector):
        self.connector = connector


    def tradingCycle(self):
        if self.connector is None:
            raise ModuleNotFoundError('Connector not placed')
