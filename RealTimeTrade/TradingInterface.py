import pandas as pd


class TradingInterface:
    def __init__(self, name: str, robotConfig: str, strategyParameters_file_way: str,
                 ticker):
        """
        Initialize Interface
        :param name: Robot's Name
        :param robotConfig: txt with next structure
        ...
        updateTime:60
        waitingParameter: 1
        apiToken:
        ...
        :param strategyParameters_file_way: txt with strategy hyper parameters
        :param ticker: which instrument we trade?
        """


        self.name = name
        config = pd.read_csv(str(robotConfig), header=None, index_col=0, sep=':')
        print(config)


monkey = TradingInterface(name='monkey', robotConfig='robotConfig.txt',
                          strategyParameters_file_way='strategyParameters.txt', ticker='CHFJPY')
