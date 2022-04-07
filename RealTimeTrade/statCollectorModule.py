from abc import ABC, abstractmethod
import pandas as pd
import os


class StatCollector:
    def __init__(self, *args):
        pass

    @abstractmethod
    def add_trade_line(self, line):
        pass


class PandasStatCollector(StatCollector):

    def __init__(self, fileToSave, header=0, sep=','):

        print('Using Pandas StatCollector')
        super().__init__()
        self._filePath = fileToSave
        self._header = header
        self._sep = sep
        if not os.access(f'{self._filePath}', os.F_OK):
            raise FileExistsError(f'File to save stat not exists\nfilepath:{self._filePath}')

    def add_trade_line(self, line):
        """

        :param line: pd.Series or dict
        :return:
        """
        file = pd.read_csv(f"{self._filePath}", header=self._header, sep=self._sep)
        if file.empty:
            file = pd.DataFrame(line)
            print('initial stat file:\n', file)
        if not file.empty:
            file = file.append(line, ignore_index=True)
            file.reset_index(drop=True, inplace=True)
            file.to_csv(f"{self._filePath}", sep=self._sep, header=self._header)
