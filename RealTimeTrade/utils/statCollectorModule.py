from abc import ABC, abstractmethod
import pandas as pd
from pandas.errors import EmptyDataError
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
        try:
            file = pd.read_csv(f"{self._filePath}", sep=self._sep, index_col=0)
            if file.empty:
                file = pd.Series(line).to_frame().T
                print('initial stat file:\n', file)
                file.reset_index(drop=True, inplace=True)
                file.to_csv(f"{self._filePath}", sep=self._sep, header=1)
                return None

        except EmptyDataError:
            file = pd.Series(line).to_frame().T
            file.reset_index(drop=True, inplace=True)
            file.to_csv(f"{self._filePath}", sep=self._sep, header=1)
            return None

        if not file.empty:
            file = file.append(pd.Series(line).T, ignore_index=True)
            file.reset_index(drop=True, inplace=True)
            file.to_csv(f"{self._filePath}", sep=self._sep)
            return None


class BQueryCollector(StatCollector):
    def __init__(self):
        super().__init__()
        pass

    def add_trade_line(self, line):
        pass
