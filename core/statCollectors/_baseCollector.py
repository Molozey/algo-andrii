from abc import ABC, abstractmethod


class StatCollector:
    def __init__(self, *args):
        pass

    @abstractmethod
    def add_trade_line(self, line):
        pass
