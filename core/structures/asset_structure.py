from dataclasses import dataclass
from typing import List, Union
import datetime
import pandas as pd
from numpy import timedelta64, datetime64

import warnings

from core.dataSuppliers.SAXO import SaxoDataProvider
from core.warnings.loggingPresets import *
from dateutil import parser
from time import sleep


@dataclass
class assetInformation:
    @classmethod
    def _time_converter(cls, stringTime: str) -> datetime.datetime:
        if stringTime[-1] == 'Z':
            return pd.to_datetime(parser.isoparse(stringTime))
        if len(stringTime) == 19:
            return datetime.datetime.strptime(stringTime, '%Y-%m-%d %H:%M:%S')

    availableSuppliers = Union[SaxoDataProvider]
    #   -----------------------------------------------
    Name: str
    Supplier: availableSuppliers
    UpdatableTime: int
    CloseAsk: List[float]
    CloseBid: List[float]
    HighAsk: List[float]
    LowAsk: List[float]
    LowBid: List[float]
    HighBid: List[float]
    OpenAsk: List[float]
    OpenBid: List[float]
    OpenMiddle: List[float]
    CloseMiddle: List[float]
    LowMiddle: List[float]
    HighMiddle: List[float]
    Time: List[datetime.datetime]
    _cachedCollectedTime: datetime.datetime
    interface: object

    def __init__(self, ticker_name: str, ticker_parameters):
        self.Name = ticker_name
        self.Supplier = ticker_parameters['supplier']
        self.UpdatableTime = pd.Timedelta(seconds=int(ticker_parameters['updatable_time']))
        self._updatableSec = self.UpdatableTime // pd.Timedelta(seconds=1)

    def add_interface(self, interface):
        self.interface = interface

    def start_cycle(self):
        if self.interface.strategy.requiredHistory is not None:
            self.download_history(self.interface.strategy.requiredHistory)

        self.looked_for_actual_dot()

    def download_history(self, lookBackTime: pd.Timedelta):
        numberOfDotes = int(lookBackTime // self.UpdatableTime)
        if self.Supplier:
            history = self.Supplier.get_asset_data_hist(ticker=self.Name,
                                                        density=self._updatableSec,
                                                        amount_intervals=numberOfDotes)
            history = pd.DataFrame(history)
            history['Time'] = history['Time'].apply(lambda x: self._time_converter(x))
            self.CloseAsk = list(history['CloseAsk'].values)
            self.CloseBid = list(history['CloseBid'].values)
            self.HighAsk = list(history['HighAsk'].values)
            self.LowAsk = list(history['LowAsk'].values)
            self.LowBid = list(history['LowBid'].values)
            self.HighBid = list(history['HighBid'].values)
            self.OpenAsk = list(history['OpenAsk'].values)
            self.OpenBid = list(history['OpenBid'].values)
            self.OpenMiddle = list(history.apply(lambda x: (x['OpenBid'] + x['OpenAsk']) / 2, axis=1).values)
            self.CloseMiddle = list(history.apply(lambda x: (x['CloseBid'] + x['CloseAsk']) / 2, axis=1).values)
            self.LowMiddle = list(history.apply(lambda x: (x['LowBid'] + x['LowAsk']) / 2, axis=1).values)
            self.HighMiddle = list(history.apply(lambda x: (x['HighBid'] + x['HighAsk']) / 2, axis=1).values)
            self.Time = list(history['Time'].values)
            self._cachedCollectedTime = datetime.datetime.now().utcnow()
            del history
            print(
                f'{Update_log}Successfully downloaded last {numberOfDotes} dotes for {self.Name} with last time {self._cachedCollectedTime}')

        else:
            warnings.warn('No brokerInterface plugged')
            return f"{Error_log}No brokerInterface"

    def looked_for_actual_dot(self):
        while pd.to_datetime(self._cachedCollectedTime).minute + (
                self._updatableSec // 60) > datetime.datetime.now().minute:
            sleep(.01)

        _first_step = True
        offset = 0
        while True:
            if _first_step is True:
                sleep(3.5)
            if type(offset) != int:
                # TODO: find a way to reduce lag from SAXO Bank
                _shiffterSaver = self.UpdatableTime - (offset - self.UpdatableTime)
            else:
                _shiffterSaver = self.UpdatableTime + pd.Timedelta(seconds=5, milliseconds=666)

            if _first_step or (pd.to_datetime(datetime.datetime.now().utcnow()) - pd.to_datetime(self._cachedCollectedTime)
                               >= _shiffterSaver):
                # print(pd.to_datetime(datetime.datetime.now().utcnow()))
                # print(self.Name, 'UPDT')
                if not _first_step:
                    history = self.Supplier.get_asset_data_hist(ticker=self.Name,
                                                                density=self._updatableSec,
                                                                amount_intervals=1)
                    history = pd.DataFrame(history)
                    while history['Time'].apply(lambda x: self._time_converter(x)).values[0] == _cachedCollectedTime:
                        sleep(3)
                        history = self.Supplier.get_asset_data_hist(ticker=self.Name,
                                                                    density=self._updatableSec,
                                                                    amount_intervals=1)
                        history = pd.DataFrame(history)

                if _first_step:
                    history = self.Supplier.get_asset_data_hist(ticker=self.Name,
                                                                density=self._updatableSec,
                                                                amount_intervals=1)
                    history = pd.DataFrame(history)

                history['Time'] = history['Time'].apply(lambda x: self._time_converter(x))
                _cachedCollectedTime = history['Time'].values[0]
                # print(self.Name, 'BEFORE', len(self.CloseAsk))
                self.CloseAsk.append(history['CloseAsk'].values[0])
                self.CloseBid.append(history['CloseBid'].values[0])
                self.HighAsk.append(history['HighAsk'].values[0])
                self.LowAsk.append(history['LowAsk'].values[0])
                self.LowBid.append(history['LowBid'].values[0])
                self.HighBid.append(history['HighBid'].values[0])
                self.OpenAsk.append(history['OpenAsk'].values[0])
                self.OpenBid.append(history['OpenBid'].values[0])
                self.Time.append(list(history['Time'].values)[0])
                self.OpenMiddle.append(history.apply(lambda x: (x['OpenBid'] + x['OpenAsk']) / 2, axis=1).values[0])
                self.CloseMiddle.append(history.apply(lambda x: (x['CloseBid'] + x['CloseAsk']) / 2, axis=1).values[0])
                self.LowMiddle.append(history.apply(lambda x: (x['LowBid'] + x['LowAsk']) / 2, axis=1).values[0])
                self.HighMiddle.append(history.apply(lambda x: (x['HighBid'] + x['HighAsk']) / 2, axis=1).values[0])

                # print(self.Name, 'BEFORE', self._cachedCollectedTime)
                if _first_step is False:
                    offset = pd.to_datetime(self._cachedCollectedTime)
                self._cachedCollectedTime = self.Time[-1]
                if _first_step is False:
                    offset = pd.to_datetime(self._cachedCollectedTime) - offset
                    # print('offset', offset)
                # print(self.Name, 'AFTER', self._cachedCollectedTime)
                # print(self.Name, 'AFTER', len(self.CloseAsk))
                del history

                if _first_step:
                    _first_step = False

