from dataclasses import dataclass
from typing import List, Union
import datetime
import pandas as pd


from core.connectors.SAXO import SaxoOrderInterface
from core

@dataclass
class assetInformation:
    @classmethod
    def _time_converter(cls, stringTime: str) -> datetime.datetime:
        if stringTime[-1] == 'Z':
            return datetime.datetime.fromisoformat(stringTime[:-1])
        if len(stringTime) == 19:
            return datetime.datetime.strptime(stringTime, '%Y-%m-%d %H:%M:%S')

    availableSuppliers = Union[SaxoOrderInterface]
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

    def __init__(self, ticker_name: str, ticker_parameters):
        self.Name = ticker_name
        self.Supplier = ticker_parameters['supplier']
        self.UpdatableTime = int(ticker_parameters['updatable_time'])

    def download_history(self, numberOfDotes):
        if self.Supplier:
            history = self.Supplier.get_asset_data_hist(ticker=self.Name, density=self.UpdatableTime,
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
            self._cachedCollectedTime = self.Time[-1]
            del history
            print(
                f'{Update_log}Successfully downloaded last {lookBack} dotes for {basicAsset.Name} with last time {basicAsset._cachedCollectedTime}')

        else:
        warnings.warn('No brokerInterface plugged')
        return f"{Error_log}No brokerInterface"