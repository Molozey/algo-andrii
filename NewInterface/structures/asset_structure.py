from dataclasses import dataclass
from typing import List
from datetime import datetime


@dataclass
class assetInformation:
    Name: str
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
    Time: List[datetime]
    _cachedCollectedTime: datetime

    def __init__(self, ticker_name: str):
        self.Name = ticker_name
