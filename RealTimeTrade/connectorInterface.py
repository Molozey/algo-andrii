from abc import ABC, abstractmethod
from saxo_openapi import API
from saxo_openapi.contrib.orders import (tie_account_to_order, MarketOrderFxSpot, StopOrderFxSpot)
from saxo_openapi.contrib.orders.onfill import TakeProfitDetails, StopLossDetails
from saxo_openapi.contrib.util import InstrumentToUic
from saxo_openapi.contrib.session import account_info
import saxo_openapi.endpoints.trading as tr
# import saxo_openapi.endpoints.portfolio as pf
from pprint import pprint
import json

class AbstractOrderInterface:
    def __init__(self):
        token = "eyJhbGciOiJFUzI1NiIsIng1dCI6IkRFNDc0QUQ1Q0NGRUFFRTlDRThCRDQ3ODlFRTZDOTEyRjVCM0UzOTQifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiVG1XWGlqam1ZdFk0ZmF0MkIwZDdYdz09IiwiY2lkIjoiVG1XWGlqam1ZdFk0ZmF0MkIwZDdYdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiNmRkMjBkMGYwYWI3NDZjZWE1NThiMzc1NWM3MzI4ZDkiLCJkZ2kiOiI4NCIsImV4cCI6IjE2NDk0MjEzOTMiLCJvYWwiOiIxRiJ9.Q0Y6xfW_2Hetns3SsRhV3LrGyddDW2TLeM5NUiGdGHw-TeXPa0M4SYWVU8FcifzmYLVm8PNv07n6TDZ2_T6wCg"
        
      
    @abstractmethod
    def instrument_to_uic(client, AccountKey, list_instruments):
        '''
        return: {CHFJPY': 8, 'EURUSD': 21, ...} 
        '''
        
        dict_uics = {}
        for instrument in list_instruments:
            spec = {'Instrument': [instrument]}
            try:
                dict_uics[instrument] = list(InstrumentToUic(client, AccountKey, spec=spec).values())[0]
            except Exception as error:
                print(error)
        return dict_uics
    
    @abstractmethod
    def get_actual_data(self):
        pass

    @abstractmethod
    def place_open_order(self):
        pass

    @abstractmethod
    def validate_open_order(self):
        pass

    @abstractmethod
    def place_close_order(self):
        pass

    @abstractmethod
    def validate_close_order(self):
        pass

# WHERE WE WILL PUT NEXT ? :
    # client to requests  process
    client = API(access_token=token)
    AccountKey = account_info(client).AccountKey
    
    
