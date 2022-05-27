from core.dataSuppliers._baseDataProvider import AbstractDataProvider
from saxo_openapi import API
from saxo_openapi.contrib.session import account_info
import saxo_openapi.endpoints.rootservices as rs
from saxo_openapi.contrib.util import InstrumentToUic
import saxo_openapi.endpoints.trading as tr
import saxo_openapi.endpoints.chart as chart
from core.connectors.SAXO import SaxoOrderInterface


class SaxoDataProvider(AbstractDataProvider, SaxoOrderInterface):
    def __init__(self, token):
        super(SaxoDataProvider, self).__init__(token=token)