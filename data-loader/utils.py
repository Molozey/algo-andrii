from datetime import date, datetime, timedelta
from ib_insync import util, IB, Future, Forex, Stock
from sqlalchemy import create_engine
import pandas as pd

sql_hostname = '23.23.90.225'
sql_username = 'research'
sql_password = '0xResearch'
sql_main_database = 'ib_data'
sql_port = '3306'

class SQLConnector:
    
    def connect(self, hostname, username, password, database, port='3306'):
        self.engine = create_engine(f'mysql+pymysql://{username}:{password}@{hostname}:{port}/{database}')
        self.conn = self.engine.raw_connection()
        self.updateIndex()
    
    def updateIndex(self):
        self.index = pd.read_sql_query(
            """select main_index.id, 
                main_index.name, 
                main_index.type as type_id, 
                instruments_types.name as type_name 
                from main_index 
                join instruments_types 
                on main_index.type = instruments_types.id;""", self.conn)
    
    def getInstrumentId(self, instrument):
        if (self.index['name'] == instrument).any() == False:
            return None
        return self.index[self.index['name'] == instrument]['id'].iloc[0]
    
    def createInstrument(self, instrument, type_):
        """Create instrument or do nothing if already exists and return it's id

        Args:

        instrument -- name of the instrument (string)

        type_ -- type of the instrument (int)
        """
        self.updateIndex()
        if (self.index['name'] == instrument).any() == False:
            if (self.index['type_name'] == type_.upper()).any() == False:
                type_id = None
            else:
                type_id = self.index[self.index['type_name'] == type_.upper()]['type_id'].iloc[0]
            if type_id is None:
                print(f"Type {type_id} not found!")
                return
            query = f"insert into main_index(type, name) values ({type_id}, '{instrument}');"
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
            self.updateIndex()
        return self.index[self.index['name'] == instrument]['id'].iloc[0]

    def saveData(self, bars):
        bars.to_sql('bars', self.engine, if_exists='append', index=False)
    
    def loadDaysSet(self, instrument):
        id = self.getInstrumentId(instrument)
        if id is None:
            return set()
        query = f"select distinct(date(time)) as date from bars where instrument = {id}; "
        return  set(pd.read_sql_query(query, self.conn)['date'])
        
    def loadData(self, instrument, start, end):
        id = self.getInstrumentId(instrument)
        if id is None:
            return pd.DataFrame(columns=['instrument','date','open','high','low','close','volume','average','barCount'])
        query = f"""select * 
                    from bars 
                    where 
                    instrument = {id}"""
        if start != None:
            query += f" and `time` >= '{start.strftime('%Y-%m-%d')+' 00:00:00'}' "
        if end != None:
            query += f" and `time` <= '{end.strftime('%Y-%m-%d')+' 23:55:55'}' "
        
        return pd.read_sql_query(query, self.conn)
    
    def loadIndex(self, update=False):
        if update:
            self.updateIndex()
        return self.index


class IBConnector():

    def connect(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)

    def loadData(self, contract, start, end, barSize='1 min'):
        df = pd.DataFrame(columns=['instrument','date','open','high','low','close','volume','average','barCount'])
        while end >= start:
            print('Got ' +end.strftime('%Y-%m-%d'))
            bars = self.ib.reqHistoricalData(
                contract, endDateTime=end.strftime('%Y%m%d')+' 23:59:00', 
                durationStr='1 D', barSizeSetting=barSize, whatToShow='MIDPOINT', useRTH=True)
            end -= timedelta(days=1)
            df = df.append(util.df(bars))
        df.rename(columns={'date' : 'time'}, inplace=True)
        return df.sort_values('time', axis=0).drop_duplicates(subset=['time'])
    
    def loadDataDaysList(self, contract, days, barSize='1 min'):
        df = pd.DataFrame(columns=['date','open','high','low','close','volume','average','barCount'])
        for day in days:
            print('Got ' +day)
            day = day.replace('-', '')
            day = day.replace(' ', '-')
            bars = self.ib.reqHistoricalData(
                contract, endDateTime=day+' 23:59:00', 
                durationStr='1 D', barSizeSetting=barSize, whatToShow='MIDPOINT', useRTH=True)
            df = df.append(util.df(bars))
        df.rename(columns={'date' : 'time'}, inplace=True)
        return df.sort_values('time', axis=0).drop_duplicates(subset=['time'])
    
    def loadDataDaysList(self, instrument, type_, exchange, days, barSize='1 min'):
        df = pd.DataFrame(columns=['date','open','high','low','close','volume','average','barCount'])
        type_dict = {'future' : 'FUT', 'stock' : 'STK', 'etf' : 'STK', 'forex' : 'CASH'}
        if type_ not in type_dict:
            print('type ' + type_ + ' not found')
            return df
        else:
            type_ = type_dict[type_.lower()]

        if type_ == 'FUT':
            instrument_name, date_of_expiration = instrument.split('-')
            exp_year, exp_month = date_of_expiration.split('.')
            contract = Future(instrument_name, '20'+exp_year+exp_month, 'GLOBEX', includeExpired=True)
        else:
            contract = Stock(instrument)
            
        for day in days:
            print('Got ' +day)
            day = day.replace('-', '')
            day = day.replace(' ', '-')
            bars = self.ib.reqHistoricalData(
                contract, endDateTime=day+' 23:59:00', 
                durationStr='1 D', barSizeSetting=barSize, whatToShow='MIDPOINT', useRTH=True)
            df = df.append(util.df(bars))
        df.rename(columns={'date' : 'time'}, inplace=True)
        return df.sort_values('time', axis=0).drop_duplicates(subset=['time'])
    
    def loadActiveFutures(self, instrument, days, endDate, rolloutInterval, barSize='1 min'):
        days = sorted(list(days))
        contract = Future(instrument, endDate.strftime('%Y%m'), 'GLOBEX', includeExpired=True)
        df = pd.DataFrame(columns=['date','open','high','low','close','volume','average','barCount'])
        for day in days[::-1]:
            while day >= endDate:
                endDate -= rolloutInterval
                contract = Future(instrument, endDate.strftime('%Y%m'), 'GLOBEX', includeExpired=True)
            bars = self.ib.reqHistoricalData(
                contract, endDateTime=day.strftime('%Y%m%d')+' 23:59:00', 
                durationStr='1 D', barSizeSetting=barSize, whatToShow='MIDPOINT', useRTH=True)
            df = df.append(util.df(bars))
        df.rename(columns={'date' : 'time'}, inplace=True)
        return df.sort_values('time', axis=0).drop_duplicates(subset=['time'])
    
    def loadData(self, instrument, type_, days, type_params, barSize='1 min'):
        if type_.lower() == 'future':
            data = self.loadActiveFutures(instrument, days, type_params['enddate'], type_params['interval'], barSize)
        else:
            if type_.lower() == 'forex':
                contract = Forex(instrument)
            elif type_.lower() == 'stock':
                contract = Stock(instrument)
            else:
                print('For now only Forex, Stock and Future are implemented')
                return None
            data = self.loadDataDaysList(contract, days, barSize)
        return data

def timeSectorToDates(start, end):
    l = {end}
    while end > start:
        if end.weekday() <= 4:
            # l.add(end.strftime('%Y-%m-%d'))
            l.add(end)
        end -= timedelta(days=1)
    return l

def saveDF(df, instrument, filename=None):
    if filename is None:
        filename = instrument+'.csv'
    df.to_csv(filename, index=False)
    print(f'Saved to {filename}')



