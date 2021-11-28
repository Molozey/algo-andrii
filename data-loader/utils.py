from datetime import datetime, timedelta
from ib_insync import util, IB
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
    
    def loadDaysList(self, instrument):
        id = self.getInstrumentId(instrument)
        if id is None:
            return set()
        query = f"select distinct(date(time)) as date from bars where instrument = {id}; "
        return set(pd.read_sql_query(query, self.conn)['date'].astype('string'))
        
    def loadData(self, instrument, start, end):
        id = self.getInstrumentId(instrument)
        if id is None:
            return pd.DataFrame(columns=['instrument','time','open','high','low','close','volume','average','barCount'])
        query = f"""select * 
                    from bars 
                    where 
                    instrument = {id} and 
                    `time` >= '{start.strftime('%Y-%m-%d %H:%m:%S')}' and 
                    `time` <= '{end.strftime('%Y-%m-%d %H:%m:%S')}'; """
        return pd.read_sql_query(query, self.conn)


class IBConnector():

    def connect(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)

    def loadData(self, contract, start, end, barSize='1 min'):
        df = pd.DataFrame(columns=['instrument','time','open','high','low','close','volume','average','barCount'])
        while end >= start:
            print('Got ' +end.strftime('%Y-%m-%d'))
            bars = self.ib.reqHistoricalData(
                contract, endDateTime=end.strftime('%Y%m%d %H:%m:%S'), 
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

def timeSectorToDateSet(start, end):
    l = {end.strftime('%Y-%m-%d')}
    while end > start:
        if end.weekday() <= 4:
            l.add(end.strftime('%Y-%m-%d'))
        end -= timedelta(days=1)
    return l

def saveDF(df, instrument, filename=None):
    if filename is None:
        filename = instrument+'.csv'
    df.to_csv(filename, index=False)
    print(f'Saved to {filename}')



