import pandas as pd
import pymysql as SQL
import pprint

from dateutil.relativedelta import relativedelta
import datetime

from tqdm import tqdm
from joblib import Parallel, delayed
import yfinance as yf
from config import Host, User, Password, DataBaseName

from pymysql.err import IntegrityError

Available_ST = pd.read_csv('Available_stocks.csv')

#db = SQL.connect(host='localhost', user='root', password='', database="FinTechProject")
#cursor = db.cursor()
ACTUAL_DATETIME = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ACTUAL_DAY = datetime.date.today()
BACK_DAY = ACTUAL_DAY - relativedelta(years=2)

Available_ST = Available_ST.iloc[:75]


def filler(stock_index, Available_ST, ACTUAL_DATETIME):
    db = SQL.connect(host=Host, user=User, password=Password, database=DataBaseName)
    cursor = db.cursor()
    stock = Available_ST.iloc[stock_index]
    Symbol = stock.Symbol
    ticker = yf.Ticker(Symbol)

    history = ticker.history(interval='1d', start=BACK_DAY, end=ACTUAL_DAY).Close
    history.index = history.index.astype(str)
    history = history.to_json()


    try:
        cursor.execute(""" INSERT INTO Ticker_Prices (Ticker, Update_Date, Day_Prices)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        Update_Date = %s,
                        Day_Prices = %s""", (Symbol, ACTUAL_DATETIME, history, ACTUAL_DATETIME, history))
        db.commit()

    except IntegrityError:
        db.rollback()
        print(f"Trouble with {Symbol}")
    db.commit()
    db.close()
    return 'ZUK'


assets = Parallel(n_jobs=-1)(delayed(filler)(stock_index, Available_ST, ACTUAL_DATETIME)
                             for stock_index in tqdm(range(Available_ST.shape[0]), leave=False, total=Available_ST.shape[0]))

