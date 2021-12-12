import pandas as pd
import pymysql as SQL
import pprint
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
Available_ST = Available_ST.iloc[:75]


def filler(stock_index, Available_ST, ACTUAL_DATETIME):
    db = SQL.connect(host=Host, user=User, password=Password, database=DataBaseName)
    cursor = db.cursor()
    stock = Available_ST.iloc[stock_index]
    Symbol = stock.Symbol
    ticker = yf.Ticker(Symbol)
    INFO = ticker.info
    #pprint.pprint(INFO)

    if 'longBusinessSummary' in INFO:
        DESCR = INFO['longBusinessSummary']
    else:
        DESCR = None

    if 'currentPrice'in INFO:
        LAST_PRICE = INFO['currentPrice']
    else:
        LAST_PRICE = None

    if 'currency' in INFO:
        CURRENCY = INFO['currency']
    else:
        CURRENCY = None

    if 'logo_url' in INFO:
        LOGO = INFO['logo_url']
    else:
        LOGO = None

    if LOGO == '':
        LOGO = None

    try:
        cursor.execute(""" UPDATE Ticker_Information
                            SET Description = %s,
                            Logo_url = %s,
                            Last_Price = %s,
                            Update_Date = %s,
                            Currency = %s
                            WHERE Ticker = %s;""", (DESCR, LOGO, LAST_PRICE, ACTUAL_DATETIME, CURRENCY, str(stock.Symbol)))
        db.commit()
    except IntegrityError:
        db.rollback()
        print(f"Trouble with {Symbol}")

    if 'marketCap' in INFO:
        CAP = INFO["marketCap"]
    else:
        CAP = None

    if 'netIncomeToCommon' in INFO:
        EARNINGS = INFO['netIncomeToCommon']
    else:
        EARNINGS = None

    if 'revenueGrowth' in INFO:
        REVENUE_GROWTH = INFO['revenueGrowth']
    else:
        REVENUE_GROWTH = None

    if 'trailingPE' in INFO:
        PE = INFO['trailingPE']
    else:
        PE = None

    try:
        cursor.execute(""" INSERT INTO Ticker_Indicators (Ticker, Last_Price, Update_Date, Market_cap, Earnings, Revenue_Growth, Price_to_Earnings)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            Last_Price = %s,
                            Update_Date = %s,
                            Market_cap = %s,
                            Earnings = %s,
                            Revenue_Growth = %s,
                            Price_to_Earnings = %s""", (Symbol, LAST_PRICE, ACTUAL_DATETIME, CAP, EARNINGS, REVENUE_GROWTH, PE, LAST_PRICE, ACTUAL_DATETIME, CAP, EARNINGS, REVENUE_GROWTH, PE))
        db.commit()

    except IntegrityError:
        db.rollback()
        print(f"Trouble with {Symbol}")
    db.commit()
    db.close()
    return 'ZUK'


assets = Parallel(n_jobs=-1)(delayed(filler)(stock_index, Available_ST, ACTUAL_DATETIME)
                             for stock_index in tqdm(range(Available_ST.shape[0]), leave=False, total=Available_ST.shape[0]))

#db.close()