import pandas as pd
import pymysql as SQL
from pymysql.err import IntegrityError
from config import Host, User, Password, DataBaseName

Available_ST = pd.read_csv('Available_stocks.csv')

db = SQL.connect(host=Host, user=User, password=Password, database=DataBaseName)
cursor = db.cursor()

for stock_index in range(Available_ST.shape[0])[:75]:
    stock = Available_ST.iloc[stock_index]
    Symbol = stock.Symbol
    Name = stock.Name
    MarketCap = stock['Market Cap']
    Sector = stock.Sector
    if str(Sector) == 'nan':
        Sector = 'Not Available...'
    Industry = stock.Industry
    Country = stock.Country
    if str(Country) == 'nan':
        Country = 'Unknown Country...'
    Price = round(float(stock['Last Sale'][1:]), 2)
    try:
        cursor.execute(""" INSERT INTO Ticker_Information (Ticker, Name, Last_Price, Country)
                            VALUES (%s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            Last_Price = %s;""", (Symbol, Name, Price, Country, Price))
        db.commit()

        cursor.execute(""" INSERT INTO Ticker_Sector (Ticker, Sector)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE
                            Sector = VALUES(Sector);""", (Symbol, Sector))
        db.commit()
    except IntegrityError:
        db.rollback()
        print(f"Trouble with {Symbol}")



db.close()