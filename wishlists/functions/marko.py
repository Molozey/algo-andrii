from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import pymysql as SQL
from DataBaseModuleFiller.config import Host, Password, DataBaseName, User

import json
"""
# Read in price data
df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
"""
def MARKO(tickers):
    db = SQL.connect(host=Host, user=User, password=Password, database=DataBaseName)
    cursor = db.cursor()
    tickers_list = tickers
    df = pd.DataFrame()
    for ticker in tickers_list:
        F_INFO = cursor.execute(f"""SELECT Ticker, Day_Prices FROM
                                    Ticker_Prices WHERE Ticker_Prices.Ticker=%s""", ticker)
        F_INFO = cursor.fetchall()
        NAME = F_INFO[0][0]
        df[f"{NAME}"] = pd.DataFrame.from_dict(json.loads(F_INFO[0][1]), orient='index')

    NAN_LOGIC = df.isnull().sum(axis=0) / df.shape[0]
    NAN_LOGIC = NAN_LOGIC[NAN_LOGIC > 0.1].index
    for columns in NAN_LOGIC:
        df.drop(f"{columns}", axis=1, inplace=True)
    if len(df.columns) > 2:
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        #ef.portfolio_performance(verbose=True)
        weights = weights
        #js = json.load(json.dumps(weights))
        js = dict(weights)
        ret = list()
        for _ in js.items():
            ret.append({'ticker': _[0],
                        'weight': _[1]})
        return ['+', ret]
    else:
        return ['-', None]

#print(MARKO(tickers=['AAL', 'AACIW', 'AAON']))