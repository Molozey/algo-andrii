import pymysql as SQL
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

Host = "82.148.19.206"
User = "molozey"
Password = "Exeshnik08!"
DataBaseName = "FinTechData"

def FIFA(ticker_name:str, columns:list):
    """

    :param ticker_name: Ticker name which data need to be converted
    :param columns: Columns that may be invert to FIFA Scale
    :return: JSON with ticker and COLUMNS
    """

    def fifa_Scaler(value, OldMax:float, OldMin:float):
        """

        :param value: value to convert
        :param OldMax: right border (MAX RATE)
        :param OldMin: left border (MIN RATE)
        :return: Beautiful rescaled value + total_rating
        """
        NewMax = 100 #   MAX VALUE
        NewMin = 30 #   MIN VALUE. BE CAREFUL NONE VALUES == NEWMIN

        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        if value:
            NewValues = int((((value - OldMin) * NewRange) // OldRange) + NewMin)
        else:
            NewValues = NewMin
        return NewValues


    ret_dict = dict()
    ret_dict.update({'ticker': ticker_name})
    for COL in columns:
        db = SQL.connect(host=Host, user=User, password=Password, database=DataBaseName)
        cursor = db.cursor()

        F_INFO = cursor.execute(f"""SELECT MAX(Indicators.{COL}), MIN(Indicators.{COL})
                                    FROM Ticker_Information as Info
                                    JOIN Ticker_Sector as Sector USING (Ticker)
                                    JOIN Ticker_Indicators as Indicators USING (Ticker)""")
        F_INFO = cursor.fetchall()

        ticker_pos = cursor.execute(f"""SELECT Indicators.{COL}
                                        FROM Ticker_Indicators as Indicators WHERE Indicators.Ticker=%s""", ticker_name)
        ticker_pos = cursor.fetchone()[0]
        db.close()

        (ColMax, ColMin,) = F_INFO[0]
        new_value = fifa_Scaler(ticker_pos, OldMax=ColMax, OldMin=ColMin)
        ret_dict.update({COL: new_value})
        SUMMARY = 0
        counter = 0
        for _ in ret_dict.items():
            if _[0] != "ticker":
                counter += 1
                SUMMARY += ret_dict[_[0]]
        ret_dict.update({'total_rating': round(SUMMARY / counter)})

    return ret_dict

# print(list(FIFA('AACG', columns=['market_cap', 'earnings', 'revenue_growth', 'price_to_earnings']).values())[1:])
# print(FIFA('AACG', columns=['market_cap', 'earnings', 'revenue_growth', 'price_to_earnings'])["market_cap"])



def get_similar(ticker, Host, User, Password, DataBaseName):
    """

    :param ticker: Name ticker to find similar tickers
    :param Host: DataBase Host
    :param User: DataBase User
    :param Password: DataBase Password
    :param DataBaseName: DataBase Name
    :return: list of similar tickers
    """
    db = SQL.connect(host=Host, user=User, password=Password, database=DataBaseName)
    cursor = db.cursor()

    INFO = cursor.execute("""SELECT Indicators.*, Sector.Sector, Info.Country
                                FROM Ticker_Information as Info
                                JOIN Ticker_Sector as Sector USING (Ticker)
                                JOIN Ticker_Indicators as Indicators USING (Ticker)""")
    INFO = cursor.fetchall()
    INFO_COLUMNS = [_[0] for _ in cursor.description]
    db.close()

    #   SIMILAR NEIGHBORS

    DF_INFO = pd.DataFrame(INFO, columns=INFO_COLUMNS)
    scaler = MinMaxScaler()
    DF_INFO[['Last_Price', 'Market_Cap', 'Earnings', 'Revenue_Growth','Price_To_Earnings']] = scaler.fit_transform(DF_INFO[['Last_Price', 'Market_Cap', 'Earnings', 'Revenue_Growth','Price_To_Earnings']])
    encode = LabelEncoder()
    DF_INFO['Country'] = encode.fit_transform(DF_INFO['Country'])
    DF_INFO['Sector'] = encode.fit_transform(DF_INFO['Sector'])

    weights = [0.05, 0.05, 0.05, 0.1, 0.15, 0.3, 0.2]
    #neighbors = NearestNeighbors(n_neighbors=5, radius=1, metric_params={})
    neighbors = NearestNeighbors(algorithm='brute',
                            metric='wminkowski',
                            metric_params={'w': weights},
                            p=1,
                            n_jobs=1,
                            n_neighbors=5)
    neighbors.fit(DF_INFO.replace(np.NAN, -1)[['Last_Price', 'Market_Cap', 'Earnings', 'Revenue_Growth','Price_To_Earnings', 'Sector', 'Country']].values)
    TICKERS_LABELS = DF_INFO.Ticker
    obj = DF_INFO[DF_INFO.Ticker == ticker].replace(np.NAN, -1)[['Last_Price', 'Market_Cap', 'Earnings', 'Revenue_Growth','Price_To_Earnings', 'Sector', 'Country']].values[0]
    dist, labels = neighbors.kneighbors([obj], n_neighbors=5)
    DF_INFO = DF_INFO.iloc[labels[0]]
    return DF_INFO.iloc[1:].Ticker.values


# #[0.0015661645172121447 6.81310402006617e-06 0.04607668975686084 0.37574404761904767 -1 5 2]
# get_similar('AACG', Host=Host, Password=Password, User=User, DataBaseName=DataBaseName)

print(type(list(FIFA(ticker_name='AACG', columns=['market_cap', 'earnings', 'revenue_growth', 'price_to_earnings']).keys())[1]))