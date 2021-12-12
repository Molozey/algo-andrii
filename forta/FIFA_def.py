import pymysql as SQL
import pandas as pd
import numpy as np

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

print(list(FIFA('AACG', columns=['market_cap', 'earnings', 'revenue_growth', 'price_to_earnings']).values())[1:])
print(FIFA('AACG', columns=['market_cap', 'earnings', 'revenue_growth', 'price_to_earnings'])["market_cap"])
