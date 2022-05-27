from sklearn import linear_model
import pandas as pd
import numpy as np


def get_half_time(openTuple: pd.Series) -> float:
    """
    Функция отдающая период полураспада
    :param openTuple:
    :return:
    """
    df_open = openTuple.to_frame()
    df_lag = df_open.shift(1)
    df_delta = df_open - df_lag
    linear_regression_model = linear_model.LinearRegression()
    df_delta = df_delta.values.reshape(len(df_delta), 1)
    df_lag = df_lag.values.reshape(len(df_lag), 1)
    linear_regression_model.fit(df_lag[1:], df_delta[1:])
    half_life = -np.log(2) / linear_regression_model.coef_.item()
    return half_life