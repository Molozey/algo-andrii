import http.client, urllib.parse
import json
import pprint
import yfinance
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sklearn.linear_model as SKmodels
import joblib


class Strategy:
    def __init__(self, dataFrame, hoursAvailable='all'):
        self.DEBUG = True
        self.BASE = 'CHFJPY'
        self.data = dataFrame
        self.data.index = pd.to_datetime(self.data.index)
        self.technicalData = self.data.copy()
        self.trainSize = 0.5
        self.model = None
        self.scalePandas = None
        self.TimeBorder = pd.Timedelta('300T')
        self.intTimeBorder = self.TimeBorder // '1T'
        self.Y_threshold = 1  # In sigmas
        self.scanTime = pd.Timedelta('400T')  # For collecting half-time
        self.intScanTime = self.scanTime // '1T'
        self.scanTimeSimplifier = pd.Timedelta('1T')
        self.intScanTimeSimplifier = self.scanTimeSimplifier // '1T'
        self.intBbandsNanShift = self.intScanTimeSimplifier + self.intScanTime
        if hoursAvailable == 'all':
            self.availableToTradeHours = [i for i in range(0, 25)]
        else:
            self.availableToTradeHours = hoursAvailable

        self.idxPos = self.data.index

        def _applyer(x, inSID):
            if self.idxPos.get_loc(x.name) % inSID.intScanTimeSimplifier != 0:
                return -666
            if (self.idxPos.get_loc(x.name)) > self.intScanTime:
                POS = self.idxPos.get_loc(x.name)
                return self.get_half_time(inSID.technicalData[f"{self.BASE}_Open"].iloc[POS - self.intScanTime:POS])
            else:
                return np.nan

        self.technicalData["HalfTime"] = self.technicalData[f"{self.BASE}_Open"].to_frame().apply(
            lambda x: _applyer(x, inSID=self), axis=1)
        self.technicalData["HalfTime"].replace(to_replace=-666, method='ffill', inplace=True)
        self.technicalData[["Mean", "Std", "HighBand", "LowBand"]] = self.technicalData[
            f"{self.BASE}_Open"].to_frame().apply(lambda x: self.calculate_BBands(x.name), axis=1, result_type='expand')

    def calculate_BBands(self, position):
        _pos = (self.idxPos.get_loc(position))
        if (_pos < self.intBbandsNanShift) or (_pos < self.technicalData["HalfTime"].loc[position]):
            return np.nan, np.nan, np.nan, np.nan

        try:
            roll = self.technicalData[f"{self.BASE}_Open"].rolling(int(self.technicalData["HalfTime"].loc[position]))
        except ValueError:
            if self.DEBUG:
                print('Incorrect HalfTime')
            return np.nan, np.nan, np.nan, np.nan

        mean, std = roll.mean().loc[position], roll.std().loc[position]
        HighB = self.Y_threshold * std + mean
        LowB = mean - self.Y_threshold * std
        return mean, std, HighB, LowB

    def get_half_time(self, openTuple: pd.Series) -> float:
        """
        Функция отдающая период полураспада
        :param openTuple:
        :return:
        """
        df_open = openTuple.to_frame()
        df_lag = df_open.shift(1)
        df_delta = df_open - df_lag
        linear_regression_model = LinearRegression()
        df_delta = df_delta.values.reshape(len(df_delta), 1)
        df_lag = df_lag.values.reshape(len(df_lag), 1)
        linear_regression_model.fit(df_lag[1:], df_delta[1:])
        half_life = -np.log(2) / linear_regression_model.coef_.item()
        return int(half_life)

    def train_model(self):
        train, test = sklearn.model_selection.train_test_split(self.data, test_size=self.trainSize,
                                                               train_size=1 - self.trainSize)

        self.scalePandas = MinMaxScaler()
        train[train.columns] = self.scalePandas.fit_transform(train[train.columns])
        trainShifted = train[['CHFJPY_Mean', 'CHFJPY_Vol', 'VIX_Vol', 'VIX_Mean', 'GOLD_Vol', 'GOLD_Mean']]
        SHIFT_ARRAY = [5, 10, 15, 20, 40, 60]
        trainColumns = trainShifted.columns
        for sf in SHIFT_ARRAY:
            shifted = trainShifted.shift(sf)
            for column in trainColumns:
                trainShifted[f"SHIFTED_{sf}_{column}"] = shifted[column]
        del trainColumns
        trainShifted = trainShifted.iloc[max(SHIFT_ARRAY):]
        train = train.iloc[max(SHIFT_ARRAY):]
        train = train.merge(trainShifted[list(filter(lambda x: x not in train.columns, trainShifted.columns))],
                            left_index=True, right_index=True)
        trainData = train.drop(['Target', 'CHFJPY_Mean_Future'], axis=1).values
        trainTarget = train.Target
        classifier = SKmodels.RidgeClassifier(alpha=.3, solver='lsqr')
        classifier.fit(X=trainData, y=trainTarget)
        return pd.Series(index=list(train.drop(['Target', 'CHFJPY_Mean_Future'], axis=1).columns),
                         data=classifier.coef_[0])


strategy = Strategy(dataFrame=pd.read_csv('data.csv', index_col=0).iloc[:800])
strategy.train_model()