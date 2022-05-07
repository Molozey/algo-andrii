import http.client, urllib.parse
import json
import pprint
import yfinance
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sklearn.linear_model as SKmodels
import joblib
import warnings
from tqdm import tqdm


class Strategy:
    def _download_all_data(self):
        API_STOCK_DATA = "Ab0Eq9OSOkVaNe9eluoMjkxSgnrFk46h08ghta6w"
        API_ALPHA_VANTAGE = "Y55JGW1SYPKKETDD"
        # Download VIX information
        ticker = yfinance.Ticker('^VIX')
        VIX = ticker.history(interval='1m', period='7d')
        VIX.index = pd.to_datetime(VIX.index).tz_convert(tz='Europe/London')
        VIX.columns = [f"VIX_{_}" for _ in VIX.columns]
        plt.plot(VIX.VIX_Open, '--')
        VIX = VIX[["VIX_Open"]]

        # Download CHFJPY information
        ticker = yfinance.Ticker('CHFJPY=X')
        CHFJPY = ticker.history(interval='1m', period='7d').iloc[:-15]
        CHFJPY.index = pd.to_datetime(CHFJPY.index).tz_convert(tz='Europe/London')
        CHFJPY.columns = [f"CHFJPY_{_}" for _ in CHFJPY.columns]
        plt.plot(CHFJPY.CHFJPY_Open, '--')
        CHFJPY = CHFJPY[["CHFJPY_Open"]]

        # Download Gold information
        ticker = yfinance.Ticker('GC=F')
        GOLD = ticker.history(interval='1m', period='7d').iloc[:-15]
        GOLD.index = pd.to_datetime(GOLD.index).tz_convert(tz='Europe/London')
        GOLD.columns = [f"GOLD_{_}" for _ in GOLD.columns]
        plt.plot(GOLD.GOLD_Open, '--')
        GOLD = GOLD[["GOLD_Open"]]

        merged = CHFJPY.merge(VIX, left_index=True, right_index=True, how='left')
        nan = np.any(merged.isna(), axis=1)
        nan = nan.where(np.logical_not(nan)).first_valid_index()
        merged = merged.loc[nan:]
        merged['FiniteVixData'] = np.any(merged.isna(), axis=1)[nan:]
        merged[VIX.columns] = merged[VIX.columns].fillna(method='ffill')

        merged = merged.merge(GOLD, left_index=True, right_index=True, how='left')
        nan = np.any(merged.isna(), axis=1)
        nan = nan.where(np.logical_not(nan)).first_valid_index()
        merged = merged.loc[nan:]
        merged['FiniteGOLDData'] = np.any(merged.isna(), axis=1)[nan:]
        merged[GOLD.columns] = merged[GOLD.columns].fillna(method='ffill')
        # merged.drop(['CHFJPY_Volume', 'CHFJPY_Dividends',
        #              'CHFJPY_Stock Splits', 'VIX_Volume', 'VIX_Dividends', 'VIX_Stock Splits',
        #              'GOLD_Stock Splits', 'GOLD_Dividends', 'GOLD_Volume'],
        #             axis=1, inplace=True)

        del nan
        # Create Mean by rolling on rollParam value.
        rollParam = self.ROLLING_FEATURE

        CHFJPY_roll = merged.rolling(rollParam)['CHFJPY_Open']
        merged['CHFJPY_Vol'] = CHFJPY_roll.std()
        merged['CHFJPY_Mean'] = CHFJPY_roll.mean()

        VIX_roll = merged.rolling(rollParam)['VIX_Open']
        merged['VIX_Vol'] = VIX_roll.std()
        merged['VIX_Mean'] = VIX_roll.mean()

        GOLD_roll = merged.rolling(rollParam)['GOLD_Open']
        merged['GOLD_Vol'] = GOLD_roll.std()
        merged['GOLD_Mean'] = GOLD_roll.mean()

        merged = merged.iloc[rollParam:]

        predictionMean = self.intTimeBorder
        merged['CHFJPY_Mean_Future'] = merged['CHFJPY_Mean'].shift(-predictionMean)
        assert merged['CHFJPY_Mean_Future'].iloc[0] == merged['CHFJPY_Mean'].iloc[0+predictionMean]
        merged = merged.iloc[:-predictionMean]
        del CHFJPY_roll, VIX_roll, GOLD_roll
        merged['Target'] = merged.apply(lambda x: 1 if x.CHFJPY_Mean_Future > x['CHFJPY_Mean'] else 0, axis=1)
        merged.to_csv('realTimeUpdate.csv')
        return merged

    # TODO: Сделать закачку данных
    def __init__(self, dataFrame=None, hoursAvailable='all', TimeB='300T', RollFeature='300T'):
        self.DEBUG = True
        self.BASE = 'CHFJPY'
        self.model = None
        self.scalePandas = None

        self.testData = None
        self.testIDX = None

        self.testSHIFTED = None
        self.testSHIFTEDIDX = None
        self.testSHIFTED_TARGET = None

        self.restAfterLoss = pd.Timedelta('30T')
        self.intRestAfterLoss = self.restAfterLoss // '1T'

        self.ROLLING_FEATURE = pd.Timedelta(f'{RollFeature}') // '1T'
        self.TimeBorder = pd.Timedelta(f'{TimeB}')
        self.intTimeBorder = self.TimeBorder // '1T'
        self.Y_threshold = 1    # In sigmas
        self.scanTime = pd.Timedelta('400T')  # For collecting half-time
        self.intScanTime = self.scanTime // '1T'
        self.scanTimeSimplifier = pd.Timedelta('1T')
        self.intScanTimeSimplifier = self.scanTimeSimplifier // '1T'
        self.intBbandsNanShift = self.intScanTimeSimplifier + self.intScanTime

        if dataFrame is None:
            self.data = self._download_all_data()
        else:
            self.data = dataFrame
        self.data.index = pd.to_datetime(self.data.index)
        self.dataIDX = self.data.index

        self.technicalData = self.data.copy()
        self.technicalDataTestBatch = None
        self.trainSize = 0.75

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
                return self.get_half_time(inSID.technicalData[f"{self.BASE}_Open"].iloc[POS-self.intScanTime:POS])
            else:
                return np.nan
        self.technicalData["HalfTime"] = self.technicalData[f"{self.BASE}_Open"].to_frame().apply(lambda x: _applyer(x, inSID=self), axis=1)
        self.technicalData["HalfTime"].replace(to_replace = -666,  method='ffill', inplace=True)
        self.technicalData[["Mean", "Std", "HighBand", "LowBand"]] = self.technicalData[f"{self.BASE}_Open"].to_frame().apply(lambda x: self.calculate_BBands(x.name), axis=1, result_type='expand')
        self.technicalDataTestBatch = sklearn.model_selection.train_test_split(self.technicalData, test_size=1 - self.trainSize,
                                                               train_size=self.trainSize,
                                                               shuffle=False)[1]
    def calculate_BBands(self, position):
        _pos = (self.idxPos.get_loc(position))
        if (_pos < self.intBbandsNanShift) or (_pos < self.technicalData["HalfTime"].loc[position]):
            return np.nan, np.nan, np.nan, np.nan

        try:
            roll = self.technicalData[f"{self.BASE}_Open"].rolling(int(self.technicalData["HalfTime"].loc[position]))
        except ValueError:
            if self.DEBUG:
                pass
                # print('Incorrect HalfTime')
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
        train, test = sklearn.model_selection.train_test_split(self.data, test_size=1 - self.trainSize,
                                                               train_size=self.trainSize,
                                                               shuffle=False)
        # print(train.iloc[0], train.iloc[-1])
        assert train.iloc[-1].name + pd.Timedelta('1T') == test.iloc[0].name
        # print(test.iloc[0], test.iloc[-1])
        self.testData = test
        self.testIDX = test.index
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
        train = train.merge(trainShifted[list(filter(lambda x: x not in train.columns, trainShifted.columns))], left_index=True, right_index=True)
        trainData = train.drop(['Target', 'CHFJPY_Mean_Future'], axis=1).values
        trainTarget = train.Target
        classifier = SKmodels.RidgeClassifier(alpha=.3, solver='lsqr')
        classifier.fit(X=trainData, y=trainTarget)
        self.model = classifier

        TEST = self.data.copy()
        TEST[TEST.columns] = self.scalePandas.transform(TEST[TEST.columns])
        TESTShifted = TEST[['CHFJPY_Mean', 'CHFJPY_Vol', 'VIX_Vol', 'VIX_Mean', 'GOLD_Vol', 'GOLD_Mean']]
        SHIFT_ARRAY = [5, 10, 15, 20, 40, 60]
        TESTColumns = TESTShifted.columns
        for sf in SHIFT_ARRAY:
            shifted = TESTShifted.shift(sf)
            for column in TESTColumns:
                TESTShifted[f"SHIFTED_{sf}_{column}"] = shifted[column]
                assert TESTShifted[f"SHIFTED_{sf}_{column}"].iloc[max(SHIFT_ARRAY)] == TESTShifted[column].iloc[max(SHIFT_ARRAY)-sf]
        del TESTColumns
        TESTShifted = TESTShifted.iloc[max(SHIFT_ARRAY):]
        TEST = TEST.iloc[max(SHIFT_ARRAY):]
        TEST = TEST.merge(TESTShifted[list(filter(lambda x: x not in TEST.columns, TESTShifted.columns))], left_index=True, right_index=True)

        self.testSHIFTED_TARGET = TEST['Target']
        self.testSHIFTED = TEST.drop(['Target', 'CHFJPY_Mean_Future'], axis=1)
        self.testSHIFTEDIDX = self.testSHIFTED.index
        TESTTarget = TEST.Target
        if self.DEBUG:
            pass
            # print(TEST.iloc[0,:])
            # print(TEST.iloc[TEST.index.get_loc(self.testData.iloc[0].name):])
        SCORER = sklearn.model_selection.train_test_split(self.testSHIFTED, test_size=1 - self.trainSize,
                                                               train_size=self.trainSize,
                                                               shuffle=False)[1]

        SCORER_TARGET = sklearn.model_selection.train_test_split(self.testSHIFTED_TARGET, test_size=1 - self.trainSize,
                                                               train_size=self.trainSize,
                                                               shuffle=False)[1]

        return self.model.score(X=SCORER.values, y=SCORER_TARGET.values), pd.Series(index=list(train.drop(['Target', 'CHFJPY_Mean_Future'], axis=1).columns), data=classifier.coef_[0])

    def _open_position_ability(self, x):
        PreProcessedData = self.testSHIFTED.iloc[self.testSHIFTEDIDX.get_loc(x.name)]
        assert PreProcessedData.name == x.name

        assert 'CHFJPY_Mean_Future' not in PreProcessedData.index
        assert 'Target' not in PreProcessedData.index
        predictor = self.model.predict(X=PreProcessedData.values.reshape(1,-1))
        # print('predictor:', predictor)
        # lambda x: 1 if x.CHFJPY_Mean_Future > x['CHFJPY_Mean'] else 0
        if (x.name.weekday() >= 5) or (x.name.weekday() < 2):
            return 0
        if x["HalfTime"] is not np.nan:
            if x["HighBand"] < x[f"{self.BASE}_Open"]:
                if x["CHFJPY_Vol"] < self.testSHIFTED.iloc[self.testSHIFTEDIDX.get_loc(x.name) - self.intTimeBorder:self.testSHIFTEDIDX.get_loc(x.name)]["CHFJPY_Vol"].mean():
                # if x["CHFJPY_Vol"] < self.testSHIFTED.iloc[:self.testSHIFTEDIDX.get_loc(x.name)]["CHFJPY_Vol"].mean():
                    if predictor == 0:
                        return -1
            if x["LowBand"] > x[f"{self.BASE}_Open"]:
                if x["CHFJPY_Vol"] < self.testSHIFTED.iloc[self.testSHIFTEDIDX.get_loc(x.name) - self.intTimeBorder:self.testSHIFTEDIDX.get_loc(x.name)]["CHFJPY_Vol"].mean():
                # if x["CHFJPY_Vol"] < self.testSHIFTED.iloc[:self.testSHIFTEDIDX.get_loc(x.name)]["CHFJPY_Vol"].mean():
                    if predictor == 1:
                        return 1
        else:
            return 0

        return 0

    def simulate_trading(self):
        self.technicalDataTestBatch['OPEN'] = self.technicalDataTestBatch.apply(lambda x: self._open_position_ability(x), axis=1)
        return self.technicalDataTestBatch


def SCORE(meanV, TimeB='100T'):
    warnings.filterwarnings("ignore")
    # strategy = Strategy(dataFrame=pd.read_csv('data.csv', index_col=0).iloc[:], TimeB=f"{meanV}T")
    strategy = Strategy(TimeB=TimeB, RollFeature=f"{int(meanV)}T")
    model = strategy.train_model()[0]
    return meanV, model


def create_predictor_power():
    for Time in [f"{x}T" for x in range(10, 400, 50)]:
        RES = joblib.Parallel(n_jobs=-1, verbose=10)(joblib.delayed(SCORE)(x, Time) for x in tqdm(np.linspace(10, 400, 16)))
        plt.figure(figsize=(12, 6))
        plt.title(f"{Time}")
        plt.plot([x[0] for x in RES], [x[1] for x in RES])
        plt.show()


def main():
    # strategy = Strategy(dataFrame=pd.read_csv('data.csv', index_col=0).iloc[:])
    strategy = Strategy(TimeB='110T', RollFeature='300T')
    print(strategy.train_model()[0])

    z = strategy.simulate_trading()

    global TradeVolume
    global SHIFT
    global THRESHOLD_LOSS
    global SLIPPADGE

    SHIFT = strategy.intTimeBorder

    SL_SLEEP = strategy.restAfterLoss

    TradeVolume = 100_0
    SLIPPADGE = .01

    THRESHOLD_LOSS = 1


    def make_profit(DF, DFIDX, x):
        # print(DF.iloc[DFIDX.get_loc(x.name + pd.Timedelta(f"{SHIFT}T"))])
        # if DFIDX.get_loc(x.name + pd.Timedelta(f"{SHIFT}T")) in DFIDX:

        # Так нельзя. Нужно запрещать торги только если сработал лосс на нужных данных
        # if (x.name + pd.Timedelta(f"{SHIFT}T") < DF.iloc[-1].name) and (DF["STOP_LOSS_LABEL"].iloc[DFIDX.get_loc(x.name)] != 1):

        if (x.name + pd.Timedelta(f"{SHIFT}T") < DF.iloc[-1].name):
            # self.technicalData[["Mean", "Std", "HighBand", "LowBand"]] = self.technicalData[f"{self.BASE}_Open"].to_frame().apply(lambda x: self.calculate_BBands(x.name), axis=1, result_type='expand')
            # NEXT_DOT = DF.iloc[DFIDX.get_loc(x.name) + int(x["HalfTime"])]
            # assert x.name + pd.Timedelta(f"{int(x['HalfTime'])}T") == NEXT_DOT.name
            # print(x.name + pd.Timedelta(f"{SHIFT}T"))
            try:
                if x["OPEN"] != 0:
                    NEXT_DOT = DF.iloc[DFIDX.get_loc(x.name + pd.Timedelta(f"{SHIFT}T"))]
                    assert x.name + pd.Timedelta(f"{SHIFT}T") == NEXT_DOT.name

                if x["OPEN"] == 1:
                    CROSS_MEAN = DF.iloc[DFIDX.get_loc(x.name) + int(x["HalfTime"]):DFIDX.get_loc(x.name) + max(int(SHIFT), int(x["HalfTime"]))]
                    # Stop loss
                    if CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open < x['CHFJPY_Open'] * (1 - THRESHOLD_LOSS)).first_valid_index() is not None:
                        CLOSE_PRICE = CROSS_MEAN.loc[CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open < x['CHFJPY_Open'] * (1 - THRESHOLD_LOSS)).first_valid_index()]["CHFJPY_Open"]
                        print('Stop loss:', (TradeVolume // x["CHFJPY_Open"]) * (CLOSE_PRICE - x["CHFJPY_Open"]) - SLIPPADGE)
                        DF.loc[x.name:x.name + SL_SLEEP]["STOP_LOSS_LABEL"] = 1
                        return (TradeVolume // x["CHFJPY_Open"]) * (CLOSE_PRICE - x["CHFJPY_Open"]) - SLIPPADGE, CLOSE_PRICE
                    # Mean crossing
                    if CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open > x['CHFJPY_Mean']).first_valid_index() is not None:
                        CLOSE_PRICE = CROSS_MEAN.loc[CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open > x['CHFJPY_Mean']).first_valid_index()]["CHFJPY_Open"]
                        return (TradeVolume // x["CHFJPY_Open"]) * (CLOSE_PRICE - x["CHFJPY_Open"]) - SLIPPADGE, CLOSE_PRICE
                    return (TradeVolume // x["CHFJPY_Open"]) * (NEXT_DOT["CHFJPY_Open"] - x["CHFJPY_Open"]) - SLIPPADGE, NEXT_DOT["CHFJPY_Open"]

                if x["OPEN"] == 0:
                    return 0, x["CHFJPY_Open"]

                if x["OPEN"] == -1:
                    CROSS_MEAN = DF.iloc[DFIDX.get_loc(x.name) + int(x["HalfTime"]):DFIDX.get_loc(x.name) + max(int(SHIFT), int(x["HalfTime"]))]
                    # Stop loss
                    if CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open > x['CHFJPY_Open'] * (1 + THRESHOLD_LOSS)).first_valid_index() is not None:
                        CLOSE_PRICE = CROSS_MEAN.loc[CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open > x['CHFJPY_Open'] * (1 + THRESHOLD_LOSS)).first_valid_index()]["CHFJPY_Open"]
                        print('Stop loss:', (TradeVolume // x["CHFJPY_Open"]) * (x["CHFJPY_Open"] - CLOSE_PRICE) - SLIPPADGE)
                        DF.loc[x.name:x.name + SL_SLEEP]["STOP_LOSS_LABEL"] = 1
                        return (TradeVolume // x["CHFJPY_Open"]) * (x["CHFJPY_Open"] - CLOSE_PRICE) - SLIPPADGE, CLOSE_PRICE
                    # Mean crossing
                    if CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open < x['CHFJPY_Mean']).first_valid_index() is not None:
                        CLOSE_PRICE = CROSS_MEAN.loc[CROSS_MEAN.where(CROSS_MEAN.CHFJPY_Open < x['CHFJPY_Mean']).first_valid_index()]["CHFJPY_Open"]
                        return (TradeVolume // x["CHFJPY_Open"]) * (x["CHFJPY_Open"] - CLOSE_PRICE) - SLIPPADGE, CLOSE_PRICE


                    return (TradeVolume // x["CHFJPY_Open"]) * (x["CHFJPY_Open"] - NEXT_DOT["CHFJPY_Open"]) - SLIPPADGE, NEXT_DOT["CHFJPY_Open"]
            except KeyError:
                print('KEY ERROR')
                NEXT_DOT = DF.iloc[DFIDX.get_loc(x.name) + int(x["HalfTime"]):].iloc[0]
                if x["OPEN"] == 1:
                    # print((TradeVolume // x["CHFJPY_Open"]) * (NEXT_DOT["CHFJPY_Open"] - x["CHFJPY_Open"]) - 3, NEXT_DOT["CHFJPY_Open"])
                    return (TradeVolume // x["CHFJPY_Open"]) * (NEXT_DOT["CHFJPY_Open"] - x["CHFJPY_Open"]) - SLIPPADGE, NEXT_DOT["CHFJPY_Open"]
                if x["OPEN"] == 0:
                    return 0, x["CHFJPY_Open"]
                if x["OPEN"] == -1:
                    # print((TradeVolume // x["CHFJPY_Open"]) * (x["CHFJPY_Open"] - NEXT_DOT["CHFJPY_Open"]) - 3, NEXT_DOT["CHFJPY_Open"])
                    return (TradeVolume // x["CHFJPY_Open"]) * (x["CHFJPY_Open"] - NEXT_DOT["CHFJPY_Open"]) - SLIPPADGE, NEXT_DOT["CHFJPY_Open"]

        else:
            return 0, x["CHFJPY_Open"]
    z["STOP_LOSS_LABEL"] = 0
    df = z
    dfIDX = z.index
    z[["PROFIT", "CLOSE_PRICE"]] = z.apply(lambda x: make_profit(df, dfIDX, x), axis=1, result_type='expand')
    # z.iloc[3650:].apply(lambda x: make_profit(df, dfIDX, x), axis=1).values
    fig, axs = plt.subplots(2, figsize=(12,12))
    axs[0].plot(z["PROFIT"].cumsum())
    axs[1].plot(z["CHFJPY_Vol"])
    fig.show()
    print(z[["PROFIT", "STOP_LOSS_LABEL"]].loc['2022-04-29 11:00:00':'2022-04-29 12:00:00'].head(60))


if __name__ == '__main__':
    main()