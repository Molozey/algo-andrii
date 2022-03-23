from mainScript import _estimator, create_strategy_config, strategy_real_time_optimize

import threading
import sys
import pandas as pd
import platform
import numpy as np
from sklearn.model_selection import ParameterGrid
global RecursionBorder

if platform.platform().split('-')[0] == 'macOS':
    systemDivide = '/'
else:
    systemDivide = '\\'

threading.stack_size(2**27)
sys.setrecursionlimit(10 ** 5)

pairName = 'EURCHF.csv'

inpData = pd.read_csv(f"../testData{systemDivide}{pairName}", index_col=1)
# Какие колонки нужны для работы
columns = ['open', 'high', 'low', 'close']
inpData = inpData[columns]
inpData.index = pd.to_datetime(inpData.index)
inpData = inpData.loc['2021-01-01':]

RecursionBorder = 1000
grid_params = {
    # Оптимизировать !!!
    'yThreshold': [50],
    # Оптимизировать !
    'varianceRatioFilter': [1.4],
    'reverseVarianceRatioFilter': [1.0],
    # Оптимизировать !!
    'restAfterLoss': [324],
    # Оптимизировать !
    'restAfterFatProfit': [1],
    # Оптимизировать !!
    'stopLossStdMultiplier': [108_00],
    # Оптимизировать !
    'takeProfitStdMultiplier': [4050],
    # Оптимизировать !!
    'varianceRatioCarreteParameter': [18],
    # Оптимизировать !!!
    'scanHalfTime': [500],
    # Оптимизировать 0!
    'halfToFat': [2],
    # Оптимизировать 0!
    'halfToLight': [1],
    # Оптимизировать 0!
    'halfToTime': [2],
    }
grid_params = ParameterGrid(grid_params)
grid_params = pd.DataFrame(grid_params).sample(frac=1, random_state=9).reset_index(drop=True)

# totalResult, RealTimeOptimizeTrades, paramsEvolution = strategy_real_time_optimize(realTimeData=inpData.iloc[:400_000].copy(), parameters=create_strategy_config(grid_params.iloc[0]), show=True)
totalResult, RealTimeOptimizeTrades, paramsEvolution = strategy_real_time_optimize(realTimeData=inpData.iloc[:400_000].copy(), parameters=create_strategy_config(grid_params.iloc[0]), show=False)
print('=========z============')
print(totalResult)
print('=========z============')
print(RealTimeOptimizeTrades)
