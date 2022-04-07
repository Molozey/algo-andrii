from stableVersion import *
import pandas as pd

firstConditions = {
    # Оптимизировать !!!
    'yThreshold': 100,
    # Оптимизировать !
    'varianceRatioFilter': 1.0,
    'reverseVarianceRatioFilter': 1.0,
    # Оптимизировать !!
    'restAfterLoss': 200,
    # Оптимизировать !
    'restAfterFatProfit': 1,
    # Оптимизировать !!
    'stopLossStdMultiplier': 80_00,
    # Оптимизировать !
    'takeProfitStdMultiplier': 4050,
    # Оптимизировать !!
    'varianceRatioCarreteParameter': 20,
    # Оптимизировать !!!
    'scanHalfTime': 1500,
    # Оптимизировать 0!
    'halfToFat': 2,
    # Оптимизировать 0!
    'halfToLight': 1,
    # Оптимизировать 0!
    'halfToTime': 2,
    }

if platform.platform().split('-')[0] == 'macOS':
    systemDivide = '/'
else:
    systemDivide = '\\'


threading.stack_size(2**27)
sys.setrecursionlimit(10 ** 5)


testData = pd.read_csv(f"DataForJavaValidation{systemDivide}lightData", index_col=1)
# Какие колонки нужны для работы
columns = ['open', 'high', 'low', 'close']
testData = testData[columns]
testData.index = pd.to_datetime(testData.index)
params = create_strategy_config(firstConditions)

#   Testing Half Time module
half_time = int(get_half_time(testData.close[:params['scanHalfTime']]))
with open(f"DataForJavaValidation{systemDivide}testHalfTime.txt", 'w+') as f:
    f.write(str(half_time))

params["rollingMean"] = int(half_time * params['halfToLight'])
params["fatRollingMean"] = int(params['halfToFat'] * half_time)
params["timeBarrier"] = int(half_time * params['halfToTime'])
if params["timeBarrier"] <= 0:
    params["timeBarrier"] = 1
params["varianceLookBack"] = int(half_time * params['halfToFat'])
params["varianceRatioCarrete"] = int((half_time *
                                           params['halfToFat']) // params['varianceRatioCarreteParameter']) + 1


logTuple = testData.open.iloc[:params['varianceLookBack'] + 1]
retTuple = np.diff(logTuple)
logTuple = logTuple[1:]

#   Basic Variance Ratio Test
VRtest = variance_ratio(logTuple=tuple(logTuple), retTuple=retTuple, params=params)
with open(f"DataForJavaValidation{systemDivide}testBasicVarianceRatio.txt", 'w+') as f:
    f.write(str(VRtest))

_log = testData.open.iloc[:max(params['varianceLookBack'], params['fatRollingMean']) + 1]
compute = {
    "retOpenPrice": np.diff(_log),
    "logOpenPrice": _log[1:]
}

#   Reversed Variance Ratio Test
ReverseVRtest = reverse_variance_ratio(preComputed=compute, params=params,
                                       timeBorderCounter=20, VRstatement=False)
with open(f"DataForJavaValidation{systemDivide}testReversedVarianceRatio.txt", 'w+') as f:
    f.write(str(ReverseVRtest))
