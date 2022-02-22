from Andrii.Strategy_Constructor.strategyHubs import HubTrend
from Andrii.Strategy_Constructor.extractors import TrendFollowingExtractor
from Andrii.Strategy_Constructor.filters.TrendFollowingFilters import *
from Andrii.Strategy_Constructor.rules_constructors.TrendFollowingRules import *
from Andrii.Strategy_Constructor.openingRulesComplex import CrossHighLow
from Andrii.Strategy_Constructor.statistics.available_stat import *
from Andrii.Strategy_Constructor.holdRules.StopLoss import *

import pandas as pd
import datetime
import matplotlib.pylab as plt

# Define extractor
horizon = TrendFollowingExtractor.StrategyExtractor()

constime = ConsistingTime(pd.Timedelta(days=0, hours=22), pd.Timedelta(days=1, hours=8), pd.Timedelta(days=1, hours=22))
horizon.apply_filters(constime)

nodolidays = NoHolidays()
horizon.apply_filters(nodolidays)

leftBorder = LeftBorder(pd.Timedelta(hours=22))
horizon.apply_filters(leftBorder)


horizon.input_params(start_horizon=datetime.timedelta(hours=-12), end_horizon=datetime.timedelta(hours=12))
# Define StrategyHub

HubStrategy = HubTrend.HubTrendFollowingStrategy()
HubStrategy.add_extractor(horizon)

# Define StrategyRuleConstructor
StrategyTFConstructor = TrendFollowingStrategyRulesConstructor()
StrategyTFConstructor.add_median_dot(pd.Timedelta(hours=10))

# Define FirstOpeningRule
CrossingLayersRule = CrossHighLow.OpeningCrossHighLowRule()

# Define Low Parameter
low = CalculateLow()

# Define High Parameter
high = CalculateHigh()

# Define HoldStopLossRule

stop_loss = HoldStopLossNoSlippage()
stop_loss_threshold = {"BUY": 0.001, "SELL": 0.001}
stop_loss.make_threshold(stop_loss_threshold)

# Adding Workers to FirstOpeningRule
CrossingLayersRule.add_low_calculation(low=low)
CrossingLayersRule.add_high_calculation(high=high)

# Adding FirstOpeningRule to StrategyConstructor
StrategyTFConstructor.add_opening_rule(CrossingLayersRule)

# Adding StopLossRule to StrategyConstructor
StrategyTFConstructor.add_hold_rule(stop_loss)

#   Add StrategyRuleConstructor to StrategyHub
HubStrategy.add_Rules_constructor(StrategyTFConstructor)
#   Testing

# First need to export data to Hub
# d = pd.read_csv('zip_data/AUDCAD_base.csv', index_col=0)
d = pd.read_csv('test_data/EURGBP.csv', index_col=0)
d.index = pd.to_datetime(d.index)
d = d.resample('1H').first()
d = d
d['line_number'] = range(1, d.shape[0]+1)

HubStrategy.transfer_init_data(d)
HubStrategy.testing_simulation()



def testing_calculate(df_results):
    POSITION_MONEY = 100_000
    SLIPADGE = 10
    results = list()
    for i in range(df_results.shape[0]):
        position = POSITION_MONEY / df_results.iloc[i].OpenPrice
        if df_results.iloc[i].OperationType == 'BUY':
            profit = position * (df_results.iloc[i].ClosePrice - df_results.iloc[i].OpenPrice) - SLIPADGE

        if df_results.iloc[i].OperationType == 'SELL':
            profit = POSITION_MONEY - position * df_results.iloc[i].ClosePrice - SLIPADGE

        owning_time = df_results.iloc[i].CloseTime - df_results.iloc[i].OpenTime
        results.append({'start_position': df_results.iloc[i].OpenTime, 'profit': profit, 'owning_time': owning_time})

    return pd.DataFrame(results)


testing_results = testing_calculate(pd.DataFrame(HubStrategy.simulation_results))
testing_results["total_pnl"] = testing_results.profit.cumsum()
testing_results.index = testing_results.start_position


plt.figure(figsize=(12, 5))
plt.plot(testing_results.total_pnl)
plt.show()