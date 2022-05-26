from Andrii.Strategy_Constructor.statistics.available_stat import CreateBBand
from Andrii.Strategy_Constructor.extractors import TrendFollowingExtractor
from Andrii.Strategy_Constructor.strategyHubs import HubTrend
from Andrii.Strategy_Constructor.rules_constructors.TrendFollowingRules import TrendFollowingStrategyRulesConstructor
from Andrii.Strategy_Constructor.openingRulesComplex.BbandStrategy import BBandsRule
from Andrii.Strategy_Constructor.holdRules.StopLoss import HoldStopLossNoSlippage
from Andrii.Strategy_Constructor.filters.TrendFollowingFilters import ConsistingTime

import pandas as pd
import matplotlib.pylab as plt
import datetime


STRATEGY_PARAM = {'moving_window': '8h',
                   'moving_lookback': '-8h',
                   'STD_PERCENT': 300,
                   'max_holding_period': '8h'}



df = pd.read_csv('test_data/AUDCAD.csv', index_col=1, header=0)
df.index = pd.to_datetime(df.index)
df = df.resample('1T').first()
df = df.iloc[10_00:]

df['line_number'] = range(1, df.shape[0] + 1)

def STRATEGY_STEP(d, STRATEGY_PARAMS):
    d = d



    #   Have Params
    HBand = CreateBBand(window_moving_average=STRATEGY_PARAMS['moving_window'], Y_threshold=STRATEGY_PARAMS['STD_PERCENT'] / 100)

    HBand.add_calculation_data(d.iloc[:10_000].close)
    plt.figure(figsize=(12, 6))
    plt.title(f"STD={STRATEGY_PARAMS['STD_PERCENT']} MOVING={STRATEGY_PARAMS['moving_window']}")
    plt.plot(d.iloc[:5_00].close, label='inp_data')
    HBand.create_parameter()
    BBands = HBand.param
    plt.plot(BBands[0], label='LowLevel')
    plt.plot(BBands[1], label='HighLevel')
    plt.legend(loc='lower right')
    plt.show()

    del HBand




    horizon = TrendFollowingExtractor.StrategyExtractor()
    # horizon.input_params(start_horizon=pd.Timedelta(STRATEGY_PARAMS['moving_lookback']), end_horizon=':')
    horizon.input_params(start_horizon=pd.Timedelta(STRATEGY_PARAMS['moving_lookback']),
                         end_horizon=pd.Timedelta(STRATEGY_PARAMS['max_holding_period']))

    constime = ConsistingTime(pd.Timedelta(STRATEGY_PARAMS['moving_window']))
    horizon.apply_filters(constime)

    HubStrategy = HubTrend.HubTrendFollowingStrategy()
    HubStrategy.add_extractor(horizon)

    StrategyTFConstructor = TrendFollowingStrategyRulesConstructor()
    StrategyTFConstructor.add_median_dot(pd.Timedelta(STRATEGY_PARAMS['moving_window']))
    BandsRule = BBandsRule()

    bolindger = CreateBBand(window_moving_average=STRATEGY_PARAMS['moving_window'],
                            Y_threshold=(STRATEGY_PARAMS['STD_PERCENT'] / 100))


    BandsRule.add_BBand(bolindger)
    stop_loss = HoldStopLossNoSlippage()
    stop_loss_threshold = {"BUY": 0.015, "SELL": 0.015}

    stop_loss.make_threshold(stop_loss_threshold)
    StrategyTFConstructor.add_opening_rule(BandsRule)
    StrategyTFConstructor.add_hold_rule(stop_loss)
    HubStrategy.add_Rules_constructor(StrategyTFConstructor)

    HubStrategy.transfer_init_data(d)
    HubStrategy.testing_simulation()

    def testing_calculate(df_results):
        POSITION_MONEY = 20_000
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
    plt.title(f"STD={STRATEGY_PARAMS['STD_PERCENT']} MOVING={STRATEGY_PARAMS['moving_window']}")
    plt.plot(testing_results.total_pnl)
    plt.show()


for _ in df.index.year.unique():
    buffer = df.loc[pd.Timestamp(year=_, month=1, day=1): pd.Timestamp(year=_+1, month=1, day=1)]
    STRATEGY_STEP(d=buffer, STRATEGY_PARAMS=STRATEGY_PARAM)
