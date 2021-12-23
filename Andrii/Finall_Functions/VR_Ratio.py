import pandas as pd
import numpy as np
import tqdm


def variance_ratio(q_position, df: pd.DataFrame):
    """

    :param q_position: Usually equal to 10. That means that q = [1 .. 1024]
    :param df: df with close prices in lower register
    :return: array:VR_RATIO, array:created time_lags
    """
    MR = list()

    result_array = list()
    q_array = list()
    #   Create time_lags like 2 ** k
    #time_lags = [1] + [2 ** i for i in range(1, q_position+1)]
    log_prices = np.log(df.close.dropna())
    ret = np.log(df.close.dropna())
    #ret = ret.pct_change()[1:].dropna()
    #ret = ret.diff().dropna()
    ret = ret.diff().dropna()
    time_lags = np.arange(1, 2 ** q_position, 50)
    for EA, time_lag in tqdm(enumerate(time_lags), total=len(time_lags), leave=False):
        buff_size = len(ret)
        means = (1 / buff_size) * np.sum(ret)
        m = time_lag * (buff_size - time_lag + 1) * (1 - (time_lag / buff_size))
        sigma_a = (1 / (buff_size - 1)) * np.sum(np.square(ret - means))
        subtract_returns = np.subtract(log_prices, np.roll(log_prices, time_lag))[time_lag:]
        _buff_ = np.sum(np.square(subtract_returns - time_lag * means))
        sigma_b = (1 / m) * _buff_
        result = (sigma_b / sigma_a)
        #plt.plot(time_lag, result, 'o', color='white')
        #plt.grid(alpha=.1)
        result_array.append(result)
        q_array.append(time_lag)
    return result_array, q_array