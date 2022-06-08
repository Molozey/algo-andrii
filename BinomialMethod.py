#! Binomical method skeleton realization for american/europenial option. With calculating risk-neutral parameters

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def risk_neutral_probability(risk_free, volatility, time_step):
    _betta = (1/2) * (np.exp(-1 * risk_free * time_step) + np.exp((risk_free + np.square(volatility)) * time_step))
    upperMultiply = _betta + np.sqrt(np.square(_betta) - 1)
    lowerMultiply = 1 / upperMultiply
    probability = (np.exp(risk_free * time_step) - lowerMultiply) / (upperMultiply - lowerMultiply)
    return {'u': upperMultiply, 'd': lowerMultiply, 'p': probability}


def modeling_asset_price(initial_dict):
    timeStep = initial_dict["expirationTime"] / initial_dict['numberOfDotes']
    risk_neutral = risk_neutral_probability(risk_free=initial_dict['riskFreeRate'],
                                            volatility=initial_dict['volatility'],
                                            time_step=timeStep)
    if initial_dict["type"] == "American":
        resultMask = np.zeros(shape=(initial_dict["numberOfDotes"], initial_dict["numberOfDotes"]))
        print('Created Mask:\n', resultMask.shape)
        for row in range(resultMask.shape[1]):
            for line in range(row):
                resultMask[line, row] = initial_dict["initialAssetPrice"] * (risk_neutral['u'] ** line) * \
                                        (risk_neutral['d'] ** (row - line))

        print('Result Mask:\n', resultMask.astype(int))
        PayOffArrayInit = np.array([max(result - initial_dict["strike"], 0) for result in resultMask[:, -1]])
        PayOffArray = np.zeros(shape=resultMask.shape)
        PayOffArray[:, -1] = PayOffArrayInit

        for line in range(resultMask.shape[0] - 2, 0, -1):
            for row in range(resultMask.shape[1] - 2, 0, -1):
                PayOffArray[line, row] = np.exp(-initial_dict["riskFreeRate"] * timeStep) * (
                    risk_neutral['p'] * PayOffArray[line+1, row+1] + (1 - risk_neutral['p']) * PayOffArray[line, row+1]
                )

        cleanValue = np.zeros(shape=resultMask.shape)
        for line in range(resultMask.shape[0]):
            for row in range(resultMask.shape[1]):
                cleanValue[line, row] = max(resultMask[line, row] - initial_dict["strike"], PayOffArray[line, row])
        print('\n\n\n')
        print(cleanValue.astype(int))
        return resultMask, PayOffArray, cleanValue

    if initial_dict["type"] == 'Europenian':
        resultMask = np.zeros(shape=(1, initial_dict["numberOfDotes"]))
        print('Created Mask:\n', resultMask.shape)
        print(resultMask.shape)
        for row in range(resultMask.shape[1]):
            resultMask[0, row] = initial_dict["initialAssetPrice"] * (risk_neutral['u'] ** row) * \
                                    (risk_neutral['d'] ** (resultMask.shape[1] - row))

        print('Result Mask:\n', resultMask)


initial_dict = {
        "riskFreeRate": 0.2,
        "volatility": 0.4,
        "initialAssetPrice": 100,
        "expirationTime": 1,
        "strike": 120,
        "type": "American",
        "numberOfDotes": 20,
    }

s, p, CV = modeling_asset_price(initial_dict)

plt.style.use('Solarize_Light2')
plt.figure(figsize=(10,10))
for n in range(initial_dict["numberOfDotes"]):
    plt.scatter(n * np.ones(n), s[:n, n], color = 'red')
plt.axhline(initial_dict["strike"])
plt.show()