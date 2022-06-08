import numpy as np
import matplotlib.pyplot as plt

initial_conditions = {
    "maturity": 1,
    "strike": 70,
    "upMove": 1.05,
    "firstSpot": 50,
    "riskFree": 0.01,
    "numberOfDotes": 20
}


def modeling_asset_price(initial_dict):
    timeStep = initial_dict["maturity"] / initial_dict['numberOfDotes']
    initial_dict['downMove'] = 1 / initial_dict["upMove"]
    initial_dict['prob'] = (np.exp(initial_dict["riskFree"] * timeStep) - initial_dict["downMove"]) / (
            initial_dict["upMove"] - initial_dict["downMove"])
    assetPriceArray = np.zeros(shape=(initial_dict["numberOfDotes"], initial_dict["numberOfDotes"]))
    for row in range(0, initial_dict["numberOfDotes"]):
        for line in range(0, row+1):
            assetPriceArray[line, row] = initial_dict["firstSpot"] * \
                                         (initial_dict["upMove"] ** line) * (initial_dict["downMove"] ** (row - line))

    valueOfOption = np.zeros(shape=(initial_dict["numberOfDotes"], initial_dict["numberOfDotes"]))
    PayOff = [max(_assetPrice - initial_conditions["strike"], 0)
              for _assetPrice in assetPriceArray[:, -1]]
    valueOfOption[:, -1] = PayOff

    # print(np.exp(-initial_dict["riskFree"] * timeStep) * (initial_dict["prob"] * 12 + (1 - initial_dict["prob"]) * 20))

    for i in range(2, initial_dict["numberOfDotes"]+1):

        valueOfOption[:, -i] = [np.exp(-initial_dict["riskFree"] * timeStep) * (
            initial_dict["prob"] * valueOfOption[j, -i + 1] + (1 - initial_dict["prob"]) * valueOfOption[j+1, -i + 1]
        )
                                for j in range(valueOfOption.shape[0] - i + 1)] + (i - 1) * [0]

    return assetPriceArray, valueOfOption


assetPrice, optionValue = modeling_asset_price(initial_dict=initial_conditions)


plt.style.use('Solarize_Light2')
plt.figure(figsize=(12, 6))

plt.title(f'Asset Price Tree; V0={optionValue[0, 0]}')
for n in range(initial_conditions["numberOfDotes"]):
    plt.scatter(n * np.ones(n+1), assetPrice[:n+1, n], color='red')
    plt.axhline(optionValue[n, -1], color='green')

plt.legend(loc='lower right')
plt.show()
