import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sc
data = pd.read_csv('Week_1_data.csv', index_col='Index')
data.plot(y='Weight', kind='hist', color='green', title='Частота появления конкретного веса')
plt.show()
sns.pairplot(data)
plt.show()
def BMI_MASS(height_inch, weight_pound):
    PerCofs = (39.37, 2.20462)
    return (weight_pound / PerCofs[0]) / ((height_inch / PerCofs[1]) ** 2)


data['BMI_MASS'] = data.apply(lambda row: BMI_MASS(row['Height'], row['Weight']), axis=1)

sns.pairplot(data=data)
plt.show()


def MASS_SORT(weight_pound):
    if weight_pound < 120:
        logic = 1
    if weight_pound >=120 and weight_pound<150:
        logic = 2
    if weight_pound >=150:
        logic = 3
    return logic

data['Weight_Category'] = data.apply(lambda row: MASS_SORT(row['Weight']), axis=1)
sns.boxplot(y='Height', x='Weight_Category', data=data, palette='rainbow',)
plt.xlabel('Весовая категория')
plt.ylabel('Рост')
plt.show()


data.plot.scatter(x='Weight', y='Height')
plt.show()


w0 = 0
w1 = 0
mass_coffs = [w0, w1]


def error_function(w):
    summator = 0
    for i in range(1, len(data)):
        summator += (data['Height'][i] - (mass_coffs[0] + mass_coffs[1] * data['Weight'][i])) ** 2
    return summator


err = error_function(mass_coffs)

def error_function(w1,w2):
    summator = 0
    for i in range(1, len(data)):
        summator += (data['Height'][i] - (w1 + w2 * data['Weight'][i])) ** 2
    return summator

def error_mass(w):
    w1=w[0]
    w2=w[1]
    return error_function(w1,w2)


def error_function(w1,w2):
    summator = 0
    for i in range(1, len(data)):
        summator += (data['Height'][i] - (w1 + w2 * data['Weight'][i])) ** 2
    return summator
w=[0,0]
bonds = ((-100,100),(-5,5))
res = sc.minimize(error_mass,[0.0,0.0],method='L-BFGS-B',bounds=((-100,100),(-5,5)))
print(res)