import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Week_1_data.csv', index_col='Index')

w0 = 0
w1 = 0
w = [w0, w1]
n = len(data)
print(data)

for lam in range(1, len(data)):
    print(data['Height'][lam] * data['Weight'][lam])