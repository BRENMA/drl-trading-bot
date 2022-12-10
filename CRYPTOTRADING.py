import math
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime, timedelta
from pandas.testing import assert_frame_equal

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

window_size = 60

print("\n\n\n'88,dPYba,,adPYba,   ,adPPYba,  8b,dPPYba,   ,adPPYba, 8b       d8  \n","88P'   '88'    '8a a8'     '8a 88P'   `'8a a8P_____88 `8b     d8'  \n","88      88      88 8b       d8 88       88 8PP'''''''  `8b   d8'   \n","88      88      88 '8a,   ,a8' 88       88 '8b,   ,aa   `8b,d8'    \n","88      88      88  `'YbbdP''  88       88  `'Ybbd8''     Y88'     \n","                                                          d8'      \n","                                                         d8'       \n")
print("> welcome to moneyDRL")
print("> Creating Testing Data")
# merging two csv files
df = pd.concat(map(pd.read_csv, ['datasets/ADA-USD.csv']), ignore_index=True)#, 'datasets/ATOM-USD.csv', 'datasets/AVAX-USD.csv', 'datasets/BTC-USD.csv', 'datasets/ETH-USD.csv', 'datasets/LINK-USD.csv', 'datasets/MATIC-USD.csv', 'datasets/SOL-USD.csv', 'datasets/XRP-USD.csv']), ignore_index=True)
df = df.dropna(axis='columns')
print(len(df))
print("> Creating X,Y sets")

min_max_scaler = MinMaxScaler()
for column in df:
    columnSeriesObj = df[column]
    if column != 'time':
        df[column] = min_max_scaler.fit_transform(columnSeriesObj.values.reshape(-1, 1))

testSize = 99.5 #%
trainData = df.head(int(len(df)*(1 - testSize/100)))
testData = df.tail(int(len(df)*(testSize/100)))

print(trainData)

output_x = []
output_y = []
for i in range(0, len(df) - (window_size + 1)):
    print(len(df) - window_size - i)
    hour_data = df.iloc[i: (i + (window_size + 1)), :]
    if hour_data['time'].iloc[-1] - hour_data['time'].iloc[0] == 3600:
        del hour_data['time']
        t = []
        output_y.append(hour_data.iloc[-1].values.tolist())
        for j in range(0, window_size):
            hrData = hour_data.iloc[j].values.tolist()
            t.append(hrData)
        output_x.append(t)
    
    output_x = np.array(output_x)
    output_x = output_x.reshape(output_x.shape[0], window_size, data_columns)
