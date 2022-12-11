import scipy as sp
import math
import pandas as pd
import requests
import json
import matplotlib.dates as mdates
import numpy as np
import pickle
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meta.data_processor import DataProcessor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns

from datetime import datetime, timedelta
from talib.abstract import MACD, RSI, CCI, DX
from binance.client import Client
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from IPython.display import display, HTML

print("\n\n\n'88,dPYba,,adPYba,   ,adPPYba,  8b,dPPYba,   ,adPPYba, 8b       d8  \n","88P'   '88'    '8a a8'     '8a 88P'   `'8a a8P_____88 `8b     d8'  \n","88      88      88 8b       d8 88       88 8PP'''''''  `8b   d8'   \n","88      88      88 '8a,   ,a8' 88       88 '8b,   ,aa   `8b,d8'    \n","88      88      88  `'YbbdP''  88       88  `'Ybbd8''     Y88'     \n","                                                          d8'      \n","                                                         d8'       \n")
print("> welcome to moneyDRL")
print("> Creating Testing Data")

ticker_list = ['BTCUSDT']

TIME_INTERVAL = '1m'

TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE= '2019-08-01'
TRADE_START_DATE = '2019-08-01'
TRADE_END_DATE = '2020-01-03'

technical_indicator_list = ["rf","rsi","macd","macd_hist","cci","dx","sar","adx","adxr","apo","aroonosc","bop","mfi","minus_di","minus_dm","mom","plus_di","plus_dm","ppo_ta","roc","rocp","trix","ultosc","willr","ad","adosc","obv","ht_dcphase","ht_sine","ht_trendmode"]

if_vix = False
     
p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(ticker_list)
p.clean_data()
df = p.dataframe

print(df.head())

dollar_bars = []
running_volume = 0
running_high, running_low = 0, math.inf
dollar_threshold = 50000000
#[time, low, high, open, close, volume, rf, rsi, macd, cci, dx, sar, adx, adxr, apo, aroonosc, bop, cmo, mfi, minus_di, minus_dm, mom, plus_di, plus_dm, ppo_ta, roc, rocp, rocr, rocr100, trix, ultosc, willr, ad, adosc, obv, roc, ht_dcphase, ht_sine, ht_trendmode]

for i in range(0, len(df)): 
    print(len(df) - i)
    
    next_timestamp, next_open, next_high, next_low, next_close, next_volume = [df.iloc[i][k] for k in ['time', 'open', 'high', 'low', 'close', 'volume']]
    #print(next_timestamp, next_open, next_high, next_low, next_close, next_volume)

    next_timestamp = pd.to_datetime(next_timestamp)

    # get the midpoint price of the next bar (the average of the open and the close)
    midpoint_price = (next_open + next_close)/2

    # get the approximate dollar volume of the bar using the volume and the midpoint price
    dollar_volume = next_volume * midpoint_price

    running_high, running_low = max(running_high, next_high), min(running_low, next_low)

    # if the next bar's dollar volume would take us over the threshold...
    if dollar_volume + running_volume >= dollar_threshold:

        # set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e. one minute after the timestamp of the last minutely bar included in the dollar bar)
        bar_timestamp = next_timestamp + pd.to_timedelta(60, 's')

        # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
        dollar_bars += [{'timestamp': bar_timestamp, 'high': running_high, 'low': running_low, 'close': next_close}]

        # reset the running volume to zero
        running_volume = 0

        # reset the running high and low to placeholder values
        running_high, running_low = 0, math.inf

    # otherwise, increment the running volume
    else:
        running_volume += dollar_volume

df = pd.DataFrame(dollar_bars)

def add_technical_indicator(df, tech_indicator_list):
    # print('Adding self-defined technical indicators is NOT supported yet.')
    # print('Use default: MACD, RSI, CCI, DX.')

    final_df = pd.DataFrame()
    for i in df.tic.unique():
        tic_df = df[df.tic == i].copy()
        tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
        tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12,
                                                                          slowperiod=26, signalperiod=9)
        tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
        final_df = final_df.append(tic_df)
    return final_df

processed_df=add_technical_indicator(df,technical_indicator_list)
processed_df.tail()













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
#for i in range(0, len(df) - (window_size + 1)):
#    print(len(df) - window_size - i)
#    hour_data = df.iloc[i: (i + (window_size + 1)), :]
#    if hour_data['time'].iloc[-1] - hour_data['time'].iloc[0] == 3600:
#        del hour_data['time']
#        t = []
#        output_y.append(hour_data.iloc[-1].values.tolist())
#        for j in range(0, window_size):
#            hrData = hour_data.iloc[j].values.tolist()
#            t.append(hrData)
#        output_x.append(t)
#    
#    output_x = np.array(output_x)
#    output_x = output_x.reshape(output_x.shape[0], window_size, data_columns)
#