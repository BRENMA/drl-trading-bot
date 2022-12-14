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
import time
from talib.abstract import MACD, RSI, CCI, DX
import talib as ta
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

ticker_list = ['ETHUSDT']
TIME_INTERVAL = '1m'
TRAIN_START_DATE = '2019-12-20'
#TRAIN_END_DATE= '2019-08-01'
#TRADE_START_DATE = '2019-08-01'
TRADE_END_DATE = '2020-01-03'

technical_indicator_list = ["rf","rsi","macd","macd_hist","cci","dx","sar","adx","adxr","apo","aroonosc","bop","mfi","minus_di","minus_dm","mom","plus_di","plus_dm","ppo_ta","roc","rocp","trix","ultosc","willr","ad","adosc","obv","ht_dcphase","ht_sine","ht_trendmode"]

if_vix = False
     
p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(ticker_list)
p.clean_data()
df = p.dataframe

#index = df.index
#df = df.insert(0, 'oldIndex', index.to_list())
df.reset_index(drop=True, inplace=True)

print(df.head())

#add FNG index
url = "https://api.alternative.me/fng/?limit=0"
response = requests.request("GET", url)
dataRes = response.json()
dfFnG = pd.json_normalize(dataRes['data'])
del dfFnG['value_classification']
del dfFnG['time_until_update']
dfFnG = dfFnG.iloc[::-1]

FnGArr = list(dfFnG.timestamp)
target = df.iloc[0:1440]["time"].to_list()
FnGStartPoint = 0

for n in range(len(target)):
    if FnGStartPoint == 0:
        for i in range(len(FnGArr)):
            if (int(FnGArr[i]) == time.mktime(datetime.strptime(target[n], '%Y-%m-%d %H:%M:%S').timetuple())):
                FnGStartPoint = i
    else:
        print("start point found")
        break

firstFnG = FnGStartPoint

DFStartIndex = df[df['time']== datetime.fromtimestamp(int(FnGArr[FnGStartPoint])).strftime('%Y-%m-%d %H:%M:%S')].index[0]
df = df.iloc[DFStartIndex:]

FnGIndArr = []
for i in range(len(df)):
    print(len(df) - i)

    dfUnixTime = int(time.mktime(datetime.strptime(df.iloc[i]['time'],"%Y-%m-%d %H:%M:%S").timetuple()))

    if dfUnixTime >= int(dfFnG.iloc[FnGStartPoint + 1]['timestamp']):
        FnGStartPoint += 1

    FnGIndArr.append(int(dfFnG.iloc[FnGStartPoint]['value']))

df.insert(0, "fngindex", FnGIndArr, True)
print(df)

dollar_bars = []
running_volume = 0
running_FnG = 0
running_open = 0
running_high, running_low = 0, math.inf
dollar_threshold = 5000000
period_lengths = [4, 8, 16]#, 32, 64, 128, 256]

for i in range(0, len(df)): 
    print(len(df) - i)
    
    next_timestamp, next_open, next_high, next_low, next_close, next_volume, next_FnG, next_tic = [df.iloc[i][k] for k in ['time', 'open', 'high', 'low', 'close', 'volume', 'fngindex', 'tic']]
    next_timestamp = pd.to_datetime(next_timestamp)

    # get the midpoint price of the next bar (the average of the open and the close)
    midpoint_price = (next_open + next_close)/2

    # get the approximate dollar volume of the bar using the volume and the midpoint price
    dollar_volume = next_volume * midpoint_price

    running_high, running_low = max(running_high, next_high), min(running_low, next_low)
 
    if running_open == 0:
        running_open = next_open

    if running_FnG == 0:
        running_FnG = next_FnG
    else:
        running_FnG = (running_FnG + next_FnG)/2

    # if the next bar's dollar volume would take us over the threshold...
    if dollar_volume + running_volume >= dollar_threshold:

        # set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e. one minute after the timestamp of the last minutely bar included in the dollar bar)
        bar_timestamp = next_timestamp + pd.to_timedelta(60, 's')

        # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
        dollar_bars += [{'tic': next_tic, 'timestamp': bar_timestamp, 'high': running_high, 'low': running_low, 'open': running_open, 'close': next_close, 'fng': running_FnG}]

        # reset the running volume to zero
        running_volume = 0
        running_FnG = 0
        running_open = 0
        running_high, running_low = 0, math.inf

    # otherwise, increment the running volume
    else:
        running_volume += dollar_volume

df = pd.DataFrame(dollar_bars)
def add_feature_columns(period_length):
    # get the price vs ewma feature
    df[f'feature_PvEWMA_{period_length}'] = df['close']/df['close'].ewm(span=period_length).mean() - 1
    # get the price vs cumulative high/low range feature
    df[f'feature_PvCHLR_{period_length}'] = (df['close'] - df['low'].rolling(period_length).min()) / (df['high'].rolling(period_length).max() - df['low'].rolling(period_length).min())
    # get the return vs rolling high/low range feature
    df[f'feature_RvRHLR_{period_length}'] = df['close'].pct_change(period_length)/((df['high']/df['low'] - 1).rolling(period_length).mean())
    # get the convexity/concavity feature
    df[f'feature_CON_{period_length}'] = (df['close'] + df['close'].shift(period_length))/(2 * df['close'].rolling(period_length+1).mean()) - 1
    # get the rolling autocorrelation feature
    df[f'feature_RACORR_{period_length}'] = df['close'].rolling(period_length).apply(lambda x: x.autocorr()).fillna(0)

# for each period length
for period_length in period_lengths:
    # add the feature columns to the bars df
    add_feature_columns(period_length)
    
# prune the nan rows at the beginning of the dataframe
df = df[period_lengths[-1]:]

# filter out the high/low/close columns and return 
#df = df[[column for column in df.columns if column not in ['high', 'low', 'close']]]

final_df = pd.DataFrame()
for i in df.tic.unique():
    #ORIGINAL INDICATORS
    tic_df = df[df.tic == i].copy()
    tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
    tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
    tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
 
    #OTHERS I ADDED
    #Overlap studies
    tic_df["rf"] = df["close"].pct_change().shift(-1)

    tic_df['sar'] = ta.SAR(df['high'], df['low'], acceleration=0., maximum=0.)

    # Added momentum indicators
    tic_df['adx'] = ta.ADX(df['high'], df['low'], df['close'])
    tic_df['adxr'] = ta.ADXR(df['high'], df['low'], df['close'])
    tic_df['apo'] = ta.APO(df['close'])
    tic_df['aroonosc'] = ta.AROONOSC(df['high'], df['low'])
    tic_df['bop'] = ta.BOP(df['open'], df['high'], df['low'], df['close'])
    tic_df['cmo'] = ta.CMO(df['close'])
    tic_df['minus_di'] = ta.MINUS_DI(df['high'], df['low'], df['close'])
    tic_df['minus_dm'] = ta.MINUS_DM(df['high'], df['low'])
    tic_df['mom'] = ta.MOM(df['close'])
    tic_df['plus_di'] = ta.PLUS_DI(df['high'], df['low'], df['close'])
    tic_df['plus_dm'] = ta.PLUS_DM(df['high'], df['low'])
    tic_df['ppo_ta'] = ta.PPO(df['close'])
    tic_df['roc'] = ta.ROC(df['close'])
    tic_df['rocp'] = ta.ROCP(df['close'])
    tic_df['rocr'] = ta.ROCR(df['close'])
    tic_df['rocr100'] = ta.ROCR100(df['close'])
    tic_df['trix'] = ta.TRIX(df['close'])
    tic_df['ultosc'] = ta.ULTOSC(df['high'], df['low'], df['close'])
    tic_df['willr'] = ta.WILLR(df['high'], df['low'], df['close'])

    # Cycle indicator functions
    tic_df['roc'] = ta.HT_DCPERIOD(df['close'])
    tic_df['ht_dcphase'] = ta.HT_DCPHASE(df['close'])
    tic_df['ht_sine'], _ = ta.HT_SINE(df['close'])
    tic_df['ht_trendmode'] = ta.HT_TRENDMODE(df['close'])

    final_df = final_df.append(tic_df)

df = final_df
df.dropna()

processed_df.index=pd.to_datetime(processed_df.time)
processed_df.drop('time', inplace=True, axis=1)
print(processed_df.tail(20))

print(df)
df.to_csv('test.csv')
print("> Storing Raw Historic Data")

#min_max_scaler = MinMaxScaler()
#for column in df:
#    columnSeriesObj = df[column]
#    if column != 'time':
#        df[column] = min_max_scaler.fit_transform(columnSeriesObj.values.reshape(-1, 1))
#
#testSize = 99.5 #%
#trainData = df.head(int(len(df)*(1 - testSize/100)))
#testData = df.tail(int(len(df)*(testSize/100)))
#
#print(trainData)
#
#output_x = []
#output_y = []
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