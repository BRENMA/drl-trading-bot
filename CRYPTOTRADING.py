from __future__ import annotations

import math
import requests
import pandas as pd
import numpy as np

import gym
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
from sklearn.preprocessing import MinMaxScaler 

from meta.data_processor import DataProcessor
from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv

from agents.elegantrl_models import DRLAgent as DRLAgent_erl

from datetime import datetime
import time
import os

from talib.abstract import MACD, RSI, CCI, DX
import talib as ta

#from imblearn.over_sampling import SMOTE

print("\n\n\n'88,dPYba,,adPYba,   ,adPPYba,  8b,dPPYba,   ,adPPYba, 8b       d8  \n","88P'   '88'    '8a a8'     '8a 88P'   `'8a a8P_____88 `8b     d8'  \n","88      88      88 8b       d8 88       88 8PP'''''''  `8b   d8'   \n","88      88      88 '8a,   ,a8' 88       88 '8b,   ,aa   `8b,d8'    \n","88      88      88  `'YbbdP''  88       88  `'Ybbd8''     Y88'     \n","                                                          d8'      \n","                                                         d8'       \n")
print("> welcome to moneyDRL")
print("> Creating Testing Data")

TICKER_LIST = ["BTCUSDT"]#, "ETHUSDT", "ADAUSDT", "ATOMUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT"]

TIME_INTERVAL = '1m'
TRAIN_START_DATE = '2019-11-01'
#TRAIN_END_DATE= '2019-08-01'
#TRADE_START_DATE = '2019-08-01'
TRADE_END_DATE = '2020-01-03'

p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(TICKER_LIST)
p.clean_data()
df = p.dataframe

#df = df[df['tic'] == 'BTCUSDT'].copy()
#df_ETHUSDT = df[df['tic'] == 'ETHUSDT'].copy()
#df_ADAUSDT = df[df['tic'] == 'ADAUSDT'].copy()
#df_ATOMUSDT = df[df['tic'] == 'ATOMUSDT'].copy()
#df_SOLUSDT = df[df['tic'] == 'SOLUSDT'].copy()
#df_DOTUSDT = df[df['tic'] == 'DOTUSDT'].copy()
#df_DOGEUSDT = df[df['tic'] == 'DOGEUSDT'].copy()
#df_AVAXUSDT = df[df['tic'] == 'AVAXUSDT'].copy()

#df_AFTER_BTC_LIST = [df_ETHUSDT]#, df_ADAUSDT, df_ATOMUSDT, df_SOLUSDT, df_DOTUSDT, df_DOGEUSDT, df_AVAXUSDT]

#unique_ticker = df.tic.unique()
#print(unique_ticker)

#price_array = np.column_stack([df[df.tic == tic].close for tic in unique_ticker])
#print(price_array)

#BASING EVERYTHING OFF THE BTC DATAFRAME
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

print("getting dollar bars")
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
        dollar_bars += [{'tic': next_tic, 'time': bar_timestamp, 'high': running_high, 'low': running_low, 'open': running_open, 'close': next_close, 'fng': running_FnG}]

        # reset the running volume to zero
        running_volume = 0
        running_FnG = 0
        running_open = 0
        running_high, running_low = 0, math.inf

    # otherwise, increment the running volume
    else:
        running_volume += dollar_volume

df = pd.DataFrame(dollar_bars)
# prune the nan rows at the beginning of the dataframe
df = df[period_lengths[-1]:]

#ADDING THE REST OF THE TOKENS
#for df_temp in df_AFTER_BTC_LIST:
#    print(df_temp)
#    print(df)
#    mergedDf = df_temp[df_temp['time'].isin(df['time'])]
#    print(mergedDf)
#    aaaaaaaaaaaaaaaaaa

def add_feature_columns(df, period_length):
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

    # for each period length
    for period_length in period_lengths:
        # add the feature columns to the bars df
        add_feature_columns(tic_df, period_length)

    # prune the nan rows at the beginning of the dataframe
    df = df[period_lengths[-1]:]

    final_df = final_df.append(tic_df)

df = final_df

df.index=pd.to_datetime(df.time)
df.drop('time', inplace=True, axis=1)
df.dropna(inplace=True)

min_max_scaler = MinMaxScaler()
for column in df:
    columnSeriesObj = df[column]
    if column != 'time' and column != 'tic':
        df[column] = min_max_scaler.fit_transform(columnSeriesObj.values.reshape(-1, 1))

unique_ticker = df.tic.unique()
print(unique_ticker)

price_array = np.column_stack([df[df.tic == tic].close for tic in unique_ticker])
print(price_array)

column_titles = list(df.columns.values)
column_titles.remove('tic')
column_titles.remove('close')
common_tech_indicator_list = [i for i in column_titles if i in df.columns.values.tolist()]
tech_array = np.hstack([df.loc[(df.tic == tic), common_tech_indicator_list] for tic in unique_ticker])
print(tech_array)
turbulence_array = np.array([])
print("Successfully transformed into array")

data_config = {'price_array': price_array, 'tech_array': tech_array}
ERL_PARAMS = {"learning_rate": 2**-15,"batch_size": 2**11, "gamma": 0.99, "seed":312,"net_dimension": 2**9,  "target_step": 5000, "eval_gap": 30, "eval_times": 1}

initial_capital = 1000
env = CryptoEnv

# TRAINING
start_time = time.time()

model_name='sac'
current_working_dir="./test_sac"
erl_params=ERL_PARAMS
break_step=5e4
if_vix=False

env_instance = env(config=data_config)

duration_train = round((time.time() - start_time), 2)

agent = DRLAgent_erl(env = env, price_array = price_array, tech_array=tech_array, turbulence_array=turbulence_array)
model = agent.get_model(model_name, model_kwargs = erl_params)
trained_model = agent.train_model(model=model,  cwd=current_working_dir, total_timesteps=break_step)

print(trained_model)



#train(
#    start_date=TRAIN_START_DATE, 
#    end_date=TRAIN_END_DATE,
#    ticker_list=TICKER_LIST, 
#    data_source='binance',
#    time_interval='5m', 
#    technical_indicator_list=INDICATORS,
#    drl_lib='elegantrl', 
#    env=env, 
#    model_name='ppo', 
#    current_working_dir='./test_ppo',
#    erl_params=ERL_PARAMS,
#    break_step=5e4,
#    if_vix=False
#)











# read parameters

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
