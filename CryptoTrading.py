from __future__ import annotations

import gym as gym
from config import *
from gym_examples.envs.trading_env import TradingEnv

from sklearn.preprocessing import MinMaxScaler
import math
import requests
import pandas as pd
from meta.data_processor import DataProcessor
from datetime import datetime
import time
from talib.abstract import MACD, RSI, CCI, DX
import talib as ta
from typing import Dict

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

print('\n'+'\n'+'\n'+"                                   )\._.,--....,'``.      " + '\n' + "                                  /;   _.. \   _\  (`._ ,." + '\n' + "                    $            `----(,_..'--(,_..'`-.;.'" + '\n' + '\n' + "                                                       /$$$$$$$  /$$$$$$$  /$$      " + "\n" +"                                                      | $$__  $$| $$__  $$| $$      " + "\n" +" /$$$$$$/$$$$   /$$$$$$  /$$$$$$$   /$$$$$$  /$$   /$$| $$  \ $$| $$  \ $$| $$      " + "\n" +"| $$_  $$_  $$ /$$__  $$| $$__  $$ /$$__  $$| $$  | $$| $$  | $$| $$$$$$$/| $$      " + "\n" +"| $$ \ $$ \ $$| $$  \ $$| $$  \ $$| $$$$$$$$| $$  | $$| $$  | $$| $$__  $$| $$      " + "\n" +"| $$ | $$ | $$| $$  | $$| $$  | $$| $$_____/| $$  | $$| $$  | $$| $$  \ $$| $$      " + "\n" +"| $$ | $$ | $$|  $$$$$$/| $$  | $$|  $$$$$$$|  $$$$$$$| $$$$$$$/| $$  | $$| $$$$$$$$" + "\n" +"|__/ |__/ |__/ \______/ |__/  |__/ \_______/ \____  $$|_______/ |__/  |__/|________/" + "\n" +"                                             /$$  | $$                              " + "\n" +"                                            |  $$$$$$/                              " + "\n" +"                                             \______/                               " + "\n" + "\n")
print("creating Testing Data")

p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(TICKER_LIST)
p.clean_data()
df = p.dataframe

def addFnG(df):
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
    dollar_threshold = 1000000

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

    return df

df = addFnG(df = df)
#df_TEST = addFnG(df = df_TEST)

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

def addIndicators(df):
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
        tic_df = tic_df.dropna()

        final_df = final_df.append(tic_df)

    df = final_df
    df.index=pd.to_datetime(df.time)
    df.drop('time', inplace=True, axis=1)
    df.drop('tic', inplace=True, axis=1)

    min_max_scaler = MinMaxScaler()

    #df[df.columns] = min_max_scaler.fit_transform(df[df.columns]) 

    for column in df.columns:
        if column != 'close':
            df[column] = min_max_scaler.fit_transform(df[[column]])

    df = df.dropna()

    return df

df = addIndicators(df = df)
#df_TEST = addIndicators(df = df_TEST)

env = gym.make('gym_examples/TradingEnv-v0', df = df, window_size = 5, frame_bound = (5, 50))
model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10_000, progress_bar=True)

model.save("dqn_crypto")
#del model  # delete trained model to demonstrate loading

#model = DQN.load("dqn_crypto", env=env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(5):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()







def buy(self):
    prev_bought_at = self.account.bought_btc_at # How much did I buy BTC for before
    if self.account.usd_balance - self.trade_amount >= 0:
        if prev_bought_at == 0 or self.account.last_transaction_was_sell or (prev_bought_at > self.account.btc_price): #or (self.account.btc_price/prev_bought_at -1 > 0.005):
            print(">> BUYING $",self.trade_amount," WORTH OF BITCOIN")
            self.account.btc_amount += self.trade_amount / self.account.btc_price
            self.account.usd_balance -= self.trade_amount
            self.account.bought_btc_at = self.account.btc_price
            self.account.last_transaction_was_sell = False
        else:
            print(">> Not worth buying more BTC at the moment")
    else:
        print(">> Not enough USD left in your account to buy BTC ")
def sell(self):
    if self.account.btc_balance - self.trade_amount >= 0:
        if self.account.btc_price > self.account.bought_btc_at: # Is it profitable?
            print(">> SELLING $",self.trade_amount," WORTH OF BITCOIN")
            self.account.btc_amount -= (self.trade_amount / self.account.btc_price)
            self.account.usd_balance += self.trade_amount
            self.account.last_transaction_was_sell = True
        else:
            print(">> Declining sale: Not profitable to sell BTC")
    else:
        print(">> Not enough BTC left in your account to buy USD ")
