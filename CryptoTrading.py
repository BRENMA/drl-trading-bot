from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt

from gym_examples.envs.trading_env import TradingEnv

from sklearn.preprocessing import MinMaxScaler

import math
import requests
import numbers
import pandas as pd
import numpy as np
import numpy.random as rd

from meta.data_processor import DataProcessor

from datetime import datetime
import time

from talib.abstract import MACD, RSI, CCI, DX
import talib as ta


print('\n'+'\n'+'\n'+"                                   )\._.,--....,'``.      " + '\n' + "                                  /;   _.. \   _\  (`._ ,." + '\n' + "                    $            `----(,_..'--(,_..'`-.;.'" + '\n' + '\n' + "                                                       /$$$$$$$  /$$$$$$$  /$$      " + "\n" +"                                                      | $$__  $$| $$__  $$| $$      " + "\n" +" /$$$$$$/$$$$   /$$$$$$  /$$$$$$$   /$$$$$$  /$$   /$$| $$  \ $$| $$  \ $$| $$      " + "\n" +"| $$_  $$_  $$ /$$__  $$| $$__  $$ /$$__  $$| $$  | $$| $$  | $$| $$$$$$$/| $$      " + "\n" +"| $$ \ $$ \ $$| $$  \ $$| $$  \ $$| $$$$$$$$| $$  | $$| $$  | $$| $$__  $$| $$      " + "\n" +"| $$ | $$ | $$| $$  | $$| $$  | $$| $$_____/| $$  | $$| $$  | $$| $$  \ $$| $$      " + "\n" +"| $$ | $$ | $$|  $$$$$$/| $$  | $$|  $$$$$$$|  $$$$$$$| $$$$$$$/| $$  | $$| $$$$$$$$" + "\n" +"|__/ |__/ |__/ \______/ |__/  |__/ \_______/ \____  $$|_______/ |__/  |__/|________/" + "\n" +"                                             /$$  | $$                              " + "\n" +"                                            |  $$$$$$/                              " + "\n" +"                                             \______/                               " + "\n" + "\n")
print("creating Testing Data")

TICKER_LIST = ["ETHUSDT"]#, "ATOMUSDT", "ADAUSDT", "BTCUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT"]
INDICATORS = ['high','low','open','close','fng','rsi','macd','macd_signal','macd_hist','cci','dx','rf','sar','adx','adxr','apo','aroonosc','bop','cmo','minus_di','minus_dm','mom','plus_di','plus_dm','ppo_ta','roc','rocp','rocr','rocr100','trix','ultosc','willr','ht_dcphase','ht_sine','ht_trendmode','feature_PvEWMA_4','feature_PvCHLR_4','feature_RvRHLR_4','feature_CON_4','feature_RACORR_4','feature_PvEWMA_8','feature_PvCHLR_8','feature_RvRHLR_8','feature_CON_8','feature_RACORR_8','feature_PvEWMA_16','feature_PvCHLR_16','feature_RvRHLR_16','feature_CON_16','feature_RACORR_16']
period_lengths = [4, 8, 16, 32, 64, 128, 256]

TIME_INTERVAL = '1m'
TRAIN_START_DATE = '2020-07-01'
TRAIN_END_DATE= '2020-08-01'

TEST_START_DATE = '2021-07-01'
TEST_END_DATE = '2021-08-01'
# To make the Agent more risk averse towards negative returns. Negative reward multiplier
NEG_MUL = 0

p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(TICKER_LIST)
p.clean_data()
df = p.dataframe

#t = DataProcessor(data_source='binance', start_date=TEST_START_DATE, end_date=TEST_END_DATE, time_interval=TIME_INTERVAL)
#t.download_data(TICKER_LIST)
#t.clean_data()
#df_TEST = t.dataframe

print(len(df))
#print(len(df_TEST))

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
    df[df.columns] = min_max_scaler.fit_transform(df[df.columns])

    df = df.dropna()

    return df

df = addIndicators(df = df)
#df_TEST = addIndicators(df = df_TEST)

#def splitIntoArr(df): 
#    unique_ticker = df.tic.unique()
#    price_array = np.column_stack([df[df.tic == tic].close for tic in unique_ticker])
#
#    #column_titles = list(df.columns.values)
#    #column_titles.remove('tic')
#    #column_titles.remove('close')
#    #common_tech_indicator_list = [i for i in column_titles if i in df.columns.values.tolist()]
#    tech_array = np.hstack([df.loc[(df.tic == tic), INDICATORS] for tic in unique_ticker])
#
#    turbulence_array = np.array([])
#
#    min_max_scaler = MinMaxScaler()
#    price_array = min_max_scaler.fit_transform(price_array)
#    tech_array = min_max_scaler.fit_transform(tech_array)
#
#    return price_array, tech_array, turbulence_array

#price_array, tech_array, turbulence_array = splitIntoArr(df = df)
#price_array_TEST, tech_array_TEST, turbulence_array_TEST = splitIntoArr(df = df_TEST)

###################
#trading env

env = gym.make('gym_examples/TradingEnv-v0', df = df, window_size = 10)

done = False

price, state = env.reset()
print(price)
print(state)

learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

class BlackjackAgent:
    def __init__( self, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95):

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:

        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update( self, obs: tuple[int, int, bool], action: int, reward: float, terminated: bool, next_obs: tuple[int, int, bool],):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])

        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)

agent = BlackjackAgent(learning_rate=learning_rate, initial_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon,)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()











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

def to_ndarray(item, dtype):
    r"""
    Overview:
        Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.
    Arguments:
        - item (:obj:`object`): the item to be changed
        - dtype (:obj:`type`): the type of wanted ndarray
    Returns:
        - item (:obj:`object`): the changed ndarray
    .. note:
        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """

    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))
