from __future__ import annotations

import ray

import math
import requests
import pandas as pd
import numpy as np
import numpy.random as rd

import os

from copy import deepcopy
from sklearn import preprocessing

import gym
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.normal import Normal
from sklearn.preprocessing import MinMaxScaler 

from meta.data_processor import DataProcessor

from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3

from datetime import datetime
import time

from talib.abstract import MACD, RSI, CCI, DX
import talib as ta

#from imblearn.over_sampling import SMOTE

print("\n\n\n'88,dPYba,,adPYba,   ,adPPYba,  8b,dPPYba,   ,adPPYba, 8b       d8  \n","88P'   '88'    '8a a8'     '8a 88P'   `'8a a8P_____88 `8b     d8'  \n","88      88      88 8b       d8 88       88 8PP'''''''  `8b   d8'   \n","88      88      88 '8a,   ,a8' 88       88 '8b,   ,aa   `8b,d8'    \n","88      88      88  `'YbbdP''  88       88  `'Ybbd8''     Y88'     \n","                                                          d8'      \n","                                                         d8'       \n")
print("> welcome to moneyDRL")
print("> Creating Testing Data")

TICKER_LIST = ["ETHUSDT"]#, "ATOMUSDT", "ADAUSDT", "BTCUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT"]
INDICATORS = ['open','fng', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'cci', 'dx', 'rf', 'sar', 'adx', 'adxr', 'apo', 'aroonosc', 'bop', 'cmo', 'minus_di', 'minus_dm', 'mom', 'plus_di', 'plus_dm', 'ppo_ta', 'roc', 'rocp', 'rocr', 'rocr100', 'trix', 'ultosc', 'willr', 'ht_dcphase', 'ht_sine', 'ht_trendmode', 'feature_PvEWMA_4', 'feature_PvCHLR_4', 'feature_RvRHLR_4', 'feature_CON_4', 'feature_RACORR_4', 'feature_PvEWMA_8', 'feature_PvCHLR_8', 'feature_RvRHLR_8', 'feature_CON_8', 'feature_RACORR_8', 'feature_PvEWMA_16', 'feature_PvCHLR_16', 'feature_RvRHLR_16', 'feature_CON_16', 'feature_RACORR_16', 'feature_PvEWMA_32', 'feature_PvCHLR_32', 'feature_RvRHLR_32', 'feature_CON_32', 'feature_RACORR_32', 'feature_PvEWMA_64', 'feature_PvCHLR_64', 'feature_RvRHLR_64', 'feature_CON_64', 'feature_RACORR_64', 'feature_PvEWMA_128', 'feature_PvCHLR_128', 'feature_RvRHLR_128', 'feature_CON_128', 'feature_RACORR_128', 'feature_PvEWMA_256', 'feature_PvCHLR_256', 'feature_RvRHLR_256', 'feature_CON_256', 'feature_RACORR_256']
period_lengths = [4, 8, 16, 32, 64, 128, 256]

TIME_INTERVAL = '1m'
TRAIN_START_DATE = '2020-07-01'
TRAIN_END_DATE= '2020-08-01'

TEST_START_DATE = '2021-07-01'
TEST_END_DATE = '2021-08-01'

p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(TICKER_LIST)
p.clean_data()
df = p.dataframe

t = DataProcessor(data_source='binance', start_date=TEST_START_DATE, end_date=TEST_END_DATE, time_interval=TIME_INTERVAL)
t.download_data(TICKER_LIST)
t.clean_data()
df_TEST = t.dataframe

print(len(df))
print(len(df_TEST))

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
df_TEST = addFnG(df = df_TEST)

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
        tic_df = tic_df[[column for column in tic_df.columns if column not in ['high', 'low']]]

        final_df = final_df.append(tic_df)

    df = final_df
    df.index=pd.to_datetime(df.time)
    df.drop('time', inplace=True, axis=1)
    df = df.dropna()
    return df

df = addIndicators(df = df)
df_TEST = addIndicators(df = df_TEST)

print(list(df.columns.values))
print(list(df_TEST.columns.values))

def splitIntoArr(df): 
    unique_ticker = df.tic.unique()
    price_array = np.column_stack([df[df.tic == tic].close for tic in unique_ticker])

    #column_titles = list(df.columns.values)
    #column_titles.remove('tic')
    #column_titles.remove('close')
    #common_tech_indicator_list = [i for i in column_titles if i in df.columns.values.tolist()]
    tech_array = np.hstack([df.loc[(df.tic == tic), INDICATORS] for tic in unique_ticker])

    turbulence_array = np.array([])

    min_max_scaler = MinMaxScaler()
    price_array = min_max_scaler.fit_transform(price_array)
    tech_array = min_max_scaler.fit_transform(tech_array)

    return price_array, tech_array, turbulence_array

price_array, tech_array, turbulence_array = splitIntoArr(df = df)
price_array_TEST, tech_array_TEST, turbulence_array_TEST = splitIntoArr(df = df_TEST)
print("Successfully transformed into array")

print(len(price_array))
print(len(price_array_TEST))

print(tech_array.shape)
print(tech_array_TEST.shape)

#CONFIG
class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)

        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.action_dim = len(TICKER_LIST)
        self.state_dim = 1 + len(INDICATORS) * self.action_dim
        #self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        #self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        self.agent_class = agent_class  # agent = agent_class(...)

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(128)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(2000)  # collect horizon_len step while exploring, then update network
        self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
        self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.eval_times = int(32)  # number of times that get episodic cumulative return
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)

#MOVING TO TRADING
class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        #turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        #self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        #self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        #self.turbulence_ary = (self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        #if self.turbulence_bool[self.day] == 0:
        min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
        for index in np.where(actions < -min_action)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.amount += (
                    price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                )
                self.stocks_cool_down[index] = 0
        for index in np.where(actions > min_action)[0]:  # buy_index:
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                buy_num_shares = min(self.amount // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.amount -= (
                    price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                )
                self.stocks_cool_down[index] = 0

        #else:  # sell all when turbulence
        #    self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
        #    self.stocks[:] = 0
        #    self.stocks_cool_down[:] = 0

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                #self.turbulence_ary[self.day],
                #self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

class ActorPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)

        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()

class CriticPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value

def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)

def get_gym_env_args(env, if_print: bool) -> dict:
    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]

    env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete}
    print(f"env_args = {repr(env_args)}") if if_print else None
    return env_args

def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs

def build_env(env_class=None, env_args=None):
    if env_class.__module__ == 'gym.envs.registration':  # special rule
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env

class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.soft_update_tau = args.soft_update_tau

        self.states = None  # assert self.states == (1, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), args.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), args.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

class AgentPPO(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(self, env, horizon_len: int) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.states[0]

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action, logprob = [t.squeeze(0) for t in get_action(state.unsqueeze(0))[:2]]

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> [float]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = torch.tensor(self.states, dtype=torch.float32).to(self.device)
        next_value = self.cri(next_state).detach()[0, 0]

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages

class PendulumEnv(gym.Wrapper):  # a demo of custom gym env
    def __init__(self):
        gym.logger.set_level(40)  # Block warning
        gym_env_name = "Pendulum-v0" if gym.__version__ < '0.18.0' else "Pendulum-v1"
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self) -> np.ndarray:  # reset the agent in env
        return self.env.reset()

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):  # agent interacts in env
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, done, info_dict = self.env.step(action * 2)
        return state.reshape(self.state_dim), float(reward), done, info_dict
    
def train_agent(args: Config):
    args.init_before_training()

    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.states = env.reset()[np.newaxis, :]

    evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                            eval_per_step=args.eval_per_step,
                            eval_times=args.eval_times,
                            cwd=args.cwd)
    torch.set_grad_enabled(False)
    while True: # start training
        buffer_items = agent.explore_env(env, args.horizon_len)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer_items)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            torch.save(agent.act.state_dict(), args.cwd + '/actor.pth')
            break  # stop training when reach `break_step` or `mkdir cwd/stop`

def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    print(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")
        
class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = []
        print(f"\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              f"\n| `time`: Time spent from the start of training to this moment."
              f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `avgS`: Average of steps in an episode."
              f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")
            
    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))
        
        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")

def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):  # cumulative_rewards and episode_steps
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(12345):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if done:
            break
    return cumulative_returns, episode_steps + 1

MODELS = {"ppo": AgentPPO}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]

class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }
        environment = self.env(config=env_config)
        env_args = {
                'config': env_config,
                'env_name': environment.env_name,
                'state_dim': environment.state_dim,
                'action_dim': environment.action_dim,
                'if_discrete': False
            }
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = Config(agent_class=agent, env_class=self.env, env_args=env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dims = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
                model.eval_times = model_kwargs["eval_times"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent_class = MODELS[model_name]
        environment.env_num = 1
        agent = agent_class(net_dimension, environment.state_dim, environment.action_dim)
        actor = agent.act
        # load agent
        try:  
            cwd = cwd + '/actor.pth'
            print(f"| load actor from: {cwd}")
            actor.load_state_dict(torch.load(cwd, map_location=lambda storage, loc: storage))
            act = actor
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = environment.step(action)

                total_asset = (
                    environment.amount
                    + (
                        environment.price_ary[environment.day] * environment.stocks
                    ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return episode_total_assets

env = StockTradingEnv
ERL_PARAMS = {"learning_rate": 3e-6, "batch_size": 2048, "gamma": 0.99, "seed":312, "net_dimension": [128,64],  "target_step": 5000, "eval_gap": 30, "eval_times": 1}

#training
def train(drl_lib, env, model_name, **kwargs,):
    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))
    env, price_array, tech_array, turbulence_array

    if drl_lib == "elegantrl":
        DRLAgent_erl = DRLAgent
        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(env=env,price_array=price_array,tech_array=tech_array,turbulence_array=turbulence_array,)
        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(model=model, cwd=cwd, total_timesteps=break_step)
train(drl_lib='elegantrl', env=env,price_array = price_array, tech_array = tech_array, turbulence_array = turbulence_array,model_name='ppo',erl_params=ERL_PARAMS,cwd='./test_ppo',break_step=1e5)

#testing
def test(drl_lib, env, model_name, net_dimension, **kwargs,):
    env_config = {
        "price_array": price_array_TEST,
        "tech_array": tech_array_TEST,
        "turbulence_array": turbulence_array_TEST,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == "elegantrl":
        DRLAgent_erl = DRLAgent
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )
        return episode_total_assets
account_value_erl = test (drl_lib='elegantrl', env=env, model_name='ppo', cwd='./test_ppo', net_dimension = [128,64])
print(account_value_erl)