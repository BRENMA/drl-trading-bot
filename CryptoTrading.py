from __future__ import annotations

from ding.config import compile_config
from CryptoConfig import main_config, create_config

from cmath import inf
from typing import Any, List, Optional, Callable, Tuple
from easydict import EasyDict
from abc import abstractmethod
import copy

from collections import deque, namedtuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
from sklearn.preprocessing import MinMaxScaler

from ding.model import DQN
from ding.policy import DQNPolicy
from ding.data import DequeBuffer

from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, eps_greedy_handler, CkptSaver


import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from enum import Enum

import math
import requests
import pandas as pd
import numpy as np
import numpy.random as rd

from meta.data_processor import DataProcessor

from datetime import datetime
import time

from talib.abstract import MACD, RSI, CCI, DX
import talib as ta

#from imblearn.over_sampling import SMOTE

print("\n\n\n'88,dPYba,,adPYba,   ,adPPYba,  8b,dPPYba,   ,adPPYba, 8b       d8  \n","88P'   '88'    '8a a8'     '8a 88P'   `'8a a8P_____88 `8b     d8'  \n","88      88      88 8b       d8 88       88 8PP'''''''  `8b   d8'   \n","88      88      88 '8a,   ,a8' 88       88 '8b,   ,aa   `8b,d8'    \n","88      88      88  `'YbbdP''  88       88  `'Ybbd8''     Y88'     \n","                                                          d8'      \n","                                                         d8'       \n")
print("> welcome to moneyDRL")
print("> Creating Testing Data")

TICKER_LIST = ["ETHUSDT"]#, "ATOMUSDT", "ADAUSDT", "BTCUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT"]
INDICATORS = ['high','low','open','close','fng','rsi','macd','macd_signal','macd_hist','cci','dx','rf','sar','adx','adxr','apo','aroonosc','bop','cmo','minus_di','minus_dm','mom','plus_di','plus_dm','ppo_ta','roc','rocp','rocr','rocr100','trix','ultosc','willr','ht_dcphase','ht_sine','ht_trendmode','feature_PvEWMA_4','feature_PvCHLR_4','feature_RvRHLR_4','feature_CON_4','feature_RACORR_4','feature_PvEWMA_8','feature_PvCHLR_8','feature_RvRHLR_8','feature_CON_8','feature_RACORR_8','feature_PvEWMA_16','feature_PvCHLR_16','feature_RvRHLR_16','feature_CON_16','feature_RACORR_16']
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

class TradingEnv(gym.Env):

    def __init__(self, df, render_mode=None, capital_frac = 0.2, cap_thresh=0.3):
        self.trade_fee_bid_percent = 0.
        self.trade_fee_ask_percent = 0.

        self.asset_data = df
        self._current_tick = None
        self._done = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = self.shape, dtype = np.float64)

        self.terminal_idx = len(self.asset_data) - 1

        self.usd_balance = 1000
        self.capital_frac = capital_frac
        self.cap_thresh = cap_thresh

        self.token_amount = 0
        self.token_balance = 0
        self.token_price = 0
        self.bought_token_at = 0
        self.last_transaction_was_sell = False

        assert render_mode is None
        self.render_mode = render_mode

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def close(self):
        raise NotImplementedError

    def step(self, action):
        #action will come as either a 0 (sell) or a 1 (buy)

        self.done = self.check_terminal()

        reward = self._calculate_reward(action)

        self._position = self._position.opposite()
        self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, reward, self._done, info

    def _calculate_reward(self, action):
        step_reward = 0


        return step_reward

    def check_terminal(self):
        if self.pointer == self.terminal_idx:
            return True
        elif self.capital <= self.initial_cap * self.cap_thresh:
            return True
        else:
            return False

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

class TradingEnv():

    def __init__(self, cfg: EasyDict) -> None:

        self._cfg = cfg
        self._env_id = cfg.env_id
        #======== param to plot =========
        self.cnt = 0

        if 'plot_freq' not in self._cfg:
            self.plot_freq = 10
        else:
            self.plot_freq = self._cfg.plot_freq
        if 'save_path' not in self._cfg:
            self.save_path = './'
        else:
            self.save_path = self._cfg.save_path
        #================================

        self.train_range = cfg.train_range
        self.test_range = cfg.test_range
        self.window_size = cfg.window_size
        self.prices = None
        self.signal_features = None
        self.feature_dim_len = None
        self.shape = (cfg.window_size, 3)

        #======== param about episode =========
        self._start_tick = 0
        self._end_tick = 0
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        #======================================

        self._init_flag = True
        # init the following variables variable at first reset.
        self._action_space = None
        self._observation_space = None
        self._reward_space = None

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, start_idx: int = None) -> Any:
        self.cnt += 1
        self.prices, self.signal_features, self.feature_dim_len = self._process_data(start_idx)
        if self._init_flag:
            self.shape = (self.window_size, self.feature_dim_len)
            self._action_space = spaces.Discrete(len(Actions))
            self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
            self._reward_space = gym.spaces.Box(-inf, inf, shape=(1, ), dtype=np.float32)
            self._init_flag = False
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.FLAT
        self._position_history = [self._position]
        self._profit_history = [1.]
        self._total_reward = 0.

        return self._get_observation()

    def random_action(self) -> Any:
        return np.array([self.action_space.sample()])

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.item()  # 0-dim array

        self._done = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._position, trade = transform(self._position, action)

        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        self._profit_history.append(float(np.exp(self._total_reward)))
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            position=self._position.value,
        )

        if self._done:
            if self._env_id[-1] == 'e' and self.cnt % self.plot_freq == 0:
                self.render()
            info['max_possible_profit'] = np.log(self.max_possible_profit())
            info['eval_episode_return'] = self._total_reward

        step_reward = to_ndarray([step_reward]).astype(np.float32)
        return BaseEnvTimestep(observation, step_reward, self._done, info)

    def _get_observation(self) -> np.ndarray:
        obs = to_ndarray(self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]).reshape(-1).astype(np.float32)

        tick = (self._current_tick - self._last_trade_tick) / self._cfg.eps_length
        obs = np.hstack([obs, to_ndarray([self._position.value]), to_ndarray([tick])]).astype(np.float32)
        return obs

    def render(self) -> None:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('profit')
        plt.plot(self._profit_history)
        plt.savefig(self.save_path + str(self._env_id) + "-profit.png")

        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('close price')
        window_ticks = np.arange(len(self._position_history))
        eps_price = self.raw_prices[self._start_tick:self._end_tick + 1]
        plt.plot(eps_price)

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.SHORT:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.LONG:
                long_ticks.append(tick)
            else:
                flat_ticks.append(tick)

        plt.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        plt.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        plt.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        plt.savefig(self.save_path + str(self._env_id) + '-price.png')

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()

    # override
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in env manager \
            (a series of vectorized env), and this method is mainly responsible for envs collecting data.
            In TradingEnv, this method will rename every env_id and generate different config.
        Arguments:
            - cfg (:obj:`dict`): Original input env config, which needs to be transformed into the type of creating \
                env instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config collector envs.
        .. note::
            Elements(env config) in collector_env_cfg/evaluator_env_cfg can be different, such as server ip and port.
        """
        collector_env_num = cfg.pop('collector_env_num')
        collector_env_cfg = [copy.deepcopy(cfg) for _ in range(collector_env_num)]
        for i in range(collector_env_num):
            collector_env_cfg[i]['env_id'] += ('-' + str(i) + 'e')
        return collector_env_cfg

    # override
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in env manager \
            (a series of vectorized env), and this method is mainly responsible for envs evaluating performance.
            In TradingEnv, this method will rename every env_id and generate different config.
        Arguments:
            - cfg (:obj:`dict`): Original input env config, which needs to be transformed into the type of creating \
                env instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config evaluator envs.
        """
        evaluator_env_num = cfg.pop('evaluator_env_num')
        evaluator_env_cfg = [copy.deepcopy(cfg) for _ in range(evaluator_env_num)]
        for i in range(evaluator_env_num):
            evaluator_env_cfg[i]['env_id'] += ('-' + str(i) + 'e')
        return evaluator_env_cfg

    @abstractmethod
    def _process_data(self, df, start_idx: int = None):
        '''
        Overview:
            used by env.reset(), process the raw data.
        Arguments:
            - start_idx (int): the start tick; if None, then randomly select.
        Returns:
            - prices: the close.
            - signal_features: feature map
            - feature_dim_len: the dimension length of selected feature
        '''

        # ====== build feature map ========
        all_feature = {k: df.loc[:, k].to_numpy() for k in INDICATORS}
        prices = df.loc[:, 'Close'].to_numpy()
        # =================================

        # ====== select features ========
        corr_matrix = pd.DataFrame(df).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool));
        
        selected_feature_name = INDICATORS

        # selecting the columns which are having absolute correlation greater than 0.95 and making a list of those columns named 'dropping_these_features'.
        dropping_these_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.99)]
        print(dropping_these_features)
        for droppingFeature in dropping_these_features:
            selected_feature_name.remove(str(droppingFeature))

        print(selected_feature_name)
    
        selected_feature = np.column_stack([all_feature[k] for k in selected_feature_name])
        feature_dim_len = len(selected_feature_name)

        # validate index
        if start_idx is None:
            if self.train_range == None or self.test_range == None:
                self.start_idx = np.random.randint(self.window_size, len(self.df) - self._cfg.eps_length)
            elif self._env_id[-1] == 'e':
                boundary = int(len(self.df) * (1 + self.test_range))
                assert len(self.df) - self._cfg.eps_length > boundary + self.window_size,\
                "parameter test_range is too large!"
                self.start_idx = np.random.randint(boundary + self.window_size, len(self.df) - self._cfg.eps_length)
            else:
                boundary = int(len(self.df) * self.train_range)
                assert boundary - self._cfg.eps_length > self.window_size,\
                "parameter test_range is too small!"
                self.start_idx = np.random.randint(self.window_size, boundary - self._cfg.eps_length)
        else:
            self.start_idx = start_idx

        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self._cfg.eps_length - 1

        return prices, selected_feature, feature_dim_len

    @abstractmethod
    def _calculate_reward(self, action: int) -> np.float32:
        step_reward = 0.
        current_price = (self.raw_prices[self._current_tick])
        last_trade_price = (self.raw_prices[self._last_trade_tick])
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent))

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        step_reward = float(step_reward)

        return step_reward

    @abstractmethod
    def max_possible_profit(self) -> float:
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            if self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]:
                while (current_tick <= self._end_tick and self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]): 
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (2 - (current_price / last_trade_price)) * (1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick
                    and self.raw_prices[current_tick] >= self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (current_price / last_trade_price) * (1 - self.trade_fee_ask_percent
                                                                            ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            last_trade_tick = current_tick - 1

        return profit

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Trading Env"

