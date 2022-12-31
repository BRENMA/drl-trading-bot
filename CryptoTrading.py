from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
from torch import optim

import gymnasium as gym
from collections import defaultdict, namedtuple, deque
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
from config import *

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

#print(df)

t = DataProcessor(data_source='binance', start_date=TEST_START_DATE, end_date=TEST_END_DATE, time_interval=TIME_INTERVAL)
t.download_data(TICKER_LIST)
t.clean_data()
df_TEST = t.dataframe

#df_BTCUSDT = df[df['tic'] == 'BTCUSDT'].copy()

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

class ReplayMemory:
    """
    Implementation of Agent memory
    """
    def __init__(self, capacity=MEMORY_LEN):
        self.memory = deque(maxlen=capacity)

    def store(self, t):
        self.memory.append(t)

    def sample(self, n):
        a = random.sample(self.memory, n)
        return a

    def __len__(self):
        return len(self.memory)

class DuellingDQN(nn.Module):
    """
    Duelling Deep Q Network Agent
    """
    def __init__(self, input_dim=STATE_SPACE, output_dim=ACTION_SPACE):
        super(DuellingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 300)
        self.fc4 = nn.Linear(300, 200)
        self.fc5 = nn.Linear(200, 10)

        self.fcs = nn.Linear(10, 1)
        self.fcp = nn.Linear(10, self.output_dim)
        self.fco = nn.Linear(self.output_dim + 1, self.output_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        xs = self.relu(self.fcs(x))
        xp = self.relu(self.fcp(x))

        x = xs + xp - xp.mean()
        return x

class DQNAgent:
    """
    Implements the Agent components
    """

    def __init__(self, actor_net=DuellingDQN, memory=ReplayMemory()):
        
        self.actor_online = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
        self.actor_target = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
        self.actor_target.load_state_dict(self.actor_online.state_dict())
        self.actor_target.eval()

        self.memory = memory

        self.actor_criterion = nn.MSELoss()
        self.actor_op = optim.Adam(self.actor_online.parameters(), lr=LR_DQN)

        self.t_step = 0

    def act(self, state, eps=0.):
        self.t_step += 1
        
        state = state[1]
        state = np.fromiter(state.values(), dtype=float)
        state = torch.from_numpy(state).float().to(DEVICE).view(1, -1)

        self.actor_online.eval()
        with torch.no_grad():
            actions = self.actor_online(state)
        self.actor_online.train()

        if random.random() > eps:
            act = np.argmax(actions.cpu().data.numpy())
        else:
            act = random.choice(np.arange(ACTION_SPACE))
        return int(act)

    def learn(self):
        if len(self.memory) <= MEMORY_THRESH:
            return 0

        if self.t_step > LEARN_AFTER and self.t_step % LEARN_EVERY==0:
            # Sample experiences from the Memory
            batch = self.memory.sample(BATCH_SIZE)

            states = np.vstack([np.fromiter(t.States[1].values(), dtype=float) for t in batch])

            states = torch.from_numpy(states).float().to(DEVICE)

            actions = np.vstack([t.Actions for t in batch])
            actions = torch.from_numpy(actions).float().to(DEVICE)

            rewards = np.vstack([t.Rewards for t in batch])
            rewards = torch.from_numpy(rewards).float().to(DEVICE)

            next_states = np.vstack([np.fromiter(t.NextStates[1].values(), dtype=float) for t in batch])
            next_states = torch.from_numpy(next_states).float().to(DEVICE)

            dones = np.vstack([t.Dones for t in batch]).astype(np.uint8)
            dones = torch.from_numpy(dones).float().to(DEVICE)

            # ACTOR UPDATE
            # Compute next state actions and state values
            next_state_values = self.actor_target(next_states).max(1)[0].unsqueeze(1)
            y = rewards + (1-dones) * GAMMA * next_state_values
            state_values = self.actor_online(states).gather(1, actions.type(torch.int64))
            # Compute Actor loss
            actor_loss = self.actor_criterion(y, state_values)
            # Minimize Actor loss
            self.actor_op.zero_grad()
            actor_loss.backward()
            self.actor_op.step()

            # return actor_loss.item()
            if self.t_step % UPDATE_EVERY == 0:
                self.soft_update(self.actor_online, self.actor_target)

    def soft_update(self, local_model, target_model, tau=TAU):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

#df_LIST = [df_BTCUSDT, df_ETHUSDT, df_SOLUSDT, df_DOGEUSDT, df_ATOMUSDT, df_ADAUSDT, df_DOTUSDT, df_AVAXUSDT]
#df_LIST_Test = [df_BTCUSDT_TEST, df_ETHUSDT_TEST, df_SOLUSDT_TEST, df_DOGEUSDT_TEST, df_ATOMUSDT_TEST, df_ADAUSDT_TEST, df_DOTUSDT_TEST, df_AVAXUSDT_TEST]

env = gym.make('gym_examples/TradingEnv-v0', df = df)

Transition = namedtuple("Transition", ["States", "Actions", "Rewards", "NextStates", "Dones"])

memory = ReplayMemory()
agent = DQNAgent(actor_net = DuellingDQN, memory = memory)

# Main training loop
learning_rate = 0.01
n_episodes = 100

scores = []
eps = EPS_START

te_score_min = -np.Inf
for episode in tqdm(range(n_episodes)):
    score = 0
    counter = 0
    episode_score = 0
    episode_score2 = 0

    test_score = 0
    test_score2 = 0

    done = False

    state = env.reset()
    while not done:
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)

        t = Transition(state, action, reward, next_state, done)
        agent.memory.store(t)
        agent.learn()

        state = next_state
        score += reward
        counter += 1
        if done:
            break

    episode_score += score
    episode_score2 += (env.store['running_capital'][-1] - env.store['running_capital'][0])

    scores.append(episode_score)
    eps = max(EPS_END, EPS_DECAY * eps)

    #for i, test_env in enumerate(env):
    state = env.reset()
    done = False
    score_te = 0
    scores_te = [score_te]

    while True:
        actions = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        score_te += reward
        scores_te.append(score_te)
        if done:
            break

        test_score += score_te
        test_score2 += (env.store['running_capital'][-1] - env.store['running_capital'][0])
    
    if test_score > te_score_min:
        te_score_min = test_score
        torch.save(agent.actor_online.state_dict(), "online.pt")
        torch.save(agent.actor_target.state_dict(), "target.pt")

    print(f"Episode: {episode}, Train Score: {episode_score:.5f}, Validation Score: {test_score:.5f}")
    print(f"Episode: {episode}, Train Value: ${episode_score2:.5f}, Validation Value: ${test_score2:.5f}", "\n")


plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
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
