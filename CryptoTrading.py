from __future__ import annotations

import gym as gym
from config import *
from gym_examples.envs.trading_env import TradingEnv

from sklearn.preprocessing import MinMaxScaler
import math
import requests
import pandas as pd
import numpy as np
from meta.data_processor import DataProcessor
from datetime import datetime
import time
from talib.abstract import MACD, RSI, CCI, DX
import talib as ta
from typing import Dict

import torch.nn as nn
from matplotlib import pyplot as plt
import random

from collections import namedtuple, deque

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from torch.utils.data import Dataset, DataLoader

print('\n'+'\n'+'\n'+"                                   )\._.,--....,'``.      " + '\n' + "                                  /;   _.. \   _\  (`._ ,." + '\n' + "                    $            `----(,_..'--(,_..'`-.;.'" + '\n' + '\n' + "                                                       /$$$$$$$  /$$$$$$$  /$$      " + "\n" +"                                                      | $$__  $$| $$__  $$| $$      " + "\n" +" /$$$$$$/$$$$   /$$$$$$  /$$$$$$$   /$$$$$$  /$$   /$$| $$  \ $$| $$  \ $$| $$      " + "\n" +"| $$_  $$_  $$ /$$__  $$| $$__  $$ /$$__  $$| $$  | $$| $$  | $$| $$$$$$$/| $$      " + "\n" +"| $$ \ $$ \ $$| $$  \ $$| $$  \ $$| $$$$$$$$| $$  | $$| $$  | $$| $$__  $$| $$      " + "\n" +"| $$ | $$ | $$| $$  | $$| $$  | $$| $$_____/| $$  | $$| $$  | $$| $$  \ $$| $$      " + "\n" +"| $$ | $$ | $$|  $$$$$$/| $$  | $$|  $$$$$$$|  $$$$$$$| $$$$$$$/| $$  | $$| $$$$$$$$" + "\n" +"|__/ |__/ |__/ \______/ |__/  |__/ \_______/ \____  $$|_______/ |__/  |__/|________/" + "\n" +"                                             /$$  | $$                              " + "\n" +"                                            |  $$$$$$/                              " + "\n" +"                                             \______/                               " + "\n" + "\n")
print("creating Testing Data")

p = DataProcessor(data_source='binance', start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(TICKER_LIST)
p.clean_data()
df = p.dataframe

t = DataProcessor(data_source='binance', start_date=TEST_START_DATE, end_date=TEST_END_DATE, time_interval=TIME_INTERVAL)
t.download_data(TICKER_LIST)
t.clean_data()
df_test = t.dataframe

def addFnG(df):
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
    dollar_threshold = DOLLAR_THRESHOLD

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

    return df

df = addFnG(df = df)
df_test = addFnG(df = df_test)

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
        tic_df['ht_dcperiod'] = ta.HT_DCPERIOD(df['close'])
        tic_df['ht_dcphase'] = ta.HT_DCPHASE(df['close'])
        tic_df['ht_sine'], _ = ta.HT_SINE(df['close'])
        tic_df['ht_trendmode'] = ta.HT_TRENDMODE(df['close'])

        # for each period length
        for period_length in period_lengths:
            # add the feature columns to the bars df
            add_feature_columns(tic_df, period_length)

        # prune the nan rows at the beginning of the dataframe
        tic_df = tic_df.dropna()

        final_df = pd.concat([tic_df, final_df])

    df = final_df
    df.index=pd.to_datetime(df.time)
    df.drop('time', inplace=True, axis=1)
    df.drop('tic', inplace=True, axis=1)

    min_max_scaler = MinMaxScaler()
    for column in df.columns:
        if column != 'close':
            df[column] = min_max_scaler.fit_transform(df[[column]])

    df = df.dropna()

    return df

df = addIndicators(df = df)
df_test = addIndicators(df = df_test)

#df.to_csv('data.csv')

#TRAINING ====
############env_build = lambda: TradingEnv(df=df, frame_bound=(10,len(df)), window_size=10)
############env = DummyVecEnv([env_build])
############
############model_train = PPO(policy = 'MlpPolicy', env = env, learning_rate=0.0005, batch_size=2048, gamma = 0.985, verbose=1)
############model_train.learn(total_timesteps=4000000)
############model_train.save("ppo_crypto")
#==============

#TESTING =======
##########env = TradingEnv(df=df, frame_bound=(10,len(df)), window_size=10)
##########model = PPO.load("ppo_crypto", env=env)
##########
##########obs = env.reset()
##########while True: 
##########    obs = obs[np.newaxis, ...]
##########    action, _states = model.predict(obs)
##########    obs, rewards, done, info = env.step(action)
##########    if done:
##########        print("info", info)
##########        break
##########
##########plt.figure(figsize=(25,10))
##########plt.cla()
##########env.render()
##########env.save_rendering('test.png')

#=======

Transition = namedtuple("Transition", ["States", "Actions", "Rewards", "NextStates", "Dones"])

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
        self.actor_op = torch.optim.Adam(self.actor_online.parameters(), lr=LR_DQN)

        self.t_step = 0


    def act(self, state, eps=0.):
        self.t_step += 1
        state = state[0]
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

            states = np.vstack([t.States for t in batch])
            states = torch.from_numpy(states).float().to(DEVICE)

            actions = np.vstack([t.Actions for t in batch])
            actions = torch.from_numpy(actions).float().to(DEVICE)

            rewards = np.vstack([t.Rewards for t in batch])
            rewards = torch.from_numpy(rewards).float().to(DEVICE)

            next_states = np.vstack([t.NextStates for t in batch])
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

            if self.t_step % UPDATE_EVERY == 0:
                self.soft_update(self.actor_online, self.actor_target)
            # return actor_loss.item()

    def soft_update(self, local_model, target_model, tau=TAU):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

memory = ReplayMemory()
agent = DQNAgent(actor_net=DuellingDQN, memory=memory)

env = TradingEnv(df=df, frame_bound=(1,len(df)), window_size=1)
test_env = TradingEnv(df=df_test, frame_bound=(1,len(df_test)), window_size=1)

# Main training loop
N_EPISODES = 200 # No of episodes/epochs
scores = []
eps = EPS_START

te_score_min = 0
for episode in range(1, 1 + N_EPISODES):
    counter = 0
    episode_score = 0
    episode_score2 = 0
    test_score = 0
    test_score2 = 0

    score = 0
    state = env.reset()
    state = state.reshape(-1, STATE_SPACE)
    while True:
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape(-1, STATE_SPACE)

        t = Transition(state, action, reward, next_state, done)
        agent.memory.store(t)
        agent.learn()

        state = next_state
        score += reward
        counter += 1
        if done:
            break

    episode_score += score

    scores.append(episode_score)
    eps = max(EPS_END, EPS_DECAY * eps)

    state_test = test_env.reset()
    done = False
    score_te = 0
    scores_te = [score_te]

    while True:
        action = agent.act(state_test)
        next_state, reward, done, _ = test_env.step(action)
        next_state = next_state.reshape(-1, STATE_SPACE)
        state_test = next_state
        score_te += reward
        scores_te.append(score_te)
        if done:
            break

    test_score += score_te

    if test_score > te_score_min:
        te_score_min = test_score
        torch.save(agent.actor_online.state_dict(), "online.pt")
        torch.save(agent.actor_target.state_dict(), "target.pt")

    print(f"Episode: {episode}, Train Score: {episode_score:.5f}")
    print(f"Episode: {episode}, Test Value: {test_score:.5f}", "\n")

