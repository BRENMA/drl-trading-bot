import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numbers

INDICATORS = ['high','low','open','close','fng','rsi','macd','macd_signal','macd_hist','cci','dx','rf','sar','adx','adxr','apo','aroonosc','bop','cmo','minus_di','minus_dm','mom','plus_di','plus_dm','ppo_ta','roc','rocp','rocr','rocr100','trix','ultosc','willr','ht_dcphase','ht_sine','ht_trendmode','feature_PvEWMA_4','feature_PvCHLR_4','feature_RvRHLR_4','feature_CON_4','feature_RACORR_4','feature_PvEWMA_8','feature_PvCHLR_8','feature_RvRHLR_8','feature_CON_8','feature_RACORR_8','feature_PvEWMA_16','feature_PvCHLR_16','feature_RvRHLR_16','feature_CON_16','feature_RACORR_16']

class TradingEnv(gym.Env):

    def __init__(self, df, capital_frac = 0.2, cap_thresh=0.3, window_size = 10):

        self.seed()
        self.df = df
        self.window_size = window_size
        
        self.terminal_idx = len(self.df) - 1

        self.selected_feature_name = INDICATORS
        self.prices, self.signal_features, self.feature_dim_len = self._process_data()

        self._current_tick = 0
        self._done = None

        self.current_price = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, dtype = np.float64)

        self.capital_frac = capital_frac
        self.cap_thresh = cap_thresh

        self.initial_capital = 1000
        self.portfolio_value = self.initial_capital
        self.running_capital = self.initial_capital
        
        self.initial_token_balance = 0
        self.token_balance = self.initial_token_balance

        self.store = {"action_store": [], "reward_store": [], "running_capital": [], "port_ret": []}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _process_data(self):
        '''
        Overview:
            used by env.reset(), process the raw data.
        Arguments:
            - start_idx (int): the start tick; if None, then randomly select.
        Returns:
            - prices: the close.
            - selected_feature: feature map
            - feature_dim_len: the dimension length of selected feature
        '''
        prices = self.df.loc[:, 'close'].to_numpy()
        
        # ====== select features ========
        corr_matrix = self.df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool));

        # selecting the columns which are having absolute correlation greater than 0.95 and making a list of those columns named 'dropping_these_features'.
        dropping_these_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        for droppingFeature in dropping_these_features:
            self.selected_feature_name.remove(str(droppingFeature))

        selected_feature = np.column_stack((self.df.loc[:, k].to_numpy() for k in self.selected_feature_name))
        feature_dim_len = len(self.selected_feature_name)
 
        self.selected_feature_name.append('port_ovr_init_cap')
        self.selected_feature_name.append('run_ovr_port')
        self.selected_feature_name.append('investmnt_ovr_init_cap')

        return prices, selected_feature, feature_dim_len

    def reset(self):
        self.portfolio_value = self.initial_capital
        self.running_capital = self.initial_capital

        self.token_balance = self.initial_token_balance

        self.store = {"action_store": [], "reward_store": [], "running_capital": [], "port_ret": []}
        
        self.current_price = 0
        self._done = False

        self._current_tick = random.randrange(0, self.terminal_idx - 1)
        return self._get_observation()

    def close(self):
        plt.close()

    def step(self, action):
        #action will come as either a 0 (sell) or a 1 (buy)
        self._current_tick += 1

        #getting current asset price
        self.current_price = self.df.iloc[self._current_tick, :]['close']

        #grab reward
        reward = self._calculate_reward(action)

        #grab current params
        observation = self._get_observation()

        self.done = self.check_terminal()

        self.store["action_store"].append(action)
        self.store["reward_store"].append(reward)
        self.store["running_capital"].append(self.running_capital)
        info = self.store

        return observation, reward, self._done, info

    def _calculate_reward(self, action):

        #how much we're investing each buy
        investment = self.running_capital * self.capital_frac

        # Buy Action
        if action == 1:
            #if not terminal state
            if self.running_capital > self.initial_cap * self.running_thresh:
                
                #update running capital based on new investement
                self.running_capital -= investment

                #get how many tokens we're gonna buy
                asset_units = investment/self.current_price

                #buy them, add them to inventory
                self.token_balance += asset_units

        # Sell Action
        elif action == 0:
            #check to make sure we have tokens to sell
            if self.token_balance > 0:

                #add the sold token cash to running capital
                self.running_capital += self.token_balance * self.current_price

                #updating
                self.token_balance = 0

        #grabbing previous portfolio value
        prev_portfolio_value = self.portfolio_value
        
        #updating portolio value
        self.portfolio_value = self.running_capital + ((self.token_balance) * self.current_price)

        #getting new profit/loss
        price_diff = self.portfolio_value - prev_portfolio_value

        #init reward
        reward = 0

        #only make the reward > 0 if we made a profit
        if price_diff > 0:
            #calculating reward.
            reward += np.log(price_diff)
                
        return reward

    def check_terminal(self):
        if self._current_tick == self.terminal_idx:
            return True
        elif self.portfolio_value <= self.initial_cap * self.cap_thresh:
            return True
        else:
            return False

    def _get_observation(self):
        price = np.array([self.prices[self._current_tick]])
        state = self.signal_features[self._current_tick]
        state = np.concatenate([state, [self.portfolio_value/self.initial_capital, self.running_capital/self.portfolio_value, self.token_balance * self.current_price/self.initial_capital]])
        print(len(state))
        print(len(self.selected_feature_name))

        state = {self.selected_feature_name[i]: state[i] for i in range(len(state))}

        return price, state

