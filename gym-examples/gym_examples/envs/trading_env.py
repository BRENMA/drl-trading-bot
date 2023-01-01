import gym as gym
from gym import spaces
from gym.utils import seeding

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

INDICATORS = ['high','low','open','close','fng','rsi','macd','macd_signal','macd_hist','cci','dx','rf','sar','adx','adxr','apo','aroonosc','bop','cmo','minus_di','minus_dm','mom','plus_di','plus_dm','ppo_ta','roc','rocp','rocr','rocr100','trix','ultosc','willr','ht_dcphase','ht_sine','ht_trendmode','feature_PvEWMA_4','feature_PvCHLR_4','feature_RvRHLR_4','feature_CON_4','feature_RACORR_4','feature_PvEWMA_8','feature_PvCHLR_8','feature_RvRHLR_8','feature_CON_8','feature_RACORR_8','feature_PvEWMA_16','feature_PvCHLR_16','feature_RvRHLR_16','feature_CON_16','feature_RACORR_16']

class TradingEnv(gym.Env):

    def __init__(self, df, capital_frac = 0.2, cap_thresh=0.3, running_thresh=0.05):

        self.seed()
        self.df = df
        self.prices, self.signal_features, self.feature_dim_len = self._process_data()

        self.window_size = 10
        self.shape = (self.window_size, self.signal_features.shape[1])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=self.shape, dtype = np.float64)

        self._start_tick = self.window_size
        self._end_tick = len(self.df) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None

        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        #self.renderN = 1

        #self.selected_feature_name = INDICATORS

        #self.current_price = None

        #self.capital_frac = capital_frac
        #self.cap_thresh = cap_thresh
        #self.running_thresh = running_thresh

        #self.initial_capital = 1000
        #self.portfolio_value = self.initial_capital
        #self.running_capital = self.initial_capital

        #self.initial_token_balance = 0
        #self.token_balance = self.initial_token_balance

        #self.store = {"action_store": [], "reward_store": [], "running_capital": [], "token_balance": [], "portfolio_value": [], "price": []}

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
        self._done = False

        #self._current_tick = random.randrange(0, self._end_tick - 1)
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        #self.portfolio_value = self.initial_capital
        #self.running_capital = self.initial_capital
        #self.token_balance = self.initial_token_balance
        #self.store = {"action_store": [], "reward_store": [], "running_capital": [], "token_balance": [], "portfolio_value": [], "price": []}
        #self.current_price = 0

        return self._get_observation()

    def close(self):
        plt.close()

    def step(self, action):
        #action will come as either a 0 (sell) or a 1 (buy)
        self._done = False
        self._current_tick += 1

        self._done = self.check_terminal()

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        self._last_trade_tick = self._current_tick

        observation = self._get_observation()

        info = dict( total_reward = self._total_reward, total_profit = self._total_profit, position = self._position.value)
        self._update_history(info)

        #getting current asset price
        #self.current_price = self.df.iloc[self._current_tick, :]['close']

        #grab current params
        #observation = self._get_observation()

        #self.store["action_store"].append(action)
        #self.store["reward_store"].append(reward)
        #self.store["running_capital"].append(self.running_capital)
        #self.store["token_balance"].append(self.token_balance)
        #self.store["portfolio_value"].append(self.portfolio_value)
        #self.store["price"].append(self.current_price)
        #info = self.store

        return observation, step_reward, self._done, info

    def _update_profit(self):
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        shares = self._total_profit / last_trade_price
        self._total_profit = shares * current_price

    def _calculate_reward(self, action):

        #how much we're investing each buy
        investment = self.running_capital * self.capital_frac

        # Buy Action
        if action == 1:
            #if not terminal state
            if self.running_capital > self.initial_capital * self.running_thresh:
                
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
        if self._current_tick == self._end_tick:
            return True
        elif self.portfolio_value <= self.initial_capital * self.cap_thresh:
            return True
        else:
            return False

    def _get_observation(self):
        #price = np.array([self.prices[self._current_tick]])
        #state = self.signal_features[self._current_tick]
        #state = np.concatenate([state, [self.portfolio_value/self.initial_capital, self.running_capital/self.portfolio_value, self.token_balance * self.current_price/self.initial_capital]])
        #state = {self.selected_feature_name[i]: state[i] for i in range(len(state))}
        #return price, state

        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def render(self):
        import matplotlib.pyplot as plt

        self.renderN += 1

        plt.clf()
        plt.figure(figsize=(10, 10), dpi=100)

        plt.xlabel('minutes')
        plt.ylabel('value')

        #plt.plot(self.store["action_store"], 'ro')
        #plt.plot(self.store["reward_store"], 'bs')
        #plt.plot(self.store["running_capital"], color = 'blue')
        #plt.plot(self.store["token_balance"], color = 'black')
        plt.plot(self.store["portfolio_value"], color = 'green')
        plt.plot(self.store["price"], color = 'red')

        plt.savefig('run' + str(self.renderN) + '-complete.png')
