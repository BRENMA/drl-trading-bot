import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):

    def __init__(self, df, render_mode=None, capital_frac = 0.2, cap_thresh=0.3, window_size = 10):
        self.df = df
        self.terminal_idx = len(self.df) - 1

        #self.prices, self.signal_features, self.feature_dim_len = self._process_data()

        self._current_tick = 0
        self._done = None

        self.current_price = None

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = self.shape, dtype = np.float64)

        self.capital_frac = capital_frac
        self.cap_thresh = cap_thresh

        self.initial_capital = 1000
        self.portfolio_value = self.initial_capital
        self.running_capital = self.initial_capital
        self.token_balance = 0

        self.store = {"action_store": [], "reward_store": [], "running_capital": [], "port_ret": []}

        assert render_mode is None
        self.render_mode = render_mode

    #def _process_data(self):
    #    '''
    #    Overview:
    #        used by env.reset(), process the raw data.
    #    Arguments:
    #        - start_idx (int): the start tick; if None, then randomly select.
    #    Returns:
    #        - prices: the close.
    #        - selected_feature: feature map
    #        - feature_dim_len: the dimension length of selected feature
    #    '''
#
    #    # ====== build feature map ========
    #    all_feature = {k: self.df.loc[:, k].to_numpy() for k in INDICATORS}
    #    prices = self.df.loc[:, 'close'].to_numpy()
    #    # =================================
#
    #    # ====== select features ========
    #    corr_matrix = pd.DataFrame(self.df).corr().abs()
    #    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool));
#
    #    selected_feature_name = INDICATORS
#
    #    # selecting the columns which are having absolute correlation greater than 0.95 and making a list of those columns named 'dropping_these_features'.
    #    dropping_these_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.99)]
    #    for droppingFeature in dropping_these_features:
    #        selected_feature_name.remove(str(droppingFeature))
    #
    #    selected_feature = np.column_stack([all_feature[k] for k in selected_feature_name])
    #    feature_dim_len = len(selected_feature_name)
#
    #    return prices, selected_feature, feature_dim_len

    def reset(self):
        self.store = {"action_store": [], "reward_store": [], "running_capital": [], "port_ret": []}
        self.current_price = 0
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def close(self):
        plt.close()

    def step(self, action):
        #action will come as either a 0 (sell) or a 1 (buy)
        self._current_tick += 1

        #getting current asset price
        self.current_price = self.df.iloc[self._current_tick, :]['close']

        #grab reward
        reward = self._calculate_reward(action, self.current_price)

        #grab current params
        observation = self._get_observation(self._current_tick)

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

    def _get_observation(self, idx):
        state = self.df[idx][1:]
        state = self.scaler.transform(state.reshape(1, -1))
        state = np.concatenate([state, [[self.portfolio_value/self.initial_capital, self.running_capital/self.portfolio_value, self.token_balance * self.current_price/self.initial_capital]]], axis=-1)
        return state

    def render(self):
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

        plt.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        plt.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        plt.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        plt.savefig(self.save_path + str(self._env_id) + '-price.png')
