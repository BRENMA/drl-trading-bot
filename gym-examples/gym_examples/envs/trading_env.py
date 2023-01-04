import gym as gym
from gym import spaces
from gymnasium.utils import seeding
from enum import Enum

import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INDICATORS = ['high','low','open','close','fng','rsi','macd','macd_signal','macd_hist','cci','dx','rf','sar','adx','adxr','apo','aroonosc','bop','cmo','minus_di','minus_dm','mom','plus_di','plus_dm','ppo_ta','roc','rocp','rocr','rocr100','trix','ultosc','willr','ht_dcphase','ht_sine','ht_trendmode','feature_PvEWMA_4','feature_PvCHLR_4','feature_RvRHLR_4','feature_CON_4','feature_RACORR_4','feature_PvEWMA_8','feature_PvCHLR_8','feature_RvRHLR_8','feature_CON_8','feature_RACORR_8','feature_PvEWMA_16','feature_PvCHLR_16','feature_RvRHLR_16','feature_CON_16','feature_RACORR_16','feature_PvEWMA_32','feature_PvCHLR_32','feature_RvRHLR_32','feature_CON_32','feature_RACORR_32','feature_PvEWMA_64','feature_PvCHLR_64','feature_RvRHLR_64','feature_CON_64','feature_RACORR_64','feature_PvEWMA_128','feature_PvCHLR_128','feature_RvRHLR_128','feature_CON_128','feature_RACORR_128','feature_PvEWMA_256','feature_PvCHLR_256','feature_RvRHLR_256','feature_CON_256','feature_RACORR_256']

class Actions(Enum):
        Sell = 0
        Buy = 1
        Do_nothing = 2

class TradingEnv(gym.Env):
        metadata = {'render.modes': ['human']}

        def __init__(self, df, window_size, frame_bound, capital_frac = 0.2, cap_thresh=0.3, running_thresh=0.05):  
                assert len(frame_bound) == 2
                self.frame_bound = frame_bound

                self.selected_feature_name = INDICATORS

                self.seed()
                self.df = df
                self.window_size = window_size
                self.prices, self.signal_features = self._process_data()
                self.shape = (window_size, self.signal_features.shape[1])

                self.action_space = spaces.Discrete(len(Actions))
                self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=self.shape, dtype = np.float64)

                self._start_tick = self.window_size
                self._end_tick = len(self.prices) - 1
                self._done = None
                self._current_tick = 0
                self._last_trade_tick = None
                self._position = None
                self._position_history = None
                self._total_reward = None
                self._total_profit = None
                self.history = None

        def seed(self, seed=None):
                self.np_random, seed = seeding.np_random(seed)
                return [seed]

        def reset(self):
                self._done = False

                #self._current_tick = random.randrange(0, self._end_tick - 1 - self.window_size)
                self._current_tick = self._start_tick
                self._last_trade_tick = self._current_tick - 1

                self._position = 0
                self._position_history = (self.window_size * [None]) 

                self._total_reward = 0.
                self._total_profit = 0.  # unit
                self.history = {}

                #self.portfolio_value = self.initial_capital
                #self.running_capital = self.initial_capital
                #self.token_balance = self.initial_token_balance

                ##self._position = 0
                ##self._position_history = (self.window_size * [None]) 
                ##self._total_reward = 0.
                ##self._total_profit = 0.  # unit

                return self._get_observation()

        def _calculate_reward(self, action):
                step_reward = 0

                current_price = self.prices[self._current_tick]
                last_price = self.prices[self._current_tick - 1]
                price_diff = current_price - last_price

                # OPEN BUY - 1
                if action == Actions.Buy.value and self._position == 0:
                        self._position = 1
                        step_reward += price_diff
                        self._last_trade_tick = self._current_tick - 1
                        self._position_history.append(1)

                elif action == Actions.Buy.value and self._position > 0:
                        step_reward += 0
                        self._position_history.append(-1)

                # CLOSE SELL - 4
                elif action == Actions.Buy.value and self._position < 0:
                        self._position = 0
                        step_reward += - 1 * (self.prices[self._current_tick - 1] - self.prices[self._last_trade_tick])
                        self._total_profit += step_reward
                        self._position_history.append(4)

                # OPEN SELL - 3
                elif action == Actions.Sell.value and self._position == 0:
                        self._position = -1
                        step_reward += -1 * price_diff
                        self._last_trade_tick = self._current_tick - 1
                        self._position_history.append(3)

                # CLOSE BUY - 2
                elif action == Actions.Sell.value and self._position > 0:
                        self._position = 0
                        step_reward += self.prices[self._current_tick -1] - self.prices[self._last_trade_tick] 
                        self._total_profit += step_reward
                        self._position_history.append(2)

                elif action == Actions.Sell.value and self._position < 0:
                        step_reward += 0
                        self._position_history.append(-1)

                # DO NOTHING - 0
                elif action == Actions.Do_nothing.value and self._position > 0:
                        step_reward += price_diff
                        self._position_history.append(0)

                elif action == Actions.Do_nothing.value and self._position < 0:
                        step_reward += -1 * price_diff
                        self._position_history.append(0)

                elif action == Actions.Do_nothing.value and self._position == 0:
                        step_reward += -1 * abs(price_diff)
                        self._position_history.append(0)

                return step_reward

        def step(self, action):
                self._done = False
                self._current_tick += 1

                if self._current_tick == self._end_tick:
                        self._done = True

                step_reward = self._calculate_reward(action)
                self._total_reward += step_reward

                observation = self._get_observation()
                info = dict(total_reward = self._total_reward, total_profit = self._total_profit, position = self._position)
                self._update_history(info)

                return observation, step_reward, self._done, info

        def _get_observation(self):
                return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

        def _update_history(self, info):
                if not self.history:
                        self.history = {key: [] for key in info.keys()}

                for key, value in info.items():
                        self.history[key].append(value)

        def render(self, mode='human'):
                window_ticks = np.arange(len(self._position_history))
                plt.plot(self.prices)

                open_buy = []
                close_buy = []
                open_sell = []
                close_sell = []
                do_nothing = []

                for i, tick in enumerate(window_ticks):
                        if self._position_history[i] is None:
                                continue

                        if self._position_history[i] == 1:
                                open_buy.append(tick)
                        elif self._position_history[i] == 2 :
                                close_buy.append(tick)
                        elif self._position_history[i] == 3 :
                                open_sell.append(tick)
                        elif self._position_history[i] == 4 :
                                close_sell.append(tick)
                        elif self._position_history[i] == 0 :
                                do_nothing.append(tick)

                plt.plot(open_buy, self.prices[open_buy], 'go', marker="^")
                plt.plot(close_buy, self.prices[close_buy], 'go', marker="v")
                plt.plot(open_sell, self.prices[open_sell], 'ro', marker="v")
                plt.plot(close_sell, self.prices[close_sell], 'ro', marker="^")
        
                plt.plot(do_nothing, self.prices[do_nothing], 'yo')

                plt.suptitle(
                        "Total Reward: %.6f" % self._total_reward + ' ~ ' +
                        "Total Profit: %.6f" % self._total_profit
                )

        def close(self):
                plt.close()

        def save_rendering(self, filepath):
                plt.savefig(filepath)

        def pause_rendering(self):
                plt.show()

        def _process_data(self):
                prices = self.df.loc[:, 'close'].to_numpy()
                
                # ====== select features ========
                corr_matrix = self.df.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool));

                # selecting the columns which are having absolute correlation greater than 0.95 and making a list of those columns named 'dropping_these_features'.
                dropping_these_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.99)]
                for droppingFeature in dropping_these_features:
                        self.selected_feature_name.remove(str(droppingFeature))

                selected_feature = np.column_stack((self.df.loc[:, k].to_numpy() for k in self.selected_feature_name))

                #self.selected_feature_name.append('port_ovr_init_cap')
                #self.selected_feature_name.append('run_ovr_port')
                #self.selected_feature_name.append('investmnt_ovr_init_cap')

                return prices, selected_feature

