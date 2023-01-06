from gym_examples.envs.trading_env import TradingEnv

from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from talib.abstract import MACD, RSI, CCI, DX
import talib as ta

from dydx3 import Client
from dydx3.constants import *
from dydx3.constants import MARKET_ETH_USD, ORDER_SIDE_BUY, ORDER_SIDE_SELL, ORDER_TYPE_MARKET, TIME_IN_FORCE_FOK
from web3 import Web3
from decimal import Decimal

import pandas as pd
import requests
import time
import math
from enum import Enum
import calendar
import datetime

INDICATORS = ['high','fng','rsi','macd','macd_signal','macd_hist','cci','dx','rf','sar','adx','adxr','apo','aroonosc','bop','minus_di','minus_dm','mom','plus_di','plus_dm','ppo_ta','roc','rocp','trix','ultosc','willr','ht_dcphase','ht_sine','ht_trendmode','feature_PvEWMA_4','feature_PvCHLR_4','feature_RvRHLR_4','feature_CON_4','feature_RACORR_4', 'feature_PvEWMA_8', 'feature_PvCHLR_8', 'feature_RvRHLR_8', 'feature_CON_8', 'feature_RACORR_8','feature_PvEWMA_16','feature_PvCHLR_16','feature_RvRHLR_16','feature_CON_16','feature_RACORR_16','feature_PvEWMA_32','feature_PvCHLR_32','feature_RvRHLR_32', 'feature_CON_32', 'feature_RACORR_32', 'feature_PvEWMA_64', 'feature_PvCHLR_64', 'feature_RvRHLR_64', 'feature_CON_64', 'feature_RACORR_64', 'feature_PvEWMA_128', 'feature_PvCHLR_128', 'feature_RvRHLR_128', 'feature_CON_128', 'feature_RACORR_128', 'feature_PvEWMA_256', 'feature_PvCHLR_256', 'feature_RvRHLR_256', 'feature_CON_256', 'feature_RACORR_256']
period_lengths = [4, 8, 16, 32, 64, 128, 256]
min_max_scaler = MinMaxScaler()

class Actions(Enum):
    Sell = 0
    Buy = 1
    Do_nothing = 2

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

    df['rsi'] = RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['cci'] = CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['dx'] = DX(df['high'], df['low'], df['close'], timeperiod=14)

    #OTHERS I ADDED
    #Overlap studies
    df["rf"] = df["close"].pct_change().shift(-1)
    df['sar'] = ta.SAR(df['high'], df['low'], acceleration=0., maximum=0.)

    # Added momentum indicators
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'])
    df['adxr'] = ta.ADXR(df['high'], df['low'], df['close'])
    df['apo'] = ta.APO(df['close'])
    df['aroonosc'] = ta.AROONOSC(df['high'], df['low'])
    df['bop'] = ta.BOP(df['open'], df['high'], df['low'], df['close'])
    df['cmo'] = ta.CMO(df['close'])
    df['minus_di'] = ta.MINUS_DI(df['high'], df['low'], df['close'])
    df['minus_dm'] = ta.MINUS_DM(df['high'], df['low'])
    df['mom'] = ta.MOM(df['close'])
    df['plus_di'] = ta.PLUS_DI(df['high'], df['low'], df['close'])
    df['plus_dm'] = ta.PLUS_DM(df['high'], df['low'])
    df['ppo_ta'] = ta.PPO(df['close'])
    df['roc'] = ta.ROC(df['close'])
    df['rocp'] = ta.ROCP(df['close'])
    df['trix'] = ta.TRIX(df['close'])
    df['ultosc'] = ta.ULTOSC(df['high'], df['low'], df['close'])
    df['willr'] = ta.WILLR(df['high'], df['low'], df['close'])

    # Cycle indicator functions
    #df['ht_dcperiod'] = ta.HT_DCPERIOD(df['close'])
    df['ht_dcphase'] = ta.HT_DCPHASE(df['close'])
    df['ht_sine'], _ = ta.HT_SINE(df['close'])
    df['ht_trendmode'] = ta.HT_TRENDMODE(df['close'])

    # for each period length
    for period_length in period_lengths:
        # add the feature columns to the bars df
        add_feature_columns(df, period_length)

    df.drop('open', inplace=True, axis=1)
    df.drop('low', inplace=True, axis=1)
    df.drop('close', inplace=True, axis=1)

    df = df.dropna()

    df[df.columns] = min_max_scaler.fit_transform(df[df.columns]) 

    return df

model = PPO.load("ppo_crypto")

class tradingExecution:

    def __init__(self):
        self.df = pd.DataFrame()
        self.window_size = 30
        self.selected_feature_name = INDICATORS

        self.current_fng = None
        self._position = 0

        self.candle = {}
        self.running_volume = 0
        self.running_FnG = 0
        self.running_open = 0
        self.running_high, self.running_low = 0, math.inf
        self.dollar_threshold = 10000#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1000000

        self.signal_features = []

    def addFnG(self):
        #add FNG index
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.request("GET", url)
        dataRes = response.json()
        dfFnG = pd.json_normalize(dataRes['data'])
        self.current_fng = dfFnG

    def tick(self, candle):
        if self.candle == {}:
            self.addFnG()

        elif candle["startedAt"] == self.candle["startedAt"]:
            pass

        else:
            print("new minute")
            next_open = float(self.candle["open"])
            next_high = float(self.candle["high"])
            next_low = float(self.candle["low"])
            next_close = float(self.candle["close"])
            next_volume = float(self.candle["baseTokenVolume"])
            
            date = datetime.datetime.utcnow()
            utc_time = calendar.timegm(date.utctimetuple())

            if utc_time - int(self.current_fng["timestamp"][0]) <  300:
                print("updating fng")
                self.addFnG()

            next_FnG = int(self.current_fng['value'][0])

            # get the midpoint price of the next bar (the average of the open and the close)
            midpoint_price = (next_open + next_close)/2

            # get the approximate dollar volume of the bar using the volume and the midpoint price
            dollar_volume = next_volume * midpoint_price

            self.running_high, self.running_low = max(self.running_high, next_high), min(self.running_low, next_low)

            if self.running_open == 0:
                self.running_open = next_open

            if self.running_FnG == 0:
                self.running_FnG = next_FnG
            else:
                self.running_FnG = (self.running_FnG + next_FnG)/2

            # if the next bar's dollar volume would take us over the threshold...
            if dollar_volume + self.running_volume >= self.dollar_threshold:
                # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
                dollar_bar = [{'high': self.running_high, 'low': self.running_low, 'open': self.running_open, 'close': next_close, 'fng': self.running_FnG}]

                if len(self.df) < 286:
                    self.df = pd.concat([pd.DataFrame(dollar_bar), self.df], ignore_index=True)
                else:
                    self.df = pd.concat([pd.DataFrame(dollar_bar), self.df], ignore_index=True)
                    self.df = self.df.iloc[:-1]

                    df_p = addIndicators(self.df)
                    action = self.predict(df_p)
                    
                    self.execute(action)

                self.reset()
                print(self.df)
            else:
                self.running_volume += dollar_volume
            
        self.candle = candle

    def predict(self, df_p):
        obs = df_p[:self.window_size+1]
        action, _states = model.predict(obs)
        return action

    def buy(self):
        print("buying now")
        date = datetime.datetime.utcnow()
        utc_time = calendar.timegm(date.utctimetuple())

        buy_order = testnet_client.private.create_order(position_id=1, market=MARKET_ETH_USD, side=ORDER_SIDE_BUY, order_type=ORDER_TYPE_MARKET, post_only=False, size="0.25", price="1", limit_fee="0.015", expiration_epoch_seconds=utc_time+120, time_in_force=TIME_IN_FORCE_FOK)
        print(buy_order.data)
        return buy_order.data

    def sell(self):
        print("buying now")
        date = datetime.datetime.utcnow()
        utc_time = calendar.timegm(date.utctimetuple())

        sell_order = testnet_client.private.create_order(position_id=0, market=MARKET_ETH_USD, side=ORDER_SIDE_SELL, order_type=ORDER_TYPE_MARKET, post_only=False, size="0.25", price="1", limit_fee="0.015", expiration_epoch_seconds=utc_time+120, time_in_force=TIME_IN_FORCE_FOK)
        print(sell_order.data)
        return sell_order.data

    def execute(self, action):
        print("executing trade")

        if action == Actions.Buy.value:
            if self._position == 0:
                buy_attempt = self.buy()
                if buy_attempt == "FILLED":
                    self._position = 1

        elif action == Actions.Sell.value:
            if self._position == 1:
                sell_attempt = self.sell()
                if sell_attempt == "FILLED":
                    self._position = 0

        elif action == Actions.Do_nothing.value:
            print("did nothing")

    def reset(self):
        self.running_volume = 0
        self.running_FnG = 0
        self.running_open = 0
        self.running_high, self.running_low = 0, math.inf

execution_env = tradingExecution()

execution_env.buy()

#testing end, real code below
#starttime = time.time()
#while True:
#    candle = client.public.get_candles(market="ETH-USD", resolution="1MIN", limit=1)
#    execution_env.tick(candle.data["candles"][0])
#    time.sleep(0.25 - ((time.time() - starttime) % 0.25))
