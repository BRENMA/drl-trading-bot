#from Dataset import Dataset
#from Model import Encoder, ReplayMemory, DuellingDQN, AttentionLayer, DQNAgent, Decoder
from Config import *
from CoinbaseAPI import CoinbaseAPI
#from TradingEnv import SingleAssetTradingEnvironment
#from tqdm import trange
import numpy as np
import argparse
import time

if __name__ == '__main__':

    print("\n\n\n'88,dPYba,,adPYba,   ,adPPYba,  8b,dPPYba,   ,adPPYba, 8b       d8  \n","88P'   '88'    '8a a8'     '8a 88P'   `'8a a8P_____88 `8b     d8'  \n","88      88      88 8b       d8 88       88 8PP'''''''  `8b   d8'   \n","88      88      88 '8a,   ,a8' 88       88 '8b,   ,aa   `8b,d8'    \n","88      88      88  `'YbbdP''  88       88  `'Ybbd8''     Y88'     \n","                                                          d8'      \n","                                                         d8'       \n")
    print("> welcome to moneyDRL")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    
    if args.data:
        coinbaseAPI = CoinbaseAPI()
        historic_data = coinbaseAPI.getCoinHistoricData(COIN_PAIRS, end = END_DATE, granularity = GRANULARITY)



# merging two csv files
#df = pd.concat(map(pd.read_csv, ['datasets/ADA-USD.csv']), ignore_index=True)#, 'datasets/ATOM-USD.csv', 'datasets/AVAX-USD.csv', 'datasets/BTC-USD.csv', 'datasets/ETH-USD.csv', 'datasets/LINK-USD.csv', 'datasets/MATIC-USD.csv', 'datasets/SOL-USD.csv', 'datasets/XRP-USD.csv']), ignore_index=True)
#df = df.dropna(axis='columns')
#print(len(df))
#print("> Creating X,Y sets")
#
#df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
#print(df)