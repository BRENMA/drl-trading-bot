from Dataset import Dataset
from Model import Encoder, ReplayMemory, DuellingDQN, AttentionLayer, DQNAgent, Decoder
from Config import *
from CoinbaseAPI import CoinbaseAPI
from TradingEnv import SingleAssetTradingEnvironment
from tqdm import trange
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

    elif args.train:
        print("> Creating Testing Data")
        dataset = Dataset()
        data = dataset.loadCoinData()
        dataTrain_x, dataTrain_y = dataset.createTrainTestSets(data)

        print("> Creating Models")
        memory = ReplayMemory()
        agent = DQNAgent(actor_net = DuellingDQN, memory=memory)

        scores = []
        act_dict = {0:-1, 1:1, 2:0}

        encoder = Encoder(dataTrain_x.shape[2], hidden_size)
        decoder = Decoder(dataTrain_x.shape[2], hidden_size)
        env = SingleAssetTradingEnvironment()

        te_score_min = -np.Inf
        with trange(N_EPISODES) as ne:
            for episode in range(1, 1 + ne):
                score = 0
                state = env.reset()
                state = state.reshape(-1, STATE_SPACE)

                for window in dataTrain_x:
                    while True:
                        actions = agent.act(state, EPS_START)
                        action = act_dict[actions]
                        next_state, reward, done, _ = env.step(action)
                        next_state = next_state.reshape(-1, STATE_SPACE)
                        print(state)

