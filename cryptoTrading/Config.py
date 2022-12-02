import torch

DATASET_DIR = "datasets/"
COIN_PAIRS = [["BTC-USD", "2018-04-01"], ["ETH-USD", "2018-04-01"], ["SOL-USD", "2020-05-01"], ["AVAX-USD", "2020-10-01"], ["ATOM-USD", "2019-06-01"], ["MATIC-USD", "2021-02-01"], ["ADA-USD", "2018-04-01"], ["XRP-USD", "2018-04-01"], ["LINK-USD", "2018-04-01"]]
GRANULARITY = 60 #3600 # Data every 1 hour
END_DATE = "2022-10-01" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

window_size = 60
data_columns = 34

initial_investment = 1000
transaction_cost = 0.0

N_EPISODES = 20

hidden_size = 15

STATE_SPACE = 34
ACTION_SPACE = 3

ACTION_LOW = -1
ACTION_HIGH = 1

GAMMA = 0.9995
TAU = 1e-3
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.9

MEMORY_LEN = 10000
MEMORY_THRESH = 500
BATCH_SIZE = 200

LR_DQN = 5e-4

LEARN_AFTER = MEMORY_THRESH
LEARN_EVERY = 3
UPDATE_EVERY = 9

COST = 3e-4
CAPITAL = 100000
NEG_MUL = 2