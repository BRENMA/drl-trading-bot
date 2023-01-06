import torch

TICKER_LIST = ["ETHUSDT"]#, "ATOMUSDT", "ADAUSDT", "BTCUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT"]
INDICATORS = ['high','low','open','close','fng','rsi','macd','macd_signal','macd_hist','cci','dx','rf','sar','adx','adxr','apo','aroonosc','bop','cmo','minus_di','minus_dm','mom','plus_di','plus_dm','ppo_ta','roc','rocp','rocr','rocr100','trix','ultosc','willr','ht_dcphase','ht_sine','ht_trendmode','feature_PvEWMA_4','feature_PvCHLR_4','feature_RvRHLR_4','feature_CON_4','feature_RACORR_4','feature_PvEWMA_8','feature_PvCHLR_8','feature_RvRHLR_8','feature_CON_8','feature_RACORR_8','feature_PvEWMA_16','feature_PvCHLR_16','feature_RvRHLR_16','feature_CON_16','feature_RACORR_16','feature_PvEWMA_32','feature_PvCHLR_32','feature_RvRHLR_32','feature_CON_32','feature_RACORR_32','feature_PvEWMA_64','feature_PvCHLR_64','feature_RvRHLR_64','feature_CON_64','feature_RACORR_64','feature_PvEWMA_128','feature_PvCHLR_128','feature_RvRHLR_128','feature_CON_128','feature_RACORR_128','feature_PvEWMA_256','feature_PvCHLR_256','feature_RvRHLR_256','feature_CON_256','feature_RACORR_256']
period_lengths = [4, 8, 16, 32, 64, 128, 256]

TIME_INTERVAL = '1m'
TRAIN_START_DATE = '2022-06-01'
TRAIN_END_DATE= '2023-01-01'

TEST_START_DATE = '2022-12-20'
TEST_END_DATE = '2023-01-01'

# To make the Agent more risk averse towards negative returns. Negative reward multiplier
NEG_MUL = 0

STATE_SPACE = 43
ACTION_SPACE = 2

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
