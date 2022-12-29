import cbpro
import json
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import talib as ta
import requests
import matplotlib.pyplot as plt

class CoinbaseAPI:

    def __init__(self):
        print("Coinbase API Initiated")
        self.public_client = cbpro.PublicClient()

    def getCoinHistoricData(self, coin_pairs, end, granularity):
        for coin_pair in coin_pairs:
            print("> Collecting historic data for "+ coin_pair[0]," from ", coin_pair[1]," to ", end ," every ",granularity," sec")

            url = "https://api.alternative.me/fng/?limit=0"
            response = requests.request("GET", url)
            dataRes = response.json()
            dfFnG = pd.json_normalize(dataRes['data'])
            del dfFnG['value_classification']
            del dfFnG['time_until_update']
            
            dfFnG = dfFnG.iloc[::-1]

            start_date = datetime.datetime.strptime(coin_pair[1], "%Y-%m-%d")
            end_date = datetime.datetime.strptime(end, "%Y-%m-%d")

            dataList = []
            while (start_date <= end_date):
                print("> Date: ", start_date)

                start_limit = start_date
                end_limit = start_date + datetime.timedelta(hours=1)

                try:
                    next_data = self.public_client.get_product_historic_rates(coin_pair[0], granularity=granularity,start=start_limit,end=end_limit)

                    for data in reversed(next_data):
                        if len(dataList) > 0:
                            last_elem = dataList[-1:][0]
                            if last_elem[0] != data[0]:
                                dataList.append(data)
                        else:
                            dataList.append(data)

                    start_date += datetime.timedelta(hours=1)
                except KeyboardInterrupt:
                    return
                except:
                    start_date += datetime.timedelta(hours=1)

            df = pd.DataFrame(dataList, columns=["time", "low", "high", "open", "close", "volume"])

            FnGArr = list(dfFnG.timestamp)
            target = df.iloc[0:1500]["time"]
            FnGStartPoint = 0

            for n in range(len(target)):
                if FnGStartPoint == 0:
                    for i in range(len(FnGArr)):
                        if (int(FnGArr[i]) == int(target[n])):
                            FnGStartPoint = i
                else:
                    print("start point found")

            DFStartIndex = df[df['time'] == int(FnGArr[FnGStartPoint])].index[0]
            df = df.iloc[DFStartIndex:]

            FnGIndArr = []
            for i in range(len(df)):
                if int(df.iloc[i]['time']) >= int(dfFnG.iloc[FnGStartPoint + 1]['timestamp']):
                    FnGStartPoint += 1
                
                FnGIndArr.append(int(dfFnG.iloc[FnGStartPoint]['value']))

            df.insert(0, "fngindex", FnGIndArr, True)

            df.index = df['time']
            del df['time']
            df = df.dropna()
            df.apply(pd.to_numeric, errors="coerce")
            df["rf"] = df["close"].pct_change().shift(-1)
            df = df.dropna();

            #add_technical_indicator
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            df['dx'] = ta.DX(df['high'], df['low'], df['close'], timeperiod=14)

            # Overlap studies
            df['sar'] = ta.SAR(df['high'], df['low'], acceleration=0., maximum=0.)

            # Added momentum indicators
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'])
            df['adxr'] = ta.ADXR(df['high'], df['low'], df['close'])
            df['apo'] = ta.APO(df['close'])
            df['aroonosc'] = ta.AROONOSC(df['high'], df['low'])
            df['bop'] = ta.BOP(df['open'], df['high'], df['low'], df['close'])
            df['cmo'] = ta.CMO(df['close'])
            df['mfi'] = ta.MFI(df['high'], df['low'], df['close'],df['volume'])
            df['minus_di'] = ta.MINUS_DI(df['high'], df['low'], df['close'])
            df['minus_dm'] = ta.MINUS_DM(df['high'], df['low'])
            df['mom'] = ta.MOM(df['close'])
            df['plus_di'] = ta.PLUS_DI(df['high'], df['low'], df['close'])
            df['plus_dm'] = ta.PLUS_DM(df['high'], df['low'])
            df['ppo_ta'] = ta.PPO(df['close'])
            df['roc'] = ta.ROC(df['close'])
            df['rocp'] = ta.ROCP(df['close'])
            df['rocr'] = ta.ROCR(df['close'])
            df['rocr100'] = ta.ROCR100(df['close'])
            df['trix'] = ta.TRIX(df['close'])
            df['ultosc'] = ta.ULTOSC(df['high'], df['low'], df['close'])
            df['willr'] = ta.WILLR(df['high'], df['low'], df['close'])

            # Volatility Indicators
            df['ad'] = ta.AD(df['high'], df['low'], df['close'],df['volume'])
            df['adosc'] = ta.ADOSC(df['high'], df['low'], df['close'],df['volume'], fastperiod=3, slowperiod=10)
            df['obv'] = ta.OBV(df['close'],df['volume'])

            # Cycle indicator functions
            df['roc'] = ta.HT_DCPERIOD(df['close'])
            df['ht_dcphase'] = ta.HT_DCPHASE(df['close'])
            df['ht_sine'], _ = ta.HT_SINE(df['close'])
            df['ht_trendmode'] = ta.HT_TRENDMODE(df['close'])

            df = df.dropna();

            #corr_matrix = pd.DataFrame(df).corr()
            #ax = sns.heatmap(
            #    corr_matrix, 
            #    xticklabels=True, 
            #    yticklabels=True,
            #    vmin=-1, vmax=1, center=0,
            #    cmap=sns.diverging_palette(20, 220, n=200),
            #    square=True
            #)
            #ax.set_xticklabels(
            #    ax.get_xticklabels(),
            #    rotation=90,
            #    horizontalalignment='right'
            #);
            #plt.show()

            corr_matrix = pd.DataFrame(df).corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool));

            # So we are selecting the columns which are having absolute correlation greater than 0.95 and making a list of those columns named 'dropping_these_features'.
            dropping_these_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            print(dropping_these_features)
            for droppingFeature in dropping_these_features:
                if str(droppingFeature) != 'open' and str(droppingFeature) != 'high' and str(droppingFeature) != 'low' and str(droppingFeature) != 'close' and str(droppingFeature) != 'volume':
                    del df[str(droppingFeature)]

            # [ time, low, high, open, close, volume, rf, rsi, macd, cci, dx, sar, adx, adxr, apo, aroonosc, bop, cmo, mfi, minus_di, minus_dm, mom, plus_di, plus_dm, ppo_ta, roc, rocp, rocr, rocr100, trix, ultosc, willr, ad, adosc, obv, roc, ht_dcphase, ht_sine, ht_trendmode]
            # x = df.iloc[:,0:37]
            # df[df.columns[0:37]] = (x-x.min())/ (x.max() - x.min())

            print(df)
            df.to_csv(rf'cryptoTrading/datasets/{coin_pair[0]}.csv')
            print("> Storing Raw Historic Data for", coin_pair[0])
            #df.to_csv(rf'cryptoTrading/datasets/{coin_pair[0]}.csv')