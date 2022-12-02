import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import *
import time
from sklearn.preprocessing import MinMaxScaler

class Dataset:
    def loadCoinData(self):
        print("> Creating Training Data for coin pairs")
        
        # merging two csv files
        df = pd.concat(map(pd.read_csv, ['cryptoTrading/datasets/ADA-USD.csv']), ignore_index=True)#, 'cryptoTrading/datasets/ATOM-USD.csv', 'cryptoTrading/datasets/AVAX-USD.csv', 'cryptoTrading/datasets/BTC-USD.csv', 'cryptoTrading/datasets/ETH-USD.csv', 'cryptoTrading/datasets/LINK-USD.csv', 'cryptoTrading/datasets/MATIC-USD.csv', 'cryptoTrading/datasets/SOL-USD.csv', 'cryptoTrading/datasets/XRP-USD.csv']), ignore_index=True)
        df = df.dropna(axis='columns')
        print(df)

        return df

    def DataSequential(self, data):
        #n_samples x timesteps x n_features
        print(len(data))
        output_x = []
        output_y = []
        for i in range(0, len(data) - (window_size + 1)):
            print(len(data) - window_size - i)
            hour_data = data.iloc[i: (i + (window_size + 1)), :]
            if hour_data['time'].iloc[-1] - hour_data['time'].iloc[0] == 3600:
                del hour_data['time']
                t = []
                output_y.append(hour_data.iloc[-1].values.tolist())
                for j in range(0, window_size):
                    hrData = hour_data.iloc[j].values.tolist()
                    t.append(hrData)
                output_x.append(t)
        
        output_x = np.array(output_x)
        output_x = output_x.reshape(output_x.shape[0], window_size, data_columns)
        return output_x, output_y

    def createTrainTestSets(self, data):
        print("> Creating X,Y sets")
        min_max_scaler = MinMaxScaler()
        for column in data:
            columnSeriesObj = data[column]
            if column != 'time':
                data[column] = min_max_scaler.fit_transform(columnSeriesObj.values.reshape(-1, 1))

        print(data)
        testSize = 99.8 #%
        trainData = data.head(int(len(data)*(1 - testSize/100)))
        testData = data.tail(int(len(data)*(testSize/100)))
        
        dataTrain_x, dataTrain_y = self.DataSequential(trainData)
        #dataTest_x, dataTest_y = self.DataSequential(testData)
        
        #X_train_torch = torch.from_numpy(dataTrain_x).type(torch.Tensor)
        #Y_train_torch = torch.from_numpy(np.array(dataTrain_y)).type(torch.Tensor)

        return dataTrain_x, dataTrain_y