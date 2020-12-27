
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.patches as mpatches
import random

df=pd.read_excel("Türkiye-time_series_covid19_confirmed_global-20082020.xlsx",index=["Country/Region","Date","Cases","Deaths","Recovered"])
# Düz Veriler
print(df.head())

raw_data= df.Cases.values

print(raw_data[:51])
dataDate=pd.to_datetime(df["Date"],format='%d-%m-%y')
dataCases=df["Cases"].astype('float')
print(type(dataCases),dataCases)
dataDeaths=df["Deaths"]
dataRecovered=df["Recovered"]
datas=[dataCases,dataDeaths,dataRecovered]
#cnt_transformer=RobustScaler()
#cnt_transformer=cnt_transformer.fit(dataCases.values.reshape(-1,1))
def lstm_Func(dataCases):
    datacases=dataCases
    scaler=MinMaxScaler(feature_range=(0,1))
    dataCases=scaler.fit_transform(dataCases.values.reshape(-1, 1))
    print(len(dataCases))

    timestep = 5
    X = []
    Y = []
    for i in range(len(dataCases)-(timestep)):
        X.append(dataCases[i:i+timestep])
        Y.append(dataCases[i+timestep])

    X = np.asanyarray(X)
    X = X.reshape((X.shape[0],X.shape[1],1))

    Y = np.asanyarray(Y)

    print(len(Y))

    k = int(0.9*len(Y))
    Xtrain = X[:k,:,:]
    Xtest = X[k:,:,:]

    Ytrain = Y[:k]
    Ytest = Y[k:]

    model = Sequential()
    model.add(LSTM(64,batch_input_shape=(None,timestep,1),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())

    model.fit(Xtrain,
              Ytrain,
              validation_data=(Xtest,Ytest),
              verbose=1,
              epochs=20,
              shuffle=False)

    Ypred = model.predict(Xtrain)
    print(Ypred)

    Ypred = scaler.inverse_transform(Ypred)
    Ypred = Ypred[:, 0]
    print(len(Ypred))
    Yreel =datacases[:k]
    print(len(Yreel),Yreel)
    plt.plot(Yreel, label='Yreel', color='blue')
    plt.plot(Ypred, label='Ypred', color='red')

    blue_patch = mpatches.Patch(color='blue', label='Yreel')
    red_patch = mpatches.Patch(color='red', label='Ypred')
    plt.legend(handles=[blue_patch, red_patch])
    plt.title('RMSE: %.4f' % np.sqrt(sum((Ypred - Yreel) ** 2) / len(Yreel)))
    plt.show()


   
for i in datas:
    lstm_Func(i)