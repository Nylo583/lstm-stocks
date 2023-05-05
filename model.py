import math
import yfinance as yfin
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyp
import tensorflow as tf
from tensorflow import keras
from keras import layers
import sklearn.preprocessing as skl

data = yfin.Ticker("AAPL")
data.info
data_hist = data.history(period="10y", interval="1d")

#pyp.plot(data_hist["Close"])
#pyp.show()

full_vals = data_hist["Close"].values
train_len = math.floor(len(full_vals))

mms = skl.MinMaxScaler((0, 1))
vals = mms.fit_transform(full_vals.reshape(-1, 1))[0:train_len, :]

trainX, trainY = [], []
for i in range(90, len(vals)):
    trainX.append(vals[i-90:i, 0])
    trainY.append(vals[i, 0])

trainX, trainY = np.array(trainX), np.array(trainY)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# testing_vals = mms.fit_transform(full_vals.reshape(-1, 1))[train_len-180:, :]
# testX, testY = [], [0]

# for i in range(90, len(testing_vals)):
#     testX.append(testing_vals[i-90:i, 0])

# testX, testY = np.array(testX), np.array(testY)
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = tf.keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX, trainY, batch_size=2, epochs=1)

# predictions = model.predict(testX)
# predictions = mms.inverse_transform(predictions)
# rmse = np.sqrt(np.mean((predictions - testY)**2))
# print(rmse)

def predict_future(model, days):
    predictions = []
    for d in range(days):
        listNums = []
        for a in range(days-d):
            listNums.append(vals[len(full_vals)-days+d+a])
        for b in range(len(predictions)):
            listNums.append(predictions[b])
        listNums = np.array(listNums)
        listNums = np.reshape(listNums, (-1, 1))
        x = []
        for c in range(days):
            x.append(listNums)
        x = np.asarray(x).astype('float32')
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        predictions.append(np.float32(model.predict(x)[0][0]))
    predictions = np.array(predictions)
    predictions = np.reshape(predictions, (-1, 1))
    predictions = mms.inverse_transform(predictions)
    print(predictions)
    return predictions
future = predict_future(model, 90)



trainRange = full_vals[:train_len]
testRange = full_vals[train_len:]
pyp.figure(figsize=(16,8))
pyp.title('Model')
pyp.xlabel('Date')
pyp.ylabel('Close Price USD ($)')
pyp.plot(trainRange)
pyp.plot(future)
# pyp.legend(['Train', 'Val', 'Predictions'], loc='lower right')
pyp.show()