# http://www.python36.com/predict-cryptocurrency-price-using-lstm/

# LSTM for closing bitcoin price with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# convert an array of values into a dataset matrix
def create_dataset(dataset):
  dataX, dataY = [], []
  for i in range(len(dataset)-1):
    dataX.append(dataset[i])
    dataY.append(dataset[i + 1])
  return np.asarray(dataX), np.asarray(dataY)
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
df = read_csv('../data/bitcoin.csv')
df = df.iloc[::-1]
df = df.drop(['Date','Open','High','Low','Volume','MarketCap'], axis=1)
dataset = df.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#prepare the X and Y label
X,y = create_dataset(dataset)
#Take 80% of data as the training sample and 20% as testing sample
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.50, shuffle=False)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='logcosh', optimizer='adadelta')
#model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2)
#save model for later use
#model.save('./savedModel')
#load_model
model = load_model('./savedModel')
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
futurePredict = model.predict(np.asarray([[testPredict[-1]]]))
futurePredict = scaler.inverse_transform(futurePredict)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
print("Jer debut")
print(scaler.inverse_transform(testX[-1]))
print(scaler.inverse_transform(testX[-2]))
print(scaler.inverse_transform(testX[-3]))
print(scaler.inverse_transform(testX[-4]))
print(scaler.inverse_transform(testX[-5]))

for i in range(1,len(testPredict)):
    if testX[-i] < testX[-i-1]: res = "A"
    else: res = "B"
    if testPredict[-i] < testPredict[-i-1]: res2 = "A"
    else: res2 = "B"
    if res == res2:
        if res == "A": print(" en BAISSE (" + repr(i) + ") " + repr(scaler.inverse_transform(testX[-i])) + "  " + repr(testPredict[-i]) )
        else: print(" en HAUSSE (" + repr(i) + ") " + repr(scaler.inverse_transform(testX[-i])) + "  " + repr(testPredict[-i]) )
    else: print(" en décalage")



print("Jer fin")

print("Price for last 5 days: ")
print(testPredict[-5:])
print("Bitcoin price for tomorrow: ", futurePredict)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict):len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
#plt.show()
