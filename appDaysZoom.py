#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
import pandas_datareader as web
plt.style.use('fivethirtyeight')
import streamlit as st

import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns
#from sklearn.model_selection import train_test_split
from tensorflow import keras

#gráfica interactiva
import plotly.graph_objects as go
import plotly.express as px



st.title('CRYPTRACKS Trend Prediction')
user_input_ticket = st.text_input('Enter Stock Ticket', 'BTC-USD')
user_input_StartDate = st.text_input('Enter start date', '2021-01-01')
#user_input_EndDate = st.text_input('Enter end date', '2022-01-01')



df = yf.download(user_input_ticket, user_input_StartDate) #ADA-USD DOT1-USD VET-USD CRO-USD BTC-USD MANA-USD ETH-USD QNT-USD
df = df.drop(columns=['Adj Close'])

#Separate dates for future plotting
train_dates = pd.to_datetime(df.index)
train_dates #Check last few dates. 
#Variables for training
cols = list(df)[0:5]
#New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)
#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

#describe the data
st.subheader('Data from')
st.write(df.describe())




#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 
#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    


trainX, trainY = np.array(trainX), np.array(trainY)

lr= 0.02 #lr= 0.01 25 epochs got 75% validation accuracy and 89% fonal accuracy
            #0.03 25 epochs got 35% validation accuracy and 36% final accuracy


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

#model.compile(optimizer='adam', loss='mse') 
model.compile(loss='mse', optimizer = keras.optimizers.Adam(lr),metrics = ['accuracy']) 



history = model.fit(trainX, trainY, epochs=60, batch_size=20, validation_split=0.1, verbose=1)
n_days_for_prediction=15
predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_days_for_prediction, freq='1d').tolist()



prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]



# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())




df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


original = df[['Open']]
original.index=pd.to_datetime(original.index)
original = original.loc[original.index >= user_input_StartDate]


diferencia = original['Open'][len(original)-1] - df_forecast['Open'][0]
#quitamos la primera fila del forecast porque es la misma que la ultima del dataset original

df_forecast_ajustado = df_forecast

for i in range(len(df_forecast)):
    df_forecast_ajustado['Open'][i] = (df_forecast['Open'][i]) + (diferencia)






#pintamos la gráfica
st.subheader('prediccion')
fig = plt.figure(figsize=(12.2,4.5))
plt.plot(original.index, original['Open'], label = 'Actual price', color= 'blue')
plt.plot(df_forecast['Date'], df_forecast['Open'], label = 'Predicted', color= 'red')
plt.xticks(rotation=90)
st.pyplot(fig)



user_input_StartDatePred = st.text_input('Enter start date', '2022-01-01')

original = df[['Open']]
original.index=pd.to_datetime(original.index)
original = original.loc[original.index >= user_input_StartDatePred]

#pintamos la gráfica
st.subheader('prediccion user zoom')
fig = plt.figure(figsize=(12.2,4.5))
plt.plot(original.index, original['Open'], label = 'Actual price', color= 'blue')
plt.plot(df_forecast['Date'], df_forecast['Open'], label = 'Predicted', color= 'red')
plt.xticks(rotation=90)
st.pyplot(fig)

#gráfica interactiva
st.subheader('prediccion interactiva')
fig = go.Figure()
fig.add_trace(go.Scatter(x=original.index, y=original['Open'],
                    mode='lines',
                    name='Price'))
fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast_ajustado['Open'],
                    mode='lines',
                    name='Predicted'))

fig.show()
#st.pyplot(fig)