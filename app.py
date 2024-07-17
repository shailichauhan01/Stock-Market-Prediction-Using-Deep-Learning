import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Load the pre-trained Keras model
model = load_model(r'C:\Users\Shaili Chauhan\Downloads\Stock_Market_Prediction_ML\Stock Predictions Model.keras')

# Streamlit header and input
st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Slicing the data for training and testing
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Feature scaling with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plotting and displaying figures with st.pyplot
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(ma_50_days, 'r', label='MA50')
ax1.plot(data.Close, 'g', label='Close Price')
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(ma_50_days, 'r', label='MA50')
ax2.plot(ma_100_days, 'b', label='MA100')
ax2.plot(data.Close, 'g', label='Close Price')
ax2.set_xlabel('Time')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(ma_100_days, 'r', label='MA100')
ax3.plot(ma_200_days, 'b', label='MA200')
ax3.plot(data.Close, 'g', label='Close Price')
ax3.set_xlabel('Time')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y_true = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y_true.append(data_test_scale[i, 0])

x = np.array(x)
y_true = np.array(y_true)

# Make predictions using the loaded model
predictions = model.predict(x)

# Invert scaling to get actual predicted prices
scale = 1 / scaler.scale_
predictions = predictions * scale
y_true = y_true * scale

# Plot original vs predicted prices
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(predictions, 'r', label='Predicted Price')
ax4.plot(y_true, 'g', label='Original Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)
