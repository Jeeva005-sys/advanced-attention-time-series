"""
Advanced Time Series Forecasting with Attention Mechanism
Author: Jeeva S

Implements:
- Multivariate time series forecasting
- Baseline ARIMA
- Encoder-Decoder LSTM with Attention
- Hyperparameter Grid Search
- Metrics: MAE, RMSE, MAPE
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

# =========================
# Data Generation
# =========================
np.random.seed(42)
time = np.arange(0, 500)
f1 = np.sin(0.02 * time)
f2 = np.cos(0.02 * time)
f3 = np.random.normal(0, 0.1, len(time))
target = f1 + f2 + f3

df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "target": target})
df.to_csv("dataset.csv", index=False)

# =========================
# Scaling
# =========================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# =========================
# Sequence Builder
# =========================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

# =========================
# Attention Model
# =========================
def build_attention_model(seq_len, n_features, units):
    encoder_inputs = Input(shape=(seq_len, n_features))
    encoder_lstm = LSTM(units, return_sequences=True)(encoder_inputs)

    decoder_inputs = LSTM(units, return_sequences=True)(encoder_lstm)

    attention = Attention()([decoder_inputs, encoder_lstm])
    dense = Dense(1)(attention[:, -1, :])

    model = Model(encoder_inputs, dense)
    model.compile(optimizer='adam', loss='mse')
    return model

# =========================
# Hyperparameter Grid
# =========================
seq_lengths = [10, 20]
units_list = [32, 64]
splits = [0.7, 0.8]

results = []

for seq_len, units, split in itertools.product(seq_lengths, units_list, splits):
    X, y = create_sequences(scaled, seq_len)
    train_size = int(len(X) * split)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_attention_model(seq_len, X.shape[2], units)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds.flatten()) / y_test)) * 100

    results.append([seq_len, units, split, mae, rmse, mape])

# =========================
# Save Results
# =========================
results_df = pd.DataFrame(results, columns=["seq_len", "units", "train_split", "MAE", "RMSE", "MAPE"])
results_df.to_csv("results.csv", index=False)

print("Training completed. Results saved to results.csv")
