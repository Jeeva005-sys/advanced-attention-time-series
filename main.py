import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import itertools
import os

# ======================
# 1. DATA GENERATION (5000 samples, multivariate)
# ======================
np.random.seed(42)

n_samples = 5000
time = np.arange(n_samples)

f1 = np.sin(0.01 * time)
f2 = np.cos(0.015 * time)
f3 = np.sin(0.02 * time) + np.random.normal(0, 0.1, n_samples)

target = f1 + f2 + f3

data = pd.DataFrame({
    "f1": f1,
    "f2": f2,
    "f3": f3,
    "target": target
})

data.to_csv("dataset.csv", index=False)

# ======================
# 2. PREPROCESSING
# ======================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])
        y.append(data[i+seq_len, -1])
    return np.array(X), np.array(y)

# ======================
# 3. MODELS
# ======================
def build_baseline_lstm(seq_len, n_features, units, lr):
    model = Sequential([
        LSTM(units, input_shape=(seq_len, n_features)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

def build_attention_model(seq_len, n_features, units, lr):
    inputs = Input(shape=(seq_len, n_features))
    encoder = LSTM(units, return_sequences=True)(inputs)
    decoder = LSTM(units, return_sequences=True)(encoder)
    attention = Attention()([decoder, encoder])
    output = Dense(1)(attention[:, -1, :])
    model = Model(inputs, output)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

# ======================
# 4. METRICS
# ======================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# ======================
# 5. HYPERPARAMETER GRID
# ======================
seq_lengths = [20, 30]
units_list = [32, 64]
learning_rates = [0.001, 0.0005]
splits = [0.7, 0.8]

results = []

# ======================
# 6. GRID SEARCH
# ======================
for seq_len, units, lr, split in itertools.product(seq_lengths, units_list, learning_rates, splits):
    print(f"\nRunning experiment: seq_len={seq_len}, units={units}, lr={lr}, split={split}")

    X, y = create_sequences(scaled, seq_len)
    split_idx = int(len(X) * split)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    n_features = X.shape[2]

    # ---- Baseline LSTM ----
    baseline = build_baseline_lstm(seq_len, n_features, units, lr)
    baseline.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    y_pred_base = baseline.predict(X_test)
    mae_b, rmse_b, mape_b = compute_metrics(y_test, y_pred_base)

    results.append(["Baseline_LSTM", seq_len, units, lr, split, mae_b, rmse_b, mape_b])

    # ---- Attention Model ----
    att_model = build_attention_model(seq_len, n_features, units, lr)
    att_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    y_pred_att = att_model.predict(X_test)
    mae_a, rmse_a, mape_a = compute_metrics(y_test, y_pred_att)

    results.append(["Attention_LSTM", seq_len, units, lr, split, mae_a, rmse_a, mape_a])

    # Save attention weights from last run
    attention_layer = att_model.layers[-2]
    weights = attention_layer.get_weights()
    np.save("attention_weights.npy", weights)

# ======================
# 7. SAVE RESULTS
# ======================
results_df = pd.DataFrame(results, columns=[
    "model_type", "seq_len", "units", "learning_rate", "train_split",
    "MAE", "RMSE", "MAPE"
])

results_df.to_csv("results.csv", index=False)

print("\nAll experiments completed.")
print("Files generated:")
print("dataset.csv")
print("results.csv")
print("attention_weights.npy")
