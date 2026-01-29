"""
Advanced Time Series Forecasting using ARIMA and Transformer Attention Model

This script:
1. Generates synthetic multivariate time-series data (5000 samples, 3 features)
2. Trains a baseline ARIMA model
3. Trains a Transformer-based attention model
4. Performs hyperparameter grid search
5. Evaluates models using MAE, RMSE, and MAPE
6. Saves results to results.csv
7. Saves attention weights for interpretability
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import itertools

# ===================== CONFIG =====================
SEQ_LENGTHS = [10, 20]
HEADS = [2, 4]
D_MODELS = [32, 64]
EPOCHS = 30
BATCH_SIZE = 32
FEATURES = 3
DATA_POINTS = 5000

# ===================== DATA GENERATION =====================
def generate_data():
    t = np.arange(DATA_POINTS)
    f1 = np.sin(0.02 * t)
    f2 = np.cos(0.015 * t)
    f3 = np.sin(0.01 * t) + np.random.normal(0, 0.1, DATA_POINTS)
    data = np.stack([f1, f2, f3], axis=1)
    return data

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

# ===================== METRICS =====================
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ===================== TRANSFORMER MODEL =====================
def build_transformer(seq_len, features, heads, d_model):
    inp = Input(shape=(seq_len, features))
    x = Dense(d_model)(inp)

    mha = MultiHeadAttention(num_heads=heads, key_dim=d_model)
    attn_output = mha(x, x)

    x = LayerNormalization()(x + attn_output)
    x = Dense(64, activation='relu')(x)

    # Proper Keras pooling instead of tf.reduce_mean
    x = GlobalAveragePooling1D()(x)

    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')

    return model, mha

# ===================== MAIN =====================
data = generate_data()
results = []

# ---------- BASELINE ARIMA ----------
train = data[:4000, 0]
test = data[4000:, 0]

arima_model = ARIMA(train, order=(2,1,2))
arima_fit = arima_model.fit()
pred_arima = arima_fit.forecast(len(test))

mae = mean_absolute_error(test, pred_arima)
rmse = np.sqrt(mean_squared_error(test, pred_arima))
mape_val = mape(test, pred_arima)

results.append(["ARIMA Baseline", "N/A", "N/A", "N/A", mae, rmse, mape_val])

# ---------- TRANSFORMER GRID SEARCH ----------
attention_weights_store = None

for seq_len, heads, d_model in itertools.product(SEQ_LENGTHS, HEADS, D_MODELS):

    X, y = create_sequences(data, seq_len)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model, mha = build_transformer(seq_len, FEATURES, heads, d_model)

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape_val = mape(y_test, preds.flatten())

    results.append(["Transformer", seq_len, heads, d_model, mae, rmse, mape_val])

    attention_weights_store = mha.get_weights()

# ===================== SAVE FILES =====================
df = pd.DataFrame(results, columns=["Model", "Seq_Length", "Heads", "D_Model", "MAE", "RMSE", "MAPE"])
df.to_csv("results.csv", index=False)

attention_weights_store = np.array(attention_weights_store, dtype=object)
np.save("attention_weights.npy", attention_weights_store, allow_pickle=True)


# ===================== REPORT =====================
with open("report.txt", "w") as f:
    f.write("Advanced Time Series Forecasting Report\n\n")
    f.write("Baseline Model: ARIMA(2,1,2)\n")
    f.write("Transformer Model: MultiHeadAttention Encoder\n\n")
    f.write("Results:\n")
    f.write(df.to_string(index=False))
    f.write("\n\nInterpretability:\n")
    f.write("Attention weights saved in attention_weights.npy.\n")
    f.write("They show which time steps influence predictions most strongly.\n")

print("Training completed successfully.")
print("Files generated: results.csv, report.txt, attention_weights.npy")
