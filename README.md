# Advanced Time Series Forecasting with Transformer and LSTM

## Project Overview
This project implements and evaluates advanced deep learning models for multivariate time series forecasting.  
The goal is to compare a baseline LSTM model with a Transformer Encoder model using an attention mechanism and analyze the interpretability of attention weights.

The system performs:
- Synthetic multivariate time series generation (5000 observations)
- Hyperparameter tuning
- Model comparison using MAE, RMSE, and MAPE
- Attention weight extraction and interpretability analysis
- Automated result reporting

---

## Models Implemented

### 1. Baseline Model: LSTM
A standard Long Short-Term Memory (LSTM) network is used as a benchmark for sequence forecasting without attention.

### 2. Transformer Encoder Model
A Transformer Encoder using Multi-Head Attention is implemented for sequence-to-one forecasting.  
This model captures long-range dependencies and allows interpretability via attention weights.

---

## Dataset
Synthetic multivariate time series with 3 features:
- Feature 1: Sine wave
- Feature 2: Cosine wave
- Feature 3: Lower-frequency sine wave
- Target: Combination of all three with noise

Total observations: **5000**

---

## Hyperparameter Search
Grid search is performed over:
- Sequence length: 20, 30
- Hidden units: 32, 64
- Attention heads: 2, 4 (Transformer only)
- Epochs: 50

Each configuration is evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

---

## Outputs
The program generates the following deliverables:
- `main.py` – Full experiment pipeline
- `results.csv` – Quantitative comparison table
- `attention_weights.npy` – Saved attention matrices
- `report.txt` – Detailed analysis and discussion

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
