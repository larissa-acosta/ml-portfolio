---
license: mit
library_name: keras
tags:
  - time-series
  - forecasting
  - lstm
  - energy
  - anomaly-detection
  - tensorflow
metrics:
  - mae
  - rmse
  - r2
---

# Steel Energy LSTM: consumption forecasting for bottleneck detection

LSTM that forecasts the next 15-minute energy consumption (`Usage_kWh`) of a steel plant from the previous 15 hours. Large forecast errors are flagged as potential **energy bottlenecks** (periods of inefficient operation).

Trained with TensorFlow 2.21 / Keras 3.13. Built for the paper *Identificação de Gargalos de Energia em Fábricas* (PUCPR, 2nd place in the university project gallery).

## Model

| | |
|---|---|
| Architecture | LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1) |
| Input | 60 time steps (15 h at 15 min) × 5 features |
| Features | `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor` |
| Target | next-step `Usage_kWh` |
| Training | Adam, MSE loss, batch 64, up to 150 epochs, 20% of the training set for validation, early stopping (patience 10; best epoch: 35) |
| Normalization | MinMaxScaler fit on the training set only (`scaler.pkl`, required for inference) |
| Framework | TensorFlow / Keras |

## Results (chronological 80/20 test split, 7,008 test sequences)

| Metric | Value |
|---|---|
| MAE | 5.00 kWh |
| RMSE | 9.05 kWh |
| R² | 0.9169 |

**Bottleneck rule:** absolute error > MAE + 1.5σ (16.31 kWh) → 534 of 7,008 periods flagged (7.62%). Mean consumption in the flagged periods: 69.71 kWh.

## Usage

```python
import pickle, numpy as np
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

repo = "larissaac/steel-energy-lstm"
model = load_model(hf_hub_download(repo, "lstm_model.keras"), compile=False)
scaler = pickle.load(open(hf_hub_download(repo, "scaler.pkl"), "rb"))

# window: array (60, 5) of raw feature values in the order listed above
x = scaler.transform(window)[None, ...]
pred_scaled = model.predict(x)[0, 0]
dummy = np.zeros((1, 5)); dummy[0, 0] = pred_scaled
pred_kwh = scaler.inverse_transform(dummy)[0, 0]
```

Try it in the Space: `larissaac/steel-energy-bottlenecks`.

## Data

Steel Industry Energy Consumption dataset (DAEWOO Steel Co., 2018, 35,040 rows at 15-minute resolution). Check the original source for its license before reuse.

## Limitations

- Single plant, single year; not validated on other facilities.
- The bottleneck threshold is computed from the test residuals (MAE + 1.5σ), so it is descriptive rather than a fixed production rule.
- The model flags large forecast errors, not confirmed process faults; a flagged period needs operational context to be interpreted.
- Keras 3 `.keras` format; load with `compile=False` and a TensorFlow version close to the one used for training.

## Authors

Ayumi Sato Brizola, Larissa Cristina Alves da Costa · PUCPR
