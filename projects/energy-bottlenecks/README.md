# Energy Bottleneck Identification in Steel Manufacturing

> 🥈 **2nd place** at the PUCPR project gallery.

Detecting energy inefficiencies ("bottlenecks") in a steel plant by comparing two recurrent approaches on real consumption data: **LSTM forecasting** and a **GRU autoencoder** for anomaly detection.

## Problem

Traditional bottleneck methods (value stream mapping, utilization analysis) work for static systems but miss dynamic bottlenecks. This project tests whether purely data-driven models can flag inefficient operation without a simulation model.

## Data

- DAEWOO Steel Co. Ltd energy consumption, **35,040 observations** (2018, 15-minute resolution).
- Public [Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) data (`Steel_industry_data.csv`, 35,040 rows); check the source page for the license before redistributing. Min-max normalization **fit on the training set only**; chronological 80/20 train/test split (no shuffling); 20% of the training set used for validation and early stopping (patience 10). IQR outliers are reported but kept, since high-consumption peaks are the bottleneck candidates.

## Method

Following *Design Science Research*:

| | LSTM (forecasting) | GRU Autoencoder (anomaly detection) |
|---|---|---|
| Input | 5 features (consumption, reactive power, power factor), window of 60 steps (15 h) | consumption only, window of 60 steps (15 h) |
| Architecture | 3 LSTM layers (128/64/32), dropout 0.2 between them, Adam, MSE loss | GRU encoder-decoder (128/128), dropout 0.2, Adam, MAE loss |
| Bottleneck rule | prediction error > MAE + 1.5σ = **16.31 kWh** | reconstruction error above the **98th percentile** of training error (0.0898) |

## Results

| Model | Key metrics | Bottlenecks flagged |
|---|---|---|
| **LSTM** | R² **0.9169**, MAE **5.00 kWh**, RMSE **9.05 kWh** | **534** periods (7.62% of test set), mean consumption 69.71 kWh |
| GRU Autoencoder | reconstruction MAE 3.78 kWh, RMSE 7.70 kWh; precision 13.4%, recall 1.3%* | 67 windows (0.96%) |

**Takeaway:** the LSTM approach gave a much better balance between accuracy and operational coverage. The GRU autoencoder reconstructed well but its anomaly threshold was poorly calibrated, so practical usefulness depends on threshold calibration and operational context.

\* Precision/recall are computed against a proxy reference (consumption above the 90th percentile of the test set), not against labelled bottlenecks.

## Demo

🤗 Model: [larissaac/steel-energy-lstm](https://huggingface.co/larissaac/steel-energy-lstm) · 🤗 Space: [larissaac/steel-energy-bottlenecks](https://huggingface.co/spaces/larissaac/steel-energy-bottlenecks)

## Reproduce

Open `bottleneck_LSTM_GRU_v2.ipynb` in Google Colab (outputs from the last run are included) and upload `Steel_industry_data.csv` (Steel Industry Energy Consumption dataset; download it from its original source). The notebook trains both models and exports them.

## My contribution

I wrote the code for the whole pipeline (preprocessing, LSTM forecasting, GRU autoencoder and evaluation), curated the dataset, and co-wrote the paper.

## Authors

Ayumi Sato Brizola, Larissa Cristina Alves da Costa · Computer Science, PUCPR

📄 Paper: [`paper_v2.pdf`](paper_v2.pdf)
