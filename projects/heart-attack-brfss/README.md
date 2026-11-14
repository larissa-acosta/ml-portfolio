# Heart Attack Prediction with BRFSS 2022

Comparing Logistic Regression, Random Forest and Gradient Boosting on **444,975 survey records** where only **5.68%** are positive cases, with a protocol designed to **prevent data leakage**.

## Problem

With severe class imbalance, accuracy is misleading (a model that always says "no" gets ~94%). The goal is a fair, leakage-free comparison driven by F1-macro and recall on the minority class.

## Data

CDC BRFSS 2022 telephone survey: 444,975 instances, 38 health indicators, target `HadHeartAttack`.

## Method

- **20% held out** as a final unseen test set (88,383 instances) before any processing.
- **SMOTE applied only inside each fold** of 5-fold stratified CV, using pipelines so synthetic samples never reach validation.
- Grid Search (3-fold inner CV, optimizing F1-macro) on a stratified 15% sample of the training data.
- Feature selection: top-20 features by Gradient Boosting importance.
- Friedman test + Nemenyi post-hoc for model comparison.

## Results (5-fold CV + SMOTE, mean ± std)

| Model | ROC-AUC | F1-macro | Recall (positive) | Precision (positive) |
|---|---|---|---|---|
| Logistic Regression | 0.8787 ± 0.0017 | 0.6167 ± 0.0017 | **0.7571** ± 0.0073 | 0.2130 ± 0.0019 |
| Random Forest | 0.8765 ± 0.0019 | 0.7054 ± 0.0033 | 0.5430 ± 0.0067 | 0.3859 ± 0.0084 |
| **Gradient Boosting** | 0.8738 ± 0.0022 | **0.7111** ± 0.0021 | 0.5053 ± 0.0083 | **0.4196** ± 0.0070 |

- On the **unseen test set**, Gradient Boosting scored F1-macro **0.72**, recall 0.52, precision 0.42 (CV and test agree, so no leakage-driven inflation).
- Friedman test: significant difference across models (p = 0.0067); Nemenyi: Gradient Boosting > Logistic Regression (p = 0.0045).
- Top-20 feature selection kept F1-macro at 0.710 (vs 0.711) with fewer inputs.
- Strongest predictor: **history of angina**, consistent with clinical literature.
- **Trade-off:** Logistic Regression has the best recall (useful for screening); Gradient Boosting has the best precision/F1 balance.

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook heart_attack_brfss.ipynb
```

## My contribution

I wrote all the code (data preparation, SMOTE, model training and evaluation), curated the BRFSS 2022 data, and co-wrote the paper.

## Authors

Alex Menegatti Secco, Ayumi Sato Brizola, Larissa Cristina Alves da Costa, Mariana de Castro · Data Science, PUCPR

📄 Paper (IEEE format): [`paper.pdf`](paper.pdf)
