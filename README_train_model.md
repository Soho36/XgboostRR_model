# train_model.py

Trains an XGBoost classifier on the dataset produced by
`build_training_dataset_v2.py` and evaluates whether the features
have genuine predictive signal.

## Input

- `training_dataset_v2.csv` — produced by the build script

## Outputs

- `rr_model.json` — trained model (load this for live prediction)
- `feature_importance.csv` — which features contributed most

## What it does, step by step

1. Loads the dataset and splits it **by time** — first 80% for training,
   last 20% for testing. No shuffling (this is time-series data).
2. Applies balanced class weights so the model doesn't just predict
   the most common bucket every time.
3. Trains XGBoost with early stopping (stops when test loss stops improving).
4. Prints a full evaluation: accuracy, precision/recall per bucket,
   confusion matrix.
5. Runs a **permutation sanity check** — trains again with shuffled labels
   to confirm the real model genuinely learned something vs random chance.
6. Saves the model and feature importances.

## How to interpret the output

| What to look at | Good sign | Bad sign |
|----------------|-----------|----------|
| Model vs baseline accuracy | Model clearly higher | Same or lower |
| Permutation check | Real model >> shuffled | Gap < 0.02 |
| Confusion matrix | Diagonal is strongest | All predictions in one column |
| Feature importance | Few features dominate | All features equal weight |

## Configuration (top of file)

| Setting | Default | Description |
|---------|---------|-------------|
| `INPUT_FILE` | `training_dataset_v2.csv` | Change if your file has a different name |
| `TRAIN_RATIO` | 0.80 | Fraction of trades used for training |
| `XGBOOST_PARAMS` | See file | Tuning parameters — defaults are conservative |

## Live prediction (after training)

```python
from xgboost import XGBClassifier
import pandas as pd

model = XGBClassifier()
model.load_model("rr_model.json")

features = pd.DataFrame([{ "tc_body_ratio": 0.7, "pullback_depth": 2.1, ... }])
bucket = model.predict(features)[0]

rr_map = {0: 1.0, 1: 1.5, 2: 2.5, 3: 3.0}
print(f"Use RR: {rr_map[bucket]}")
```

The feature names and order must exactly match what the build script produced.
