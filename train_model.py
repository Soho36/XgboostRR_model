"""
train_model.py
==============
Trains an XGBoost classifier on training_dataset.csv to predict RR bucket.
Outputs:
  - rr_model.json        (trained model — load this for live prediction)
  - feature_importance.csv

Run AFTER build_training_dataset.py.

Usage:
  python train_model.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import json

# ── CONFIGURATION ────────────────────────────────────────────────────────────

INPUT_FILE  = "training_dataset.csv"
MODEL_FILE  = "rr_model.json"
FI_FILE     = "feature_importance.csv"

TRAIN_RATIO = 0.80   # first 80% of trades (by time) for training

# XGBoost params — conservative defaults for small-to-medium datasets
XGBOOST_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 4,      # shallow = less overfit
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,      # require at least 5 samples per leaf
    gamma             = 0.1,    # minimum loss reduction to split
    eval_metric       = "mlogloss",
    early_stopping_rounds = 30,
    random_state      = 42,
)

# ── LOAD DATA ────────────────────────────────────────────────────────────────

print("Loading dataset ...")
df = pd.read_csv(INPUT_FILE, parse_dates=["entry_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
print(f"  {len(df)} trades loaded.")

# Metadata / label / non-feature columns to exclude
META_COLS = ["entry_time", "exit_time", "entry_price",
             "mae", "mfe", "pnl", "sl_distance", "achievable_rr", "rr_bucket"]

feature_cols = [c for c in df.columns if c not in META_COLS]
print(f"  {len(feature_cols)} feature columns: {feature_cols}")

X = df[feature_cols]
y = df["rr_bucket"]

# ── TEMPORAL TRAIN / TEST SPLIT ──────────────────────────────────────────────
# CRITICAL: sort by time, split sequentially. Never shuffle for time-series.

split_idx = int(len(df) * TRAIN_RATIO)
split_date = df["entry_time"].iloc[split_idx]

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTrain: {len(X_train)} trades  ({df['entry_time'].iloc[0].date()} → {split_date.date()})")
print(f"Test : {len(X_test)} trades   ({split_date.date()} → {df['entry_time'].iloc[-1].date()})")

# ── CLASS BALANCE CHECK ──────────────────────────────────────────────────────

print("\nClass distribution in training set:")
for b in sorted(y_train.unique()):
    n   = (y_train == b).sum()
    pct = n / len(y_train) * 100
    print(f"  Bucket {b}: {n:4d}  ({pct:.1f}%)")

# Compute scale_pos_weight equivalent via sample_weight
# This helps XGBoost not just predict the majority class
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight("balanced", y_train)

# ── TRAIN ────────────────────────────────────────────────────────────────────

print("\nTraining XGBoost ...")

model = XGBClassifier(**XGBOOST_PARAMS)
model.fit(
    X_train, y_train,
    sample_weight   = sample_weights,
    eval_set        = [(X_test, y_test)],
    verbose         = False,
)

best_iter = model.best_iteration
print(f"  Best iteration: {best_iter} (early stopping at patience=30)")

# ── EVALUATE ─────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

print("\n── Test Set Performance ────────────────────────────────────────────────")
print(classification_report(y_test, y_pred,
                             target_names=[f"Bucket {i}" for i in range(4)]))

print("Confusion matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
    index  =[f"Actual {i}" for i in range(4)],
    columns=[f"Pred {i}"   for i in range(4)]
)
print(cm_df.to_string())

# ── BASELINE COMPARISON ──────────────────────────────────────────────────────
# A model that always predicts the majority class — our minimum bar to beat

majority_class = y_train.mode()[0]
baseline_acc   = (y_test == majority_class).mean()
model_acc      = (y_pred == y_test.values).mean()

print(f"\nBaseline accuracy (always predict bucket {majority_class}): {baseline_acc:.3f}")
print(f"Model accuracy:                                             {model_acc:.3f}")

if model_acc > baseline_acc + 0.03:
    print("  ✓ Model beats baseline — there is some predictive signal.")
elif model_acc > baseline_acc:
    print("  ~ Model slightly above baseline — signal is weak, interpret carefully.")
else:
    print("  ✗ Model does not beat baseline — features may not contain useful signal.")
    print("    Consider: more data, different features, or different label definition.")

# ── PERMUTATION SANITY CHECK ─────────────────────────────────────────────────
# Shuffle labels and retrain — model should perform WORSE than real labels.
# If shuffled performance is similar to real, there's data leakage or the
# model is just memorising structure.

print("\nRunning permutation sanity check (shuffled labels) ...")
y_train_shuffled = y_train.sample(frac=1, random_state=99).values
model_shuffle = XGBClassifier(**XGBOOST_PARAMS)
model_shuffle.fit(
    X_train, y_train_shuffled,
    eval_set  = [(X_test, y_test)],
    verbose   = False,
)
shuffled_acc = (model_shuffle.predict(X_test) == y_test.values).mean()
print(f"  Shuffled-label model accuracy: {shuffled_acc:.3f}")
print(f"  Real model accuracy:           {model_acc:.3f}")
if model_acc > shuffled_acc + 0.02:
    print("  ✓ Real model meaningfully outperforms shuffled — signal looks genuine.")
else:
    print("  ✗ Real and shuffled models perform similarly — possible leakage or no signal.")

# ── FEATURE IMPORTANCE ───────────────────────────────────────────────────────

print("\n── Feature Importance (top 10) ─────────────────────────────────────────")
fi = pd.DataFrame({
    "feature":    feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(fi.head(10).to_string(index=False))
fi.to_csv(FI_FILE, index=False)
print(f"\nFull importance saved to {FI_FILE}")

# ── SAVE MODEL ───────────────────────────────────────────────────────────────

model.save_model(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ── HOW TO USE IN LIVE TRADING ───────────────────────────────────────────────

print("""
── Live Prediction Example ──────────────────────────────────────────────────
  from xgboost import XGBClassifier
  import pandas as pd

  model = XGBClassifier()
  model.load_model("rr_model.json")

  # At entry signal time, build one feature row (same columns as training):
  features = pd.DataFrame([{
      "atr_14":           12.5,
      "atr_pctrank":      0.72,
      "rvol_30":          0.0003,
      # ... all other feature columns ...
  }])

  bucket = model.predict(features)[0]
  rr_map = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5}
  rr_to_use = rr_map[bucket]
  print(f"Predicted RR bucket: {bucket} → use RR {rr_to_use}")
""")
