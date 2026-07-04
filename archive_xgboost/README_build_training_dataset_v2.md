# build_training_dataset_v2.py

Reads your raw MT5 exports and OHLCV data, computes candle-based features
for each trade, and saves a clean CSV ready for model training.

## Inputs (must be in the same folder)

- `trade_stats.csv` — MT5 export with MAE, MFE, SL distance per trade
- `MT5_databento-ohlcv-1m.csv` — 1-minute OHLCV data

## Output

- `training_dataset_v2.csv` — one row per trade, 29 features + label columns

## What it does, step by step

1. Loads `trade_stats.csv` and computes `achievable_rr = MFE / SL_distance`
2. Assigns each trade an `rr_bucket` label (0/1/2/3) based on that RR value
3. Loads the full OHLCV history into memory
4. For each trade, slices back to the entry timestamp and computes 29 features
   from the trigger candle and the candles surrounding it
5. Saves the result to `training_dataset_v2.csv`

## Configuration (top of file)

| Setting | Default | Description |
|---------|---------|-------------|
| `RR_BINS` | 0, 1, 2, 3, ∞ | Bucket boundaries |
| `CONTEXT_BARS` | 20 | How many bars before trigger to analyse |

## Notes

- All features are normalised (no absolute prices) so data from 2010 and 2026
  is directly comparable
- The script automatically skips trades where there isn't enough OHLCV history
  before the entry (e.g. the very first bars in the file)
- MT5 sometimes appends summary/footer rows to the CSV — these are detected
  and dropped automatically
- Runtime: a few minutes for 25,000 trades on 15 years of 1-min data
