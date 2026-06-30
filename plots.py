"""
diagnostic_plots.py
===================
Visualises the relationship between your top features and achievable_rr.
No model — just direct plots to see if rule-based filters are worth pursuing.

Run AFTER build_training_dataset_v2.py has produced training_dataset_v2.csv.

Outputs (saved to same folder):
  plot_1_rr_by_hour.png
  plot_2_rr_by_dow.png
  plot_3_pullback_depth.png
  plot_4_dist_to_swing_high.png
  plot_5_tc_size_vs_atr.png
  plot_6_feature_correlations.png
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.use("Agg")  # no display needed — saves to file

INPUT_FILE = "output_files/training_dataset_v2.csv"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#444466",
    "axes.labelcolor": "#ccccdd",
    "xtick.color": "#ccccdd",
    "ytick.color": "#ccccdd",
    "text.color": "#ccccdd",
    "grid.color": "#333355",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.size": 11,
})

BUCKET_COLORS = ["#e05252", "#e09a52", "#52a8e0", "#52e07a"]
BUCKET_LABELS = ["Bucket 0\n(<1.0 RR)", "Bucket 1\n(1-2 RR)",
                 "Bucket 2\n(2-3 RR)", "Bucket 3\n(>3 RR)"]

plots_folder = "plots"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(INPUT_FILE, parse_dates=["entry_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
print(f"  {len(df):,} trades  |  {df['entry_time'].min().date()} → {df['entry_time'].max().date()}")

df["hour"] = df["entry_time"].dt.hour
df["dow"] = df["entry_time"].dt.dayofweek  # 0=Mon … 4=Fri

# ── Plot 1: Average achievable_rr by hour ─────────────────────────────────────
print("\nPlot 1: RR by hour of day ...")
hourly = df.groupby("hour")["achievable_rr"].agg(["mean", "median", "count"]).reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(hourly["hour"], hourly["mean"], color="#52a8e0", alpha=0.7, label="Mean RR")
ax.plot(hourly["hour"], hourly["median"], color="#52e07a", marker="o",
        linewidth=2, label="Median RR")
ax.axhline(df["achievable_rr"].mean(), color="#e09a52", linestyle="--",
           linewidth=1.5, label=f"Overall mean ({df['achievable_rr'].mean():.2f})")
ax.set_xlabel("Hour of day (UTC)")
ax.set_ylabel("Achievable RR")
ax.set_title("Average Achievable RR by Hour of Day\n"
             "Hours clearly above the orange line = better session windows for higher RR targets")
ax.set_xticks(hourly["hour"])
ax.legend()
ax.grid(axis="y")

# Trade count as secondary label
ax2 = ax.twinx()
ax2.plot(hourly["hour"], hourly["count"], color="#aaaaaa", linestyle=":",
         linewidth=1, alpha=0.6)
ax2.set_ylabel("Trade count", color="#aaaaaa")
ax2.tick_params(axis="y", colors="#aaaaaa")
save(fig, f"{plots_folder}/plot_1_rr_by_hour.png")

# ── Plot 2: RR bucket distribution by hour ────────────────────────────────────
print("Plot 2: Bucket distribution by hour ...")
hourly_buckets = (df.groupby(["hour", "rr_bucket"])
                  .size()
                  .unstack(fill_value=0))
hourly_buckets_pct = hourly_buckets.div(hourly_buckets.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(13, 5))
bottom = np.zeros(len(hourly_buckets_pct))
for b, (color, label) in enumerate(zip(BUCKET_COLORS, BUCKET_LABELS)):
    if b in hourly_buckets_pct.columns:
        vals = hourly_buckets_pct[b].values
        ax.bar(hourly_buckets_pct.index, vals, bottom=bottom,
               color=color, alpha=0.85, label=label)
        bottom += vals

ax.set_xlabel("Hour of day (UTC)")
ax.set_ylabel("% of trades in each bucket")
ax.set_title("RR Bucket Distribution by Hour\n"
             "Hours with more green (Bucket 3) = naturally higher RR outcomes")
ax.set_xticks(hourly_buckets_pct.index)
ax.legend(loc="upper right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
save(fig, f"{plots_folder}/plot_2_rr_by_hour_buckets.png")

# ── Plot 3: Day of week ────────────────────────────────────────────────────────
print("Plot 3: RR by day of week ...")
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
daily = df.groupby("dow")["achievable_rr"].agg(["mean", "median", "count"]).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(daily["dow"], daily["mean"], color="#52a8e0", alpha=0.7, label="Mean RR")
ax.plot(daily["dow"], daily["median"], color="#52e07a", marker="o",
        linewidth=2, label="Median RR")
ax.axhline(df["achievable_rr"].mean(), color="#e09a52", linestyle="--",
           linewidth=1.5, label=f"Overall mean")
ax.set_xticks(daily["dow"])
ax.set_xticklabels([dow_labels[d] for d in daily["dow"]])
ax.set_ylabel("Achievable RR")
ax.set_title("Average Achievable RR by Day of Week")
ax.legend()
ax.grid(axis="y")
save(fig, f"{plots_folder}/plot_3_rr_by_dow.png")

# ── Plot 4: Pullback depth vs achievable_rr ───────────────────────────────────
print("Plot 4: Pullback depth vs RR ...")

# Bin pullback_depth into 10 quantile buckets and show mean RR per bucket
df["pd_bin"] = pd.qcut(df["pullback_depth"], q=10, duplicates="drop")
pd_stats = df.groupby("pd_bin", observed=True)["achievable_rr"].agg(["mean", "median", "count"])

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(pd_stats))
ax.bar(x, pd_stats["mean"], color="#52a8e0", alpha=0.7, label="Mean RR")
ax.plot(x, pd_stats["median"], color="#52e07a", marker="o",
        linewidth=2, label="Median RR")
ax.axhline(df["achievable_rr"].mean(), color="#e09a52", linestyle="--",
           linewidth=1.5, label="Overall mean")
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{str(b.left)[:5]}\nto\n{str(b.right)[:5]}" for b in pd_stats.index],
    fontsize=8
)
ax.set_xlabel("Pullback depth (ATR units) — left = shallow, right = deep")
ax.set_ylabel("Achievable RR")
ax.set_title("Pullback Depth vs Achievable RR\n"
             "Rising trend = deeper pullbacks lead to higher RR outcomes")
ax.legend()
ax.grid(axis="y")
save(fig, f"{plots_folder}/plot_4_pullback_depth.png")

# ── Plot 5: Distance to swing high vs achievable_rr ───────────────────────────
print("Plot 5: Distance to swing high vs RR ...")

df["sh_bin"] = pd.qcut(df["dist_to_swing_high_20"], q=10, duplicates="drop")
sh_stats = df.groupby("sh_bin", observed=True)["achievable_rr"].agg(["mean", "median", "count"])

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(sh_stats))
ax.bar(x, sh_stats["mean"], color="#52a8e0", alpha=0.7, label="Mean RR")
ax.plot(x, sh_stats["median"], color="#52e07a", marker="o",
        linewidth=2, label="Median RR")
ax.axhline(df["achievable_rr"].mean(), color="#e09a52", linestyle="--",
           linewidth=1.5, label="Overall mean")
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{str(b.left)[:5]}\nto\n{str(b.right)[:5]}" for b in sh_stats.index],
    fontsize=8
)
ax.set_xlabel("Distance to swing high (ATR units) — left = near resistance, right = clear air above")
ax.set_ylabel("Achievable RR")
ax.set_title("Distance to Swing High vs Achievable RR\n"
             "Rising trend = more room above entry leads to higher RR outcomes")
ax.legend()
ax.grid(axis="y")
save(fig, f"{plots_folder}/plot_5_dist_to_swing_high.png")

# ── Plot 6: Trigger candle size vs achievable_rr ──────────────────────────────
print("Plot 6: Trigger candle size vs RR ...")

df["tc_bin"] = pd.qcut(df["tc_size_vs_atr"], q=10, duplicates="drop")
tc_stats = df.groupby("tc_bin", observed=True)["achievable_rr"].agg(["mean", "median", "count"])

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(tc_stats))
ax.bar(x, tc_stats["mean"], color="#52a8e0", alpha=0.7, label="Mean RR")
ax.plot(x, tc_stats["median"], color="#52e07a", marker="o",
        linewidth=2, label="Median RR")
ax.axhline(df["achievable_rr"].mean(), color="#e09a52", linestyle="--",
           linewidth=1.5, label="Overall mean")
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{str(b.left)[:4]}\nto\n{str(b.right)[:4]}" for b in tc_stats.index],
    fontsize=8
)
ax.set_xlabel("Trigger candle size (multiples of ATR) — left = tiny candle, right = large candle")
ax.set_ylabel("Achievable RR")
ax.set_title("Trigger Candle Size vs Achievable RR\n"
             "Larger SL candle mechanically requires bigger move for same RR")
ax.legend()
ax.grid(axis="y")
save(fig, f"{plots_folder}/plot_6_tc_size_vs_atr.png")

# ── Plot 7: Feature correlation heatmap ───────────────────────────────────────
print("Plot 7: Feature correlations with achievable_rr ...")

feature_cols = [
    "tc_body_ratio", "tc_size_vs_atr", "tc_vol_ratio",
    "pullback_depth", "dist_to_swing_high_20", "dist_to_swing_high_10",
    "entry_in_range_20", "atr_pctrank", "consecutive_red_before",
    "prior5_avg_direction", "prior5_close_slope",
    "prior_1_direction", "prior_2_direction", "prior_3_direction",
    "hour_sin", "hour_cos",
]
feature_cols = [c for c in feature_cols if c in df.columns]

corr = df[feature_cols + ["achievable_rr"]].corr()["achievable_rr"].drop("achievable_rr")
corr_sorted = corr.sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#e05252" if v < 0 else "#52e07a" for v in corr_sorted.values]
ax.barh(corr_sorted.index, corr_sorted.values, color=colors, alpha=0.8)
ax.axvline(0, color="#aaaaaa", linewidth=1)
ax.set_xlabel("Pearson correlation with achievable_rr")
ax.set_title("Feature Correlations with Achievable RR\n"
             "Longer bar = stronger (linear) relationship. Most will be near 0.")
ax.grid(axis="x")
save(fig, f"{plots_folder}/plot_7_correlations.png")

# ── Summary stats printout ────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────────")
print(f"\nOverall achievable_rr:  mean={df['achievable_rr'].mean():.2f}  "
      f"median={df['achievable_rr'].median():.2f}  "
      f"std={df['achievable_rr'].std():.2f}")

print("\nTop 5 hours by mean achievable_rr:")
top_hours = (df.groupby("hour")["achievable_rr"]
             .mean()
             .sort_values(ascending=False)
             .head(5))
for h, v in top_hours.items():
    n = (df["hour"] == h).sum()
    print(f"  Hour {h:02d}:00  mean RR = {v:.2f}  ({n} trades)")

print("\nBottom 5 hours by mean achievable_rr:")
bot_hours = (df.groupby("hour")["achievable_rr"]
             .mean()
             .sort_values()
             .head(5))
for h, v in bot_hours.items():
    n = (df["hour"] == h).sum()
    print(f"  Hour {h:02d}:00  mean RR = {v:.2f}  ({n} trades)")

print("\nCorrelations with achievable_rr (strongest first):")
for feat, val in corr.abs().sort_values(ascending=False).head(8).items():
    direction = "+" if corr[feat] > 0 else "-"
    print(f"  {feat:<30s}  r = {direction}{val:.4f}")

print("\nAll 7 plots saved. Open them to see if clear patterns exist.")
print("If plot_1 shows big differences between hours → time filter is your best lever.")
print("If plot_4 shows rising trend → pullback depth filter is worth adding.")
