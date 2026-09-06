"""
CostGuard — Exploratory Data Analysis (EDA)
=============================================
Visual exploration of the Kaggle/ULB Credit Card Fraud Detection dataset.
Covers class distribution, feature distributions, correlations, and Amount analysis.

Run: jupyter notebook notebooks/eda.py  (or convert to .ipynb)
Or execute directly: python notebooks/eda.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Load Dataset
# ==========================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nBasic Statistics:")
print(df.describe().round(2))
print(f"\nMissing Values: {df.isnull().sum().sum()}")

# ==========================================
# 2. Class Distribution
# ==========================================
fraud_count = df["Class"].sum()
legit_count = len(df) - fraud_count
fraud_pct = (fraud_count / len(df)) * 100

print(f"\n--- Class Distribution ---")
print(f"Legitimate: {legit_count:,} ({100 - fraud_pct:.3f}%)")
print(f"Fraud:      {fraud_count:,} ({fraud_pct:.3f}%)")
print(f"Imbalance:  {legit_count / fraud_count:.0f}:1")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart
colors = ["#475569", "#e11d48"]
bars = axes[0].bar(["Legitimate", "Fraud"], [legit_count, fraud_count], color=colors, width=0.5)
axes[0].set_ylabel("Transaction Count")
axes[0].set_title("Class Distribution (Absolute)")
for bar, val in zip(bars, [legit_count, fraud_count]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"{val:,}", ha="center", va="bottom", fontsize=9)

# Log-scale bar chart to see fraud
bars2 = axes[1].bar(["Legitimate", "Fraud"], [legit_count, fraud_count], color=colors, width=0.5)
axes[1].set_ylabel("Transaction Count (Log Scale)")
axes[1].set_title("Class Distribution (Log Scale)")
axes[1].set_yscale("log")
for bar, val in zip(bars2, [legit_count, fraud_count]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                 f"{val:,}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
plt.close(fig)
print("Saved: class_distribution.png")

# ==========================================
# 3. Amount Distribution
# ==========================================
print(f"\n--- Amount Statistics ---")
print(f"Mean:   ${df['Amount'].mean():.2f}")
print(f"Median: ${df['Amount'].median():.2f}")
print(f"Max:    ${df['Amount'].max():.2f}")
print(f"Min:    ${df['Amount'].min():.2f}")
print(f"Std:    ${df['Amount'].std():.2f}")

fraud_amounts = df[df["Class"] == 1]["Amount"]
legit_amounts = df[df["Class"] == 0]["Amount"]

print(f"\nFraud transactions — Mean: ${fraud_amounts.mean():.2f}, Median: ${fraud_amounts.median():.2f}, Max: ${fraud_amounts.max():.2f}")
print(f"Legit transactions — Mean: ${legit_amounts.mean():.2f}, Median: ${legit_amounts.median():.2f}, Max: ${legit_amounts.max():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Full histogram
axes[0].hist(df["Amount"], bins=100, color="#475569", alpha=0.8, edgecolor="none")
axes[0].set_xlabel("Transaction Amount ($)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Amount Distribution (All Transactions)")
axes[0].axvline(df["Amount"].median(), color="#e11d48", linestyle="--", linewidth=1, label=f"Median: ${df['Amount'].median():.2f}")
axes[0].legend(fontsize=8)

# Amount < $500 zoom
small = df[df["Amount"] < 500]["Amount"]
axes[1].hist(small, bins=100, color="#475569", alpha=0.8, edgecolor="none")
axes[1].set_xlabel("Transaction Amount ($)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Amount < $500 (Zoomed)")

# Fraud vs Legit comparison
axes[2].hist(legit_amounts[legit_amounts < 500], bins=80, alpha=0.6, color="#475569", label="Legitimate", density=True)
axes[2].hist(fraud_amounts[fraud_amounts < 500], bins=40, alpha=0.7, color="#e11d48", label="Fraud", density=True)
axes[2].set_xlabel("Transaction Amount ($)")
axes[2].set_ylabel("Density")
axes[2].set_title("Amount: Fraud vs Legitimate (Normalized)")
axes[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=150)
plt.close(fig)
print("Saved: amount_distribution.png")

# ==========================================
# 4. Time Distribution
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["Time"] / 3600, bins=48, color="#475569", alpha=0.8, edgecolor="none")
axes[0].set_xlabel("Time (Hours Since First Transaction)")
axes[0].set_ylabel("Transaction Count")
axes[0].set_title("Transaction Volume Over Time")

# Fraud over time
fraud_times = df[df["Class"] == 1]["Time"] / 3600
axes[1].hist(fraud_times, bins=48, color="#e11d48", alpha=0.8, edgecolor="none")
axes[1].set_xlabel("Time (Hours)")
axes[1].set_ylabel("Fraud Count")
axes[1].set_title("Fraud Transactions Over Time")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "time_distribution.png", dpi=150)
plt.close(fig)
print("Saved: time_distribution.png")

# ==========================================
# 5. Feature Correlation Heatmap (V1-V28)
# ==========================================
pca_features = [f"V{i}" for i in range(1, 29)]
corr = df[pca_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(pca_features)))
ax.set_yticks(range(len(pca_features)))
ax.set_xticklabels(pca_features, fontsize=7, rotation=45, ha="right")
ax.set_yticklabels(pca_features, fontsize=7)
ax.set_title("Feature Correlation Matrix (V1–V28)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)
print("Saved: correlation_heatmap.png")

# ==========================================
# 6. Top Discriminative Features (by class separation)
# ==========================================
# Measure separation using the absolute difference in means between fraud and legit
mean_diff = {}
for col in pca_features + ["Amount", "Time"]:
    fraud_mean = df[df["Class"] == 1][col].mean()
    legit_mean = df[df["Class"] == 0][col].mean()
    mean_diff[col] = abs(fraud_mean - legit_mean) / (df[col].std() + 1e-8)

mean_diff_sorted = dict(sorted(mean_diff.items(), key=lambda x: x[1], reverse=True))
top_features = list(mean_diff_sorted.keys())[:8]

print(f"\n--- Top 8 Most Discriminative Features (by standardized mean difference) ---")
for feat, val in list(mean_diff_sorted.items())[:8]:
    print(f"  {feat:<8}: {val:.4f}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, feat in enumerate(top_features):
    ax = axes[i]
    legit_vals = df[df["Class"] == 0][feat]
    fraud_vals = df[df["Class"] == 1][feat]
    ax.hist(legit_vals, bins=50, alpha=0.5, color="#475569", label="Legit", density=True)
    ax.hist(fraud_vals, bins=50, alpha=0.7, color="#e11d48", label="Fraud", density=True)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylabel("Density" if i % 4 == 0 else "")

fig.suptitle("Top 8 Discriminative Features: Fraud vs Legitimate Distributions", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "top_features.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: top_features.png")

# ==========================================
# 7. Amount Skewness Analysis (justifies RobustScaler)
# ==========================================
from scipy import stats

amount_skew = stats.skew(df["Amount"])
amount_kurtosis = stats.kurtosis(df["Amount"])

print(f"\n--- Amount Skewness Analysis ---")
print(f"Skewness:  {amount_skew:.4f}  (>2 = heavily right-skewed)")
print(f"Kurtosis:  {amount_kurtosis:.4f}  (high = heavy tails)")
print(f"Justification: RobustScaler is appropriate because Amount is heavily")
print(f"  right-skewed (skew={amount_skew:.2f}) with extreme outliers.")
print(f"  StandardScaler would be distorted by the $25K max transaction.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].boxplot([legit_amounts, fraud_amounts], tick_labels=["Legitimate", "Fraud"],
                patch_artist=True, boxprops=dict(facecolor="#e2e8f0"),
                medianprops=dict(color="#e11d48", linewidth=2))
axes[0].set_ylabel("Transaction Amount ($)")
axes[0].set_title("Amount Box Plot by Class")

# QQ plot for Amount
from scipy.stats import probplot
probplot(df["Amount"], dist="norm", plot=axes[1])
axes[1].set_title("Amount Q-Q Plot (vs. Normal)")
axes[1].get_lines()[0].set_color("#475569")
axes[1].get_lines()[1].set_color("#e11d48")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "amount_skewness.png", dpi=150)
plt.close(fig)
print("Saved: amount_skewness.png")

print("\n" + "=" * 60)
print("EDA complete. All figures saved to notebooks/figures/")
print("=" * 60)
