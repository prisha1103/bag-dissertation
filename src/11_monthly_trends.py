"""
Monthly Temporal Trend Analysis.

Tracks how Reddit sentiment moves month-by-month per brand, with creative
director transition dates marked where applicable. Brand-official text has
no reliable dates (mostly evergreen "About" copy), so this is Reddit-only --
say this explicitly in your Methodology, it's a real constraint, not an
oversight.

IMPORTANT: some brand-months will have very few posts (especially early
years, and Supreme which only has ~4 months of data at all -- see the
n_posts column and DON'T over-read a trend line built from <10 posts in a
given month. This script flags low-volume months rather than hiding them.

Run in Colab or locally (no GPU needed).
Needs: bag_sentiment_emotion.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MIN_POSTS_FOR_RELIABLE_MONTH = 10

TRANSITIONS = {
    "Chanel":    pd.Timestamp("2024-12-12"),
    "Dior":      pd.Timestamp("2025-06-02"),
    "Gucci":     pd.Timestamp("2025-03-13"),
    "Off-White": pd.Timestamp("2024-02-01"),
}

df = pd.read_csv("bag_sentiment_emotion.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
reddit = df[df["source_type"] == "reddit"].dropna(subset=["date"])
reddit["year_month"] = reddit["date"].dt.to_period("M").dt.to_timestamp()

# ---------------------------------------------------------------------------
# 1. Monthly sentiment + post volume per brand
# ---------------------------------------------------------------------------
monthly = (
    reddit.groupby(["brand", "year_month"])["sentiment_signed"]
    .agg(mean_sentiment="mean", n_posts="count")
    .reset_index()
)
monthly["reliable"] = monthly["n_posts"] >= MIN_POSTS_FOR_RELIABLE_MONTH
monthly.to_csv("monthly_sentiment_trend.csv", index=False)

print(f"Months with fewer than {MIN_POSTS_FOR_RELIABLE_MONTH} posts (treat cautiously):")
print(monthly[~monthly["reliable"]].groupby("brand").size().to_string())
print(f"\nTotal months per brand:")
print(monthly.groupby("brand").size().to_string())
print("\nSaved: monthly_sentiment_trend.csv")

# ---------------------------------------------------------------------------
# 2. Plot: small multiples (one subplot per brand) with a rolling average
#    to smooth month-to-month noise. Far more legible than overlaying all
#    6 brands on one axis, and the rolling average avoids over-reading
#    single noisy months.
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
brands = sorted(monthly["brand"].unique())
ROLLING_WINDOW = 6  # months

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
axes = axes.flatten()

for ax, brand in zip(axes, brands):
    b = monthly[(monthly["brand"] == brand) & (monthly["reliable"])].sort_values("year_month")
    if len(b) == 0:
        ax.set_title(f"{brand} (no reliable months)", fontsize=12)
        continue

    # raw monthly points, faded, for transparency about the underlying noise
    ax.scatter(b["year_month"], b["mean_sentiment"], s=10, color="lightsteelblue",
               alpha=0.6, zorder=1)

    # rolling average -- the actual trend line to read
    b = b.set_index("year_month")
    rolling = b["mean_sentiment"].rolling(f"{ROLLING_WINDOW*30}D", min_periods=2).mean()
    ax.plot(rolling.index, rolling.values, color="#C44E52", linewidth=2.2, zorder=3)

    if brand in TRANSITIONS:
        ax.axvline(TRANSITIONS[brand], color="black", linestyle="--", alpha=0.6, linewidth=1.2)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_title(brand, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=30)

fig.suptitle(f"Monthly Reddit Sentiment by Brand ({ROLLING_WINDOW}-month rolling average, "
             f"faded dots = raw monthly values)\nDashed line = creative director transition date "
             f"(where applicable). Months with <{MIN_POSTS_FOR_RELIABLE_MONTH} posts excluded.",
             fontsize=12, fontweight="bold", y=1.02)
fig.text(0.02, 0.5, "Mean Sentiment (-1 to +1)", va="center", rotation="vertical", fontsize=12)
plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig("charts/05_monthly_sentiment_trend.png", bbox_inches="tight")
plt.show()
plt.close()

# ---------------------------------------------------------------------------
# 3. Post volume chart -- small multiples too, same reason
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
axes = axes.flatten()
for ax, brand in zip(axes, brands):
    b = monthly[monthly["brand"] == brand].sort_values("year_month")
    ax.plot(b["year_month"], b["n_posts"], color="#4C72B0", linewidth=1.3)
    ax.axhline(MIN_POSTS_FOR_RELIABLE_MONTH, color="red", linestyle=":", alpha=0.6, linewidth=1)
    ax.set_yscale("log")
    ax.set_title(brand, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Monthly Post Volume by Brand (log scale; red line = reliability threshold)",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.02, 0.5, "Number of Reddit Posts", va="center", rotation="vertical", fontsize=12)
plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig("charts/06_monthly_post_volume.png", bbox_inches="tight")
plt.show()
plt.close()

print("\nSaved charts: 05_monthly_sentiment_trend.png, 06_monthly_post_volume.png")
