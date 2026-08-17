"""
Step 6: Generate PPT/dissertation charts directly from your actual result
CSVs -- no hardcoded numbers, so these regenerate correctly if you ever
rerun any upstream step.

Run in Colab or locally (no GPU needed).
Needs: bag_composite_scores.csv, transition_significance_tests.csv
Output: PNG files saved to a 'charts' folder
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
os.makedirs("charts", exist_ok=True)

# ---------------------------------------------------------------------------
# Chart 1: Composite BAG Score by brand
# ---------------------------------------------------------------------------
bag = pd.read_csv("bag_composite_scores.csv", index_col=0)
bag_sorted = bag.sort_values("BAG_score_equal_weight")

colors = ["#4C72B0" if c == "luxury" else "#DD8452" for c in bag_sorted["category"]]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(bag_sorted.index, bag_sorted["BAG_score_equal_weight"], color=colors)
for bar, score in zip(bars, bag_sorted["BAG_score_equal_weight"]):
    ax.text(score + 0.015, bar.get_y() + bar.get_height() / 2, f"{score:.3f}",
            va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("BAG Score (higher = greater brand/consumer language divergence)")
ax.set_title("Composite Brand Authenticity Gap Score by Brand", fontsize=14, fontweight="bold")
ax.set_xlim(0, bag_sorted["BAG_score_equal_weight"].max() * 1.15)
legend_elements = [mpatches.Patch(color="#4C72B0", label="Luxury"),
                    mpatches.Patch(color="#DD8452", label="Streetwear")]
ax.legend(handles=legend_elements, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/01_bag_score_by_brand.png", bbox_inches="tight")
plt.show()
plt.close()
print("Saved 01_bag_score_by_brand.png")

# ---------------------------------------------------------------------------
# Chart 2: Category comparison (luxury vs streetwear)
# ---------------------------------------------------------------------------
cat_stats = bag.groupby("category")["BAG_score_equal_weight"].agg(["mean", "std", "count"])
cat_stats = cat_stats.reindex(["luxury", "streetwear"])  # consistent order

fig, ax = plt.subplots(figsize=(7, 5.5))
bar_colors = ["#4C72B0", "#DD8452"]
bars = ax.bar(cat_stats.index, cat_stats["mean"], yerr=cat_stats["std"], capsize=8,
              color=bar_colors, width=0.55)
for bar, mean in zip(bars, cat_stats["mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2, mean + cat_stats["std"].max() * 0.15,
            f"{mean:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean BAG Score")
n_per_cat = cat_stats["count"].iloc[0]
ax.set_title(f"BAG Score by Brand Category\n(n={n_per_cat} brands per category; error bars = SD)",
             fontsize=14, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/02_category_comparison.png", bbox_inches="tight")
plt.show()
plt.close()
print("Saved 02_category_comparison.png")

# ---------------------------------------------------------------------------
# Charts 3 & 4: Pre/post transition, emotion gap + semantic similarity
# ---------------------------------------------------------------------------
trans = pd.read_csv("transition_significance_tests.csv")
brands = trans["brand"].tolist()
x = np.arange(len(brands))
width = 0.35

def plot_transition_metric(df, metric_prefix, ylabel, title, sig_col,
                            colors, sig_note, filename, ylim=None):
    pre = df[f"{metric_prefix}_pre"].values
    post = df[f"{metric_prefix}_post"].values
    pre_lo = df[f"{metric_prefix}_pre_ci_low"].values
    pre_hi = df[f"{metric_prefix}_pre_ci_high"].values
    post_lo = df[f"{metric_prefix}_post_ci_low"].values
    post_hi = df[f"{metric_prefix}_post_ci_high"].values
    sig = df[sig_col].values

    pre_err = [pre - pre_lo, pre_hi - pre]
    post_err = [post - post_lo, post_hi - post]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width/2, pre, width, yerr=pre_err, capsize=4,
           label="Before transition", color="#8C8C8C")
    ax.bar(x + width/2, post, width, yerr=post_err, capsize=4,
           label="After transition", color=colors)

    y_range = max(pre_hi.max(), post_hi.max()) - min(pre_lo.min(), post_lo.min())
    for i, s in enumerate(sig):
        y_max = max(pre_hi[i], post_hi[i]) + y_range * 0.04
        ax.text(x[i], y_max, s, ha="center", fontsize=13, fontweight="bold",
                color="black" if s != "ns" else "gray")

    ax.set_xticks(x)
    ax.set_xticklabels(brands)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.text(0.5, 1.06, sig_note, transform=ax.transAxes, ha="center",
            fontsize=10, color="dimgray")
    ax.legend(loc="upper left", bbox_to_anchor=(0.68, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"charts/{filename}", bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved {filename}")

sig_note = "95% bootstrap CI; *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"

plot_transition_metric(
    trans, "emotion_gap",
    ylabel="Emotion Gap (distance from brand-official emotion profile)",
    title="Emotion Gap Before vs. After Creative Director Transition",
    sig_col="emotion_gap_sig", colors="#55A868", sig_note=sig_note,
    filename="03_emotion_gap_transition.png",
)

plot_transition_metric(
    trans, "semantic_sim",
    ylabel="Semantic Similarity (centroid cosine similarity)",
    title="Semantic Similarity Before vs. After Creative Director Transition",
    sig_col="semantic_sim_sig", colors="#C44E52", sig_note=sig_note,
    filename="04_semantic_similarity_transition.png",
)

print("\nAll charts saved to the 'charts' folder.")
