"""
Day 5: Composite BAG Score.

Combines all 4 dimensions collected so far into ONE score per brand.
Higher BAG score = greater divergence between brand-official language and
consumer language (i.e. lower perceived authenticity alignment).

Each dimension is converted into a "gap" (0 = identical, bigger = more
divergent), then min-max normalized to 0-1 across the 6 brands so no single
dimension dominates just because of its raw scale, then averaged.

Run this in Colab (or locally, no GPU needed) after you have:
  sentiment_summary.csv
  emotion_summary.csv
  semantic_similarity_summary.csv
  weighted_topic_overlap.csv
all sitting in the same folder.
"""

import pandas as pd
import numpy as np

BRANDS = ["Chanel", "Dior", "Gucci", "Supreme", "Off-White", "Palace"]
CATEGORY = {
    "Chanel": "luxury", "Dior": "luxury", "Gucci": "luxury",
    "Supreme": "streetwear", "Off-White": "streetwear", "Palace": "streetwear",
}

# ---------------------------------------------------------------------------
# 1. Sentiment gap: |mean sentiment(brand_official) - mean sentiment(reddit)|
# ---------------------------------------------------------------------------
sent = pd.read_csv("sentiment_summary.csv")
sent_pivot = sent.pivot(index="brand", columns="source_type", values="mean")
sent_gap = (sent_pivot["brand_official"] - sent_pivot["reddit"]).abs()
sent_gap.name = "sentiment_gap"

# ---------------------------------------------------------------------------
# 2. Emotion gap: Euclidean distance between mean emotion vectors
#    (8 NRC emotions -> one distance number per brand)
# ---------------------------------------------------------------------------
emo = pd.read_csv("emotion_summary.csv")
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
emo_gap_rows = {}
for brand in BRANDS:
    b_row = emo[(emo["brand"] == brand) & (emo["source_type"] == "brand_official")]
    c_row = emo[(emo["brand"] == brand) & (emo["source_type"] == "reddit")]
    if len(b_row) and len(c_row):
        b_vec = b_row[EMOTIONS].values[0]
        c_vec = c_row[EMOTIONS].values[0]
        emo_gap_rows[brand] = np.linalg.norm(b_vec - c_vec)
emo_gap = pd.Series(emo_gap_rows, name="emotion_gap")

# ---------------------------------------------------------------------------
# 3. Semantic gap: 1 - centroid similarity
#    (using centroid rather than pairwise -- pairwise mean is dominated by
#    sentence-length/specificity differences rather than brand alignment;
#    centroid captures overall meaning-space closeness, which is what BAG
#    is conceptually asking)
# ---------------------------------------------------------------------------
sim = pd.read_csv("semantic_similarity_summary.csv").set_index("brand")
sem_gap = (1 - sim["centroid_similarity"])
sem_gap.name = "semantic_gap"

# ---------------------------------------------------------------------------
# 4. Topical gap: 1 - weighted topic overlap
# ---------------------------------------------------------------------------
topic = pd.read_csv("weighted_topic_overlap.csv").set_index("brand")
topic_gap = (1 - topic["weighted_topic_overlap_pct"])
topic_gap.name = "topic_gap"

# ---------------------------------------------------------------------------
# 5. Combine, min-max normalize each dimension 0-1 across the 6 brands,
#    then average with equal weights (default) -- and show a weighted
#    version + sensitivity check so you can justify the choice in your
#    Methodology rather than just asserting equal weighting is "fine."
# ---------------------------------------------------------------------------
bag = pd.concat([sent_gap, emo_gap, sem_gap, topic_gap], axis=1)
bag["category"] = bag.index.map(CATEGORY)

def minmax(s):
    span = s.max() - s.min()
    if span == 0:
        raise ValueError(
            f"Column '{s.name}' is constant across all brands (all values = {s.iloc[0]}). "
            "This means that dimension produced no real signal -- likely a bug upstream "
            "(check the raw gap values before normalizing) rather than a genuine finding. "
            "Fix the upstream computation before trusting the composite score."
        )
    return (s - s.min()) / span

DIMENSIONS = ["sentiment_gap", "emotion_gap", "semantic_gap", "topic_gap"]

# fail loudly here rather than silently producing NaN downstream
for dim in DIMENSIONS:
    print(f"{dim}: min={bag[dim].min():.4f}, max={bag[dim].max():.4f}, "
          f"std={bag[dim].std():.4f}  {'<-- WARNING: near-zero variance!' if bag[dim].std() < 1e-6 else ''}")

norm = bag[DIMENSIONS].apply(minmax)

# equal weighting (default / primary reported score)
bag["BAG_score_equal_weight"] = norm.mean(axis=1)

# sensitivity check: recompute under 3 alternative weighting schemes to show
# the brand RANKING is (or isn't) robust to how you weight the dimensions
weight_schemes = {
    "sentiment_heavy": {"sentiment_gap": 0.4, "emotion_gap": 0.2, "semantic_gap": 0.2, "topic_gap": 0.2},
    "semantic_heavy":  {"sentiment_gap": 0.15, "emotion_gap": 0.15, "semantic_gap": 0.5, "topic_gap": 0.2},
    "topic_heavy":     {"sentiment_gap": 0.15, "emotion_gap": 0.15, "semantic_gap": 0.2, "topic_gap": 0.5},
}
for scheme_name, weights in weight_schemes.items():
    col = f"BAG_score_{scheme_name}"
    bag[col] = sum(norm[dim] * w for dim, w in weights.items())  # noqa: no NaN risk now -- checked above

bag = bag.sort_values("BAG_score_equal_weight", ascending=False)

print("Composite BAG Score (higher = greater brand/consumer language divergence):\n")
print(bag.round(3).to_string())

# Rank correlation across weighting schemes -- if ranks barely move, that's
# a strong "the result is robust to weighting choice" line for your writeup
rank_cols = [c for c in bag.columns if c.startswith("BAG_score")]
rank_corr = bag[rank_cols].rank(ascending=False).corr(method="spearman")
print("\nSpearman rank correlation between weighting schemes (closer to 1.0 = more robust ranking):")
print(rank_corr.round(3))

bag.to_csv("bag_composite_scores.csv")
print("\nSaved: bag_composite_scores.csv")

print("\nBy category:")
print(bag.groupby("category")["BAG_score_equal_weight"].agg(["mean", "std", "count"]).round(3))
