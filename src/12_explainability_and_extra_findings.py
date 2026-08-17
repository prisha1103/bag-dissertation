"""
Final analysis pass:
  1. Exact contribution decomposition of the composite BAG score (explainability,
     no model-fitting needed since it's a known linear formula)
  2. Convergent validity: do BAG score, aspect-gap, and dimension-gap rankings agree?
  3. Overclaim indices: originality overclaim, commercial-framing avoidance

Run in Colab or locally (no GPU needed).
Needs: bag_composite_scores.csv, aspect_bag_gap.csv, authenticity_dimension_summary.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

CATEGORY = {
    "Chanel": "luxury", "Dior": "luxury", "Gucci": "luxury",
    "Supreme": "streetwear", "Off-White": "streetwear", "Palace": "streetwear",
}

# ===========================================================================
# 1. EXACT CONTRIBUTION DECOMPOSITION (explainability, no model fitting)
# ===========================================================================
bag = pd.read_csv("bag_composite_scores.csv", index_col=0)
DIMENSIONS = ["sentiment_gap", "emotion_gap", "semantic_gap", "topic_gap"]

def minmax(s):
    return (s - s.min()) / (s.max() - s.min())

norm = bag[DIMENSIONS].apply(minmax)
contributions = norm * 0.25  # equal weight -- exact contribution per dimension
contributions["category"] = bag["category"]
contributions.to_csv("bag_score_contributions.csv")

print("=== EXACT CONTRIBUTION OF EACH DIMENSION TO EACH BRAND'S BAG SCORE ===")
print(contributions.round(4).to_string())

# stacked bar chart: shows exactly WHY each brand scored the way it did
order = bag.sort_values("BAG_score_equal_weight").index
fig, ax = plt.subplots(figsize=(9, 6))
bottom = np.zeros(len(order))
colors = {"sentiment_gap": "#4C72B0", "emotion_gap": "#DD8452",
          "semantic_gap": "#55A868", "topic_gap": "#C44E52"}
for dim in DIMENSIONS:
    vals = contributions.loc[order, dim].values
    ax.barh(order, vals, left=bottom, label=dim.replace("_", " ").title(), color=colors[dim])
    bottom += vals
ax.set_xlabel("Contribution to Composite BAG Score")
ax.set_title("What Drives Each Brand's BAG Score\n(exact linear decomposition, equal weighting)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/07_bag_score_decomposition.png", bbox_inches="tight")
plt.show()
plt.close()
print("\nSaved: bag_score_contributions.csv, charts/07_bag_score_decomposition.png")

# ===========================================================================
# 2. CONVERGENT VALIDITY: do independent methods agree on brand ranking?
# ===========================================================================
print("\n\n=== CONVERGENT VALIDITY ACROSS THREE INDEPENDENT METHODS ===")

# BAG composite score rank
bag_rank = bag["BAG_score_equal_weight"].rank(ascending=False)

# aspect-level: each brand's TOP aspect gap value
aspect_gap = pd.read_csv("aspect_bag_gap.csv").dropna(subset=["aspect_gap"])
top_aspect = aspect_gap.groupby("brand")["aspect_gap"].max()
aspect_rank = top_aspect.rank(ascending=False)

# dimension-level: each brand's TOP authenticity dimension gap value
auth_dim = pd.read_csv("authenticity_dimension_summary.csv")
pivot = auth_dim.pivot_table(index=["brand", "top_dimension"], columns="source_type", values="pct").reset_index().fillna(0)
pivot["dimension_gap"] = (pivot["brand_official"] - pivot["reddit"]).abs()
top_dim = pivot.groupby("brand")["dimension_gap"].max()
dim_rank = top_dim.rank(ascending=False)

convergence = pd.DataFrame({
    "BAG_composite_rank": bag_rank,
    "top_aspect_gap_rank": aspect_rank,
    "top_dimension_gap_rank": dim_rank,
}).dropna()
convergence.to_csv("convergent_validity_ranks.csv")
print(convergence.to_string())

rho_1, p_1 = stats.spearmanr(convergence["BAG_composite_rank"], convergence["top_aspect_gap_rank"])
rho_2, p_2 = stats.spearmanr(convergence["BAG_composite_rank"], convergence["top_dimension_gap_rank"])
rho_3, p_3 = stats.spearmanr(convergence["top_aspect_gap_rank"], convergence["top_dimension_gap_rank"])

print(f"\nBAG score vs top aspect gap:     Spearman rho={rho_1:.3f}, p={p_1:.3f}")
print(f"BAG score vs top dimension gap:  Spearman rho={rho_2:.3f}, p={p_2:.3f}")
print(f"Top aspect gap vs dimension gap: Spearman rho={rho_3:.3f}, p={p_3:.3f}")
print("\nNOTE: n=6 brands -- these correlations are descriptive/exploratory, not")
print("adequately powered for a formal significance claim. Report rho values as")
print("evidence of directional agreement, not as confirmed statistical relationships.")

# ===========================================================================
# 3. OVERCLAIM INDICES
# ===========================================================================
print("\n\n=== ORIGINALITY OVERCLAIM INDEX ===")
print("(brand-official % framed as 'uniqueness/originality' minus consumer % -- ")
print("positive = brand claims originality more than consumers actually frame it)")

orig = auth_dim[auth_dim["top_dimension"] == "uniqueness and originality"]
orig_pivot = orig.pivot(index="brand", columns="source_type", values="pct").fillna(0)
orig_pivot["originality_overclaim"] = orig_pivot.get("brand_official", 0) - orig_pivot.get("reddit", 0)
orig_pivot["category"] = orig_pivot.index.map(CATEGORY)
orig_pivot = orig_pivot.sort_values("originality_overclaim", ascending=False)
print(orig_pivot.round(4).to_string())
orig_pivot.to_csv("originality_overclaim_index.csv")

print("\n\n=== COMMERCIAL-FRAMING AVOIDANCE INDEX ===")
print("(consumer % framed as 'commercial/profit-driven' minus brand-official % --")
print("positive = consumers frame the brand as commercial far more than brand copy admits)")

comm = auth_dim[auth_dim["top_dimension"] == "commercial and profit-driven"]
comm_pivot = comm.pivot(index="brand", columns="source_type", values="pct").fillna(0)
comm_pivot["commercial_avoidance"] = comm_pivot.get("reddit", 0) - comm_pivot.get("brand_official", 0)
comm_pivot["category"] = comm_pivot.index.map(CATEGORY)
comm_pivot = comm_pivot.sort_values("commercial_avoidance", ascending=False)
print(comm_pivot.round(4).to_string())
comm_pivot.to_csv("commercial_avoidance_index.csv")

print("\nSaved: convergent_validity_ranks.csv, originality_overclaim_index.csv, "
      "commercial_avoidance_index.csv")
