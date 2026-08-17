"""
Dimension-Level Gap Analysis.

Turns your existing aspect_bag_gap.csv and authenticity_dimension_summary.csv
into the genuinely novel finding: WHICH dimension drives each brand's BAG
score, and whether the same dimensions matter across luxury vs streetwear.

This is analysis of data you already have -- no new model runs needed.

Run in Colab or locally (no GPU needed).
Needs: aspect_bag_gap.csv, authenticity_dimension_summary.csv,
       bag_composite_scores.csv (for category labels)
"""

import pandas as pd
import numpy as np

CATEGORY = {
    "Chanel": "luxury", "Dior": "luxury", "Gucci": "luxury",
    "Supreme": "streetwear", "Off-White": "streetwear", "Palace": "streetwear",
}

# ---------------------------------------------------------------------------
# 1. Aspect-level gap ranking: for each brand, which aspect has the
#    largest brand-official vs consumer sentiment gap?
# ---------------------------------------------------------------------------
aspect_gap = pd.read_csv("aspect_bag_gap.csv")
aspect_gap["category"] = aspect_gap["brand"].map(CATEGORY)
aspect_gap_clean = aspect_gap.dropna(subset=["aspect_gap"])

print("=== TOP DRIVING ASPECT PER BRAND ===")
top_aspect_per_brand = (
    aspect_gap_clean.sort_values("aspect_gap", ascending=False)
    .groupby("brand")
    .first()[["aspect", "aspect_gap", "category"]]
    .sort_values("aspect_gap", ascending=False)
)
print(top_aspect_per_brand.to_string())
top_aspect_per_brand.to_csv("top_driving_aspect_per_brand.csv")

print("\n=== MEAN ASPECT GAP BY CATEGORY (which aspects matter more for luxury vs streetwear) ===")
category_aspect_means = (
    aspect_gap_clean.groupby(["category", "aspect"])["aspect_gap"]
    .mean()
    .reset_index()
    .sort_values(["category", "aspect_gap"], ascending=[True, False])
)
print(category_aspect_means.to_string(index=False))
category_aspect_means.to_csv("category_aspect_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Authenticity dimension gap: for each brand, compare the % distribution
#    of brand-official vs consumer text across dimensions. Largest absolute
#    % difference = the dimension where brand self-image and consumer
#    perception diverge most.
# ---------------------------------------------------------------------------
auth_dim = pd.read_csv("authenticity_dimension_summary.csv")

pivot = auth_dim.pivot_table(
    index=["brand", "top_dimension"], columns="source_type", values="pct"
).reset_index().fillna(0)
pivot["category"] = pivot["brand"].map(CATEGORY)

if "brand_official" in pivot.columns and "reddit" in pivot.columns:
    pivot["dimension_gap"] = (pivot["brand_official"] - pivot["reddit"]).abs()

    print("\n=== TOP DRIVING AUTHENTICITY DIMENSION PER BRAND ===")
    top_dim_per_brand = (
        pivot.sort_values("dimension_gap", ascending=False)
        .groupby("brand")
        .first()[["top_dimension", "dimension_gap", "category"]]
        .sort_values("dimension_gap", ascending=False)
    )
    print(top_dim_per_brand.to_string())
    top_dim_per_brand.to_csv("top_driving_dimension_per_brand.csv")

    print("\n=== MEAN DIMENSION GAP BY CATEGORY ===")
    category_dim_means = (
        pivot.groupby(["category", "top_dimension"])["dimension_gap"]
        .mean()
        .reset_index()
        .sort_values(["category", "dimension_gap"], ascending=[True, False])
    )
    print(category_dim_means.to_string(index=False))
    category_dim_means.to_csv("category_dimension_comparison.csv", index=False)

    # statistical test: does the SAME dimension rank as "most divergent"
    # across categories, or do luxury/streetwear diverge on different
    # dimensions entirely? (descriptive, not a formal test -- n=3 per
    # category is too small for a reliable significance test here, say so)
    print("\n=== Does the same dimension drive the gap for both categories? ===")
    luxury_top = category_dim_means[category_dim_means["category"] == "luxury"].iloc[0]
    street_top = category_dim_means[category_dim_means["category"] == "streetwear"].iloc[0]
    print(f"Luxury's top-divergence dimension: {luxury_top['top_dimension']} "
          f"(mean gap {luxury_top['dimension_gap']:.3f})")
    print(f"Streetwear's top-divergence dimension: {street_top['top_dimension']} "
          f"(mean gap {street_top['dimension_gap']:.3f})")
    if luxury_top["top_dimension"] == street_top["top_dimension"]:
        print("-> SAME dimension drives divergence in both categories.")
    else:
        print("-> DIFFERENT dimensions drive divergence -- categories diverge "
              "on different aspects of authenticity, not just by degree.")
        print("NOTE: n=3 brands per category -- report this as a descriptive "
              "pattern in your specific sample, not a statistically tested claim.")

print("\nSaved: top_driving_aspect_per_brand.csv, category_aspect_comparison.csv, "
      "top_driving_dimension_per_brand.csv, category_dimension_comparison.csv")
