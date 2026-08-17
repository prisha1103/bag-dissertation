"""
Day 5b (extension): Aspect-Based Sentiment Analysis.
Reuses the same RoBERTa sentiment pipeline you already ran -- no new model
needed. The only new piece is tagging WHICH aspect(s) a text mentions via
keyword matching, then computing sentiment per aspect instead of overall.

Run in Colab, GPU runtime. Needs: bag_sentiment_emotion.csv (has text +
sentiment columns already) OR bag_corpus_clean.csv if starting fresh.

Output: aspect_sentiment_summary.csv (brand x source_type x aspect)
"""

import pandas as pd
import torch
from transformers import pipeline

df = pd.read_csv("bag_sentiment_emotion.csv")  # already has sentiment_signed
df["text"] = df["text"].astype(str)

# ---------------------------------------------------------------------------
# 1. Define aspects with keyword sets.
#    NOTE: keyword lists are a starting point -- skim a sample of matched
#    texts per aspect and adjust; this is normal ABSA practice at this
#    scope (full aspect extraction models are out of reach in the time
#    you have, and a documented, reasoned keyword lexicon is a legitimate,
#    citable lightweight ABSA approach for a dissertation of this size).
# ---------------------------------------------------------------------------
ASPECTS = {
    "price_value": ["price", "expensive", "cheap", "worth", "value", "overpriced",
                     "cost", "affordable", "money", "pricey"],
    "exclusivity": ["exclusive", "limited", "rare", "hype", "drop", "sold out",
                     "scarce", "hard to get", "waitlist"],
    "quality": ["quality", "craftsmanship", "material", "stitching", "fabric",
                 "well made", "cheaply made", "durable", "falling apart", "authentic"],
    "originality": ["original", "unique", "copy", "knockoff", "design", "creative",
                     "innovative", "generic", "boring design"],
    "resale_culture": ["resell", "resale", "flip", "bin", "cop", "retail price",
                         "stockx", "grailed", "profit"],
    "overexposure": ["everywhere", "overexposed", "overrated", "basic", "played out",
                       "oversaturated", "too popular", "mainstream now"],
    "heritage_history": ["heritage", "history", "founded", "legacy", "tradition",
                           "since 19", "iconic", "classic"],
}

def tag_aspects(text):
    text_lower = text.lower()
    return [a for a, kws in ASPECTS.items() if any(kw in text_lower for kw in kws)]

print("Tagging aspects (keyword matching, fast, no model needed)...")
df["aspects"] = df["text"].apply(tag_aspects)

# explode so each aspect mention becomes its own row for groupby
exploded = df.explode("aspects").dropna(subset=["aspects"])
print(f"{len(exploded)} aspect-tagged rows from {len(df)} total texts "
      f"({exploded['text'].nunique()} unique texts mention at least one aspect)")

# ---------------------------------------------------------------------------
# 2. Aggregate sentiment per brand x source_type x aspect
#    (reuses sentiment_signed column already computed by 02_sentiment_emotion.py
#    -- no re-running the transformer model needed)
# ---------------------------------------------------------------------------
aspect_summary = (
    exploded.groupby(["brand", "source_type", "aspects"])["sentiment_signed"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"aspects": "aspect", "mean": "mean_sentiment"})
)
aspect_summary.to_csv("aspect_sentiment_summary.csv", index=False)

print("\nAspect sentiment summary (sample):")
print(aspect_summary.head(20).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Aspect-level BAG gap: brand_official vs reddit sentiment gap PER ASPECT
#    (only meaningful where brand_official text mentions that aspect at all --
#    with 5 rows per brand, expect many brand/aspect combos with 0 mentions)
# ---------------------------------------------------------------------------
pivot = aspect_summary.pivot_table(
    index=["brand", "aspect"], columns="source_type", values="mean_sentiment"
).reset_index()
if "brand_official" in pivot.columns and "reddit" in pivot.columns:
    pivot["aspect_gap"] = (pivot["brand_official"] - pivot["reddit"]).abs()
    pivot.to_csv("aspect_bag_gap.csv", index=False)
    print("\nAspect-level brand vs consumer sentiment gap (where both sides have coverage):")
    print(pivot.dropna(subset=["aspect_gap"]).sort_values("aspect_gap", ascending=False).to_string(index=False))

print("\nSaved: aspect_sentiment_summary.csv, aspect_bag_gap.csv")
