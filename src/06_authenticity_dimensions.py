"""
Day 5c (extension): Authenticity Dimension Classification via Zero-Shot.

Your professor's suggestion of an "authenticity dimension classifier"
normally implies training a supervised model, which needs labeled data
you don't have and days you don't have. Zero-shot classification gets you
the same academic framing without training: the model (BART fine-tuned on
NLI) scores how well each candidate label describes a text, with NO
brand-specific training required. This is a well-established, citable
NLP technique (Yin et al., 2019) -- name it as such in your Methodology.

Run in Colab, GPU runtime.
!pip install -q transformers

Input:  bag_sentiment_emotion.csv (or bag_corpus_clean.csv)
Output: authenticity_dimensions.csv (row-level top dimension + score)
        authenticity_dimension_summary.csv (brand x source_type x dimension %)

WARNING: zero-shot classification is much slower per-item than sentiment
classification (it runs the model once per candidate label per text).
For 25k rows this can take a while even on GPU -- the script SAMPLES the
Reddit side by default (see SAMPLE_SIZE) to keep runtime reasonable while
still being large enough for a valid per-brand comparison. All 30
brand-official rows are always included in full.
"""

import pandas as pd
import torch
from transformers import pipeline

df = pd.read_csv("bag_sentiment_emotion.csv")
df["text"] = df["text"].astype(str)

# ---------------------------------------------------------------------------
# 1. Sample Reddit rows for runtime reasons; keep ALL brand-official rows.
#    Stratified by brand so every brand gets equal representation.
#    Increase SAMPLE_SIZE if you have time to spare; state your chosen
#    sample size and reasoning in your Methodology either way.
# ---------------------------------------------------------------------------
SAMPLE_SIZE_PER_BRAND = 300  # 300 x 6 brands = 1800 Reddit rows classified

official = df[df["source_type"] == "brand_official"]
reddit_sampled = (
    df[df["source_type"] == "reddit"]
    .groupby("brand", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), SAMPLE_SIZE_PER_BRAND), random_state=42))
)
work_df = pd.concat([official, reddit_sampled], ignore_index=True)
print(f"Classifying {len(work_df)} rows ({len(official)} brand-official + "
      f"{len(reddit_sampled)} sampled Reddit)")

# ---------------------------------------------------------------------------
# 2. Zero-shot classification with authenticity-relevant candidate labels
# ---------------------------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
)

AUTHENTICITY_DIMENSIONS = [
    "heritage and history",
    "uniqueness and originality",
    "symbolism and status",
    "quality and craftsmanship commitment",
    "commercial and profit-driven",
]

def classify_batch(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for t in batch:
            out = classifier(t, AUTHENTICITY_DIMENSIONS, multi_label=False)
            results.append({
                "top_dimension": out["labels"][0],
                "top_dimension_score": out["scores"][0],
            })
        if i % (batch_size * 10) == 0:
            print(f"  classifying: {i}/{len(texts)}")
    return results

class_results = classify_batch(work_df["text"].tolist())
class_df = pd.DataFrame(class_results)
work_df = pd.concat([work_df.reset_index(drop=True), class_df], axis=1)

work_df.to_csv("authenticity_dimensions.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Summarize: what % of each brand's text (official vs consumer) falls
#    into each authenticity dimension
# ---------------------------------------------------------------------------
summary = (
    work_df.groupby(["brand", "source_type", "top_dimension"])
    .size()
    .reset_index(name="count")
)
summary["pct"] = summary.groupby(["brand", "source_type"])["count"].transform(lambda x: x / x.sum())
summary.to_csv("authenticity_dimension_summary.csv", index=False)

print("\nAuthenticity dimension distribution (sample):")
print(summary.head(20).to_string(index=False))
print("\nSaved: authenticity_dimensions.csv, authenticity_dimension_summary.csv")
