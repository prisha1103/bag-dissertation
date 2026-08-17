"""
Day 6: Creative Director Transition Analysis.

IMPORTANT SCOPE NOTE (write this into your Methodology/Limitations):
Only 4 of the 6 brands can be validly included here:
  - Chanel, Dior, Gucci: clean, documented creative-director transitions
    in 2024-2025, with Reddit data extending well before and after.
  - Off-White: Ib Kamara named creative director Feb 2024 (exact day not
    publicly specified in sources found -- using 2024-02-01 as an
    approximation; this is noted as a limitation).
  - Supreme is EXCLUDED: its Reddit corpus only covers Apr-Aug 2026, a
    4-month window entirely AFTER any relevant event, so there is no
    "before" period to compare.
  - Palace is EXCLUDED: no documented creative-director change has
    occurred at Palace itself (founder Lev Tanju remains; his Fila+ role
    is a side project at a different company, not a Palace transition).
  This asymmetry is itself a finding worth discussing: creative-director
  cycling appears to be a structural feature of luxury houses that
  streetwear brands in this sample simply don't exhibit in the same way.

Run in Colab or locally (no GPU needed -- reuses embeddings/scores you
already computed).
Needs: bag_corpus_clean.csv, bag_embeddings.npy, bag_sentiment_emotion.csv,
       bag_topics.csv all in the same folder.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

TRANSITIONS = {
    "Chanel":    pd.Timestamp("2024-12-12", tz="UTC"),  # Blazy appointment announced
    "Dior":      pd.Timestamp("2025-06-02", tz="UTC"),  # Anderson full role confirmed
    "Gucci":     pd.Timestamp("2025-03-13", tz="UTC"),  # Demna appointment announced
    "Off-White": pd.Timestamp("2024-02-01", tz="UTC"),  # Kamara named creative director (approx.)
}

# ---------------------------------------------------------------------------
# 1. Load everything, aligned by original row order (embeddings were saved
#    in the same order as bag_corpus_clean.csv, so we reload that exact
#    file to keep indices matching)
# ---------------------------------------------------------------------------
base = pd.read_csv("bag_corpus_clean.csv")
base["date"] = pd.to_datetime(base["date"], errors="coerce", utc=True)
embeddings = np.load("bag_embeddings.npy")
assert len(base) == len(embeddings), "Row count mismatch between corpus and embeddings!"

sent_emo = pd.read_csv("bag_sentiment_emotion.csv")
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

topics = pd.read_csv("bag_topics.csv")

results = []

for brand, split_date in TRANSITIONS.items():
    print(f"\n=== {brand} (transition: {split_date.date()}) ===")

    brand_mask = base["brand"] == brand
    official_mask = brand_mask & (base["source_type"] == "brand_official")
    reddit_mask = brand_mask & (base["source_type"] == "reddit")
    pre_mask = reddit_mask & (base["date"] < split_date)
    post_mask = reddit_mask & (base["date"] >= split_date)

    n_pre, n_post = pre_mask.sum(), post_mask.sum()
    print(f"  Reddit posts: {n_pre} before, {n_post} after")
    if n_pre < 50 or n_post < 50:
        print(f"  WARNING: very small sample on one side ({n_pre} pre / {n_post} post) "
              f"-- interpret this brand's before/after comparison cautiously")

    # --- semantic similarity: official centroid vs pre/post reddit centroids ---
    official_emb = embeddings[official_mask.values]
    pre_emb = embeddings[pre_mask.values]
    post_emb = embeddings[post_mask.values]
    official_centroid = official_emb.mean(axis=0, keepdims=True)
    pre_sim = cosine_similarity(official_centroid, pre_emb.mean(axis=0, keepdims=True))[0][0]
    post_sim = cosine_similarity(official_centroid, post_emb.mean(axis=0, keepdims=True))[0][0]

    # --- sentiment: mean signed sentiment, pre vs post ---
    brand_sent_emo = sent_emo[sent_emo["brand"] == brand]
    # re-derive pre/post masks on this frame using its own date column
    se_dates = pd.to_datetime(brand_sent_emo["date"], errors="coerce", utc=True)
    se_reddit = brand_sent_emo["source_type"] == "reddit"
    se_pre = se_reddit & (se_dates < split_date)
    se_post = se_reddit & (se_dates >= split_date)
    pre_sentiment = brand_sent_emo.loc[se_pre, "sentiment_signed"].mean()
    post_sentiment = brand_sent_emo.loc[se_post, "sentiment_signed"].mean()

    # --- emotion: Euclidean distance between official vector and pre/post reddit vectors ---
    official_emo_vec = brand_sent_emo.loc[
        brand_sent_emo["source_type"] == "brand_official", EMOTIONS
    ].mean().values
    pre_emo_vec = brand_sent_emo.loc[se_pre, EMOTIONS].mean().values
    post_emo_vec = brand_sent_emo.loc[se_post, EMOTIONS].mean().values
    pre_emo_gap = np.linalg.norm(official_emo_vec - pre_emo_vec)
    post_emo_gap = np.linalg.norm(official_emo_vec - post_emo_vec)

    # --- topic overlap: weighted overlap pre vs post ---
    brand_topics = topics[topics["brand"] == brand]
    t_dates = pd.to_datetime(brand_topics["date"], errors="coerce", utc=True)
    t_reddit = brand_topics["source_type"] == "reddit"
    official_topics = set(
        brand_topics.loc[brand_topics["source_type"] == "brand_official", "topic_reduced"]
    ) - {-1}
    pre_topics_df = brand_topics.loc[t_reddit & (t_dates < split_date)]
    post_topics_df = brand_topics.loc[t_reddit & (t_dates >= split_date)]
    pre_overlap = (pre_topics_df["topic_reduced"].isin(official_topics).sum()
                   / len(pre_topics_df)) if len(pre_topics_df) else np.nan
    post_overlap = (post_topics_df["topic_reduced"].isin(official_topics).sum()
                    / len(post_topics_df)) if len(post_topics_df) else np.nan

    results.append({
        "brand": brand, "transition_date": split_date.date(),
        "n_pre": n_pre, "n_post": n_post,
        "semantic_sim_pre": pre_sim, "semantic_sim_post": post_sim,
        "semantic_sim_change": post_sim - pre_sim,
        "sentiment_pre": pre_sentiment, "sentiment_post": post_sentiment,
        "sentiment_change": post_sentiment - pre_sentiment,
        "emotion_gap_pre": pre_emo_gap, "emotion_gap_post": post_emo_gap,
        "emotion_gap_change": post_emo_gap - pre_emo_gap,
        "topic_overlap_pre": pre_overlap, "topic_overlap_post": post_overlap,
        "topic_overlap_change": post_overlap - pre_overlap,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("creative_director_transition_analysis.csv", index=False)

print("\n\n=== SUMMARY: pre/post creative director transition ===")
print(results_df.round(4).to_string(index=False))
print("\nSaved: creative_director_transition_analysis.csv")
print("\nReminder: Supreme and Palace excluded -- see script docstring for why. "
      "State this explicitly in your Results, don't just omit them silently.")
