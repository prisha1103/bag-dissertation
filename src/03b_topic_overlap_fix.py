"""
Add-on fix: run this AFTER 03_topics_similarity.py in the same Colab session
(needs df, topic_model, topics still in memory -- or reload bag_topics.csv).

Fixes two things:
1. Excludes topic -1 (BERTopic's "no confident cluster" bucket) from overlap
   counting, since including it makes "both sides have noise" look like
   "both sides share a real topic."
2. Uses BERTopic's built-in reduce_outliers to reassign as many -1 docs as
   possible to their nearest real topic, cutting down that 47% noise rate.
"""

import pandas as pd

# If starting fresh in a new cell/session, reload:
# df = pd.read_csv("bag_topics.csv")
# topics = df["topic"].tolist()

# ---------------------------------------------------------------------------
# 1. Reduce outliers: reassign -1 docs to nearest real topic where possible
#    (uses c-TF-IDF similarity under the hood -- standard BERTopic method)
# ---------------------------------------------------------------------------
print("Outlier rate before reduction:", (df["topic"] == -1).mean().round(3))

new_topics = topic_model.reduce_outliers(df["text"].tolist(), topics, strategy="c-tf-idf")
df["topic_reduced"] = new_topics

print("Outlier rate after reduction:", (df["topic_reduced"] == -1).mean().round(3))

# ---------------------------------------------------------------------------
# 2. Recompute topic overlap using the reduced topics AND excluding -1
#    from the overlap count (any remaining -1 after reduction is genuine
#    noise, not a topic, and shouldn't count as "shared")
# ---------------------------------------------------------------------------
print("\nTopic overlap (excluding noise topic -1):")
overlap_rows = []
for brand in df["brand"].unique():
    official_topics = set(df[(df["brand"] == brand) & (df["source_type"] == "brand_official")]["topic_reduced"]) - {-1}
    consumer_topics = set(df[(df["brand"] == brand) & (df["source_type"] == "reddit")]["topic_reduced"]) - {-1}
    overlap = official_topics & consumer_topics
    overlap_pct = len(overlap) / len(official_topics) if official_topics else 0.0
    overlap_rows.append({
        "brand": brand,
        "official_topics": sorted(official_topics),
        "consumer_topics_sample_count": len(consumer_topics),
        "shared_topics": sorted(overlap),
        "pct_official_topics_shared": round(overlap_pct, 3),
    })
    print(f"  {brand}: official={sorted(official_topics)}, "
          f"shared={sorted(overlap)} ({overlap_pct:.0%} of official topics also used by consumers)")

topic_overlap_df = pd.DataFrame(overlap_rows)
topic_overlap_df.to_csv("topic_overlap_summary.csv", index=False)

df.to_csv("bag_topics.csv", index=False)  # overwrite with topic_reduced column added
print("\nSaved: topic_overlap_summary.csv, updated bag_topics.csv")
