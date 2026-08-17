"""
Replaces the binary topic-overlap metric with a weighted one.
Run after 03b_topic_overlap_fix.py (needs df with 'topic_reduced' column).

Binary overlap ("does this topic appear at all in consumer text") is
close to guaranteed near 100% here because Reddit outnumbers brand-official
text ~900:1 -- almost every topic in the whole model has at least one
Reddit post in it by sheer volume, regardless of brand.

This instead asks: of ALL this brand's Reddit posts, what fraction fall
into the SAME topics as that brand's official content? That's a number
that can actually differ meaningfully between brands.
"""

import pandas as pd

df = pd.read_csv("bag_topics.csv")

rows = []
for brand in df["brand"].unique():
    official = df[(df["brand"] == brand) & (df["source_type"] == "brand_official")]
    consumer = df[(df["brand"] == brand) & (df["source_type"] == "reddit")]

    official_topics = set(official["topic_reduced"]) - {-1}
    total_consumer = len(consumer)
    consumer_in_official_topics = consumer["topic_reduced"].isin(official_topics).sum()

    weighted_overlap_pct = consumer_in_official_topics / total_consumer if total_consumer else 0.0

    rows.append({
        "brand": brand,
        "n_official_topics": len(official_topics),
        "total_consumer_posts": total_consumer,
        "consumer_posts_in_official_topics": consumer_in_official_topics,
        "weighted_topic_overlap_pct": round(weighted_overlap_pct, 4),
    })

weighted_df = pd.DataFrame(rows).sort_values("weighted_topic_overlap_pct", ascending=False)
weighted_df.to_csv("weighted_topic_overlap.csv", index=False)

print("Weighted topic overlap (% of a brand's Reddit conversation that falls")
print("into the same topics as that brand's official content):\n")
print(weighted_df.to_string(index=False))
print("\nSaved: weighted_topic_overlap.csv")
