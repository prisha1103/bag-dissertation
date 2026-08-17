"""
Day 3-4: Topic Modeling (BERTopic) + Semantic Similarity (Sentence-BERT)
Run this in Colab with a GPU runtime.

!pip install -q bertopic sentence-transformers umap-learn hdbscan

Input:  bag_corpus_clean.csv (or bag_sentiment_emotion.csv if you want to
        carry sentiment/emotion columns through to the final merged file)
Output: bag_topics.csv               (row-level topic assignment)
        topic_summary.csv            (top words per topic)
        semantic_similarity_summary.csv  (per-brand brand-vs-consumer similarity)
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from umap import UMAP

RANDOM_SEED = 42  # fixed for reproducibility -- BERTopic's UMAP step is
                   # stochastic by default, meaning re-running without a
                   # fixed seed produces DIFFERENT topic clusters (and
                   # therefore different topic_gap values) each time.
                   # State this seed explicitly in your Methodology.

df = pd.read_csv("bag_corpus_clean.csv")
df["text"] = df["text"].astype(str)

# ---------------------------------------------------------------------------
# 1. Embed everything once with Sentence-BERT -- reused for BOTH
#    topic modeling (BERTopic can take precomputed embeddings) and
#    semantic similarity, so we don't embed twice.
# ---------------------------------------------------------------------------
print("Loading Sentence-BERT model...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")  # fast, strong general-purpose model

print("Encoding all texts (this is the slow step, ~few minutes for 25k rows on GPU)...")
embeddings = sbert.encode(
    df["text"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
)
np.save("bag_embeddings.npy", embeddings)  # save so you never have to re-embed

# ---------------------------------------------------------------------------
# 2. Semantic similarity: per brand, compare the CENTROID of brand-official
#    embeddings against the centroid of consumer (reddit) embeddings.
#    This answers "how close is brand language to consumer language in
#    meaning-space overall" -- the core BAG semantic dimension.
# ---------------------------------------------------------------------------
rows = []
for brand in df["brand"].unique():
    brand_mask = (df["brand"] == brand) & (df["source_type"] == "brand_official")
    consumer_mask = (df["brand"] == brand) & (df["source_type"] == "reddit")

    brand_emb = embeddings[brand_mask.values]
    consumer_emb = embeddings[consumer_mask.values]

    if len(brand_emb) == 0 or len(consumer_emb) == 0:
        continue

    brand_centroid = brand_emb.mean(axis=0, keepdims=True)
    consumer_centroid = consumer_emb.mean(axis=0, keepdims=True)
    centroid_sim = cosine_similarity(brand_centroid, consumer_centroid)[0][0]

    # also compute mean pairwise similarity (brand text vs EVERY consumer text)
    # -- more expensive but gives you a distribution, not just one number
    pairwise = cosine_similarity(brand_emb, consumer_emb)
    mean_pairwise_sim = pairwise.mean()
    std_pairwise_sim = pairwise.std()

    rows.append({
        "brand": brand,
        "centroid_similarity": centroid_sim,
        "mean_pairwise_similarity": mean_pairwise_sim,
        "std_pairwise_similarity": std_pairwise_sim,
        "n_brand_official": len(brand_emb),
        "n_consumer": len(consumer_emb),
    })

sim_summary = pd.DataFrame(rows).sort_values("centroid_similarity", ascending=False)
sim_summary.to_csv("semantic_similarity_summary.csv", index=False)
print("\nSemantic similarity (higher = brand and consumer language closer in meaning):")
print(sim_summary)

# ---------------------------------------------------------------------------
# 3. Topic modeling with BERTopic, reusing the same embeddings.
#    NOTE: with only 30 brand-official rows total, per-brand topic modeling
#    on the brand-official side alone is not viable (BERTopic needs a
#    reasonable number of docs to form clusters). Instead: run ONE topic
#    model across the WHOLE corpus (both source types together), then look
#    at which topics each source type/brand concentrates in. This is a
#    more defensible approach given the row-count imbalance -- explain this
#    choice explicitly in your Methodology.
# ---------------------------------------------------------------------------
print("\nFitting BERTopic on full corpus (fixed random seed for reproducibility)...")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                   metric="cosine", random_state=RANDOM_SEED)
topic_model = BERTopic(embedding_model=sbert, umap_model=umap_model,
                        calculate_probabilities=False, verbose=True)
topics, _ = topic_model.fit_transform(df["text"].tolist(), embeddings=embeddings)

df["topic"] = topics
df.to_csv("bag_topics.csv", index=False)

topic_info = topic_model.get_topic_info()
topic_info.to_csv("topic_summary.csv", index=False)
print("\nTop topics found:")
print(topic_info.head(15))

# per brand x source_type: topic distribution overlap
# (what % of each brand's official topics also appear in that brand's consumer topics)
print("\nTopic overlap check (per brand, do official and consumer text share top topics?):")
for brand in df["brand"].unique():
    official_topics = set(df[(df["brand"] == brand) & (df["source_type"] == "brand_official")]["topic"])
    consumer_topics = set(df[(df["brand"] == brand) & (df["source_type"] == "reddit")]["topic"])
    overlap = official_topics & consumer_topics
    print(f"  {brand}: official topics={official_topics}, "
          f"overlap with consumer topics={len(overlap)}")

print("\nSaved: bag_embeddings.npy, semantic_similarity_summary.csv, "
      "bag_topics.csv, topic_summary.csv")
