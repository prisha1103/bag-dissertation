"""
Step 5c: Statistical significance testing for the creative-director
transition analysis. Addresses the imbalanced-sample-size critique directly:
Chanel (3440 vs 242), Dior (3200 vs 222), Gucci (4417 vs 361) are all
lopsided; Off-White (2151 vs 2248) is the one balanced case. A point
difference in means tells you nothing about whether that gap could just be
noise from unequal sample sizes -- these tests do.

Methods used, and why:
  - Sentiment (row-level distribution): Mann-Whitney U test. Non-parametric,
    doesn't assume normality, robust to the sample-size imbalance itself
    (unlike a t-test, which gets more sensitive to imbalance).
  - Topic overlap (binary: in official topics y/n): two-proportion z-test.
    Directly appropriate for comparing two proportions from two sample sizes.
  - Emotion gap & semantic similarity (distances between AGGREGATE vectors,
    not per-post values -- there's no single "emotion gap per post" to run
    a rank test on): bootstrap resampling (2000 iterations). Resample posts
    with replacement within each period, recompute the gap/similarity each
    time, and report a 95% CI plus the bootstrap p-value (proportion of
    resampled differences that cross zero).

Run in Colab or locally (no GPU needed).
Needs: bag_corpus_clean.csv, bag_embeddings.npy, bag_sentiment_emotion.csv,
       bag_topics.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
rng = np.random.default_rng(RANDOM_SEED)

TRANSITIONS = {
    "Chanel":    pd.Timestamp("2024-12-12", tz="UTC"),
    "Dior":      pd.Timestamp("2025-06-02", tz="UTC"),
    "Gucci":     pd.Timestamp("2025-03-13", tz="UTC"),
    "Off-White": pd.Timestamp("2024-02-01", tz="UTC"),
}

base = pd.read_csv("bag_corpus_clean.csv")
base["date"] = pd.to_datetime(base["date"], errors="coerce", utc=True)
embeddings = np.load("bag_embeddings.npy")
sent_emo = pd.read_csv("bag_sentiment_emotion.csv")
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
topics = pd.read_csv("bag_topics.csv")

def bootstrap_gap_ci(official_vec_source, pre_data, post_data, metric_fn, n_boot=N_BOOTSTRAP):
    """
    Generic bootstrap: resample pre_data and post_data (with replacement),
    recompute metric_fn(official_vec_source, resampled_data) each time,
    return arrays of bootstrapped pre/post values.
    """
    pre_vals, post_vals = [], []
    n_pre, n_post = len(pre_data), len(post_data)
    for _ in range(n_boot):
        pre_sample = pre_data[rng.integers(0, n_pre, n_pre)]
        post_sample = post_data[rng.integers(0, n_post, n_post)]
        pre_vals.append(metric_fn(official_vec_source, pre_sample))
        post_vals.append(metric_fn(official_vec_source, post_sample))
    return np.array(pre_vals), np.array(post_vals)

def emotion_gap_metric(official_vec, sample_emo_matrix):
    return np.linalg.norm(official_vec - sample_emo_matrix.mean(axis=0))

def semantic_sim_metric(official_centroid, sample_emb):
    return cosine_similarity(official_centroid, sample_emb.mean(axis=0, keepdims=True))[0][0]

results = []

for brand, split_date in TRANSITIONS.items():
    print(f"\n=== {brand} (transition: {split_date.date()}) ===")
    brand_mask = base["brand"] == brand
    official_mask = brand_mask & (base["source_type"] == "brand_official")
    reddit_mask = brand_mask & (base["source_type"] == "reddit")
    pre_mask = reddit_mask & (base["date"] < split_date)
    post_mask = reddit_mask & (base["date"] >= split_date)
    n_pre, n_post = pre_mask.sum(), post_mask.sum()
    print(f"  n_pre={n_pre}, n_post={n_post} (imbalance ratio: {n_pre/n_post:.1f}:1)")

    # --- 1. Sentiment: Mann-Whitney U on row-level sentiment_signed ---
    brand_sent_emo = sent_emo[sent_emo["brand"] == brand].copy()
    se_dates = pd.to_datetime(brand_sent_emo["date"], errors="coerce", utc=True)
    se_reddit = brand_sent_emo["source_type"] == "reddit"
    se_pre = se_reddit & (se_dates < split_date)
    se_post = se_reddit & (se_dates >= split_date)
    pre_sent_vals = brand_sent_emo.loc[se_pre, "sentiment_signed"].dropna().values
    post_sent_vals = brand_sent_emo.loc[se_post, "sentiment_signed"].dropna().values
    u_stat, sent_p = stats.mannwhitneyu(pre_sent_vals, post_sent_vals, alternative="two-sided")

    # --- 2. Topic overlap: two-proportion z-test ---
    brand_topics = topics[topics["brand"] == brand]
    t_dates = pd.to_datetime(brand_topics["date"], errors="coerce", utc=True)
    t_reddit = brand_topics["source_type"] == "reddit"
    official_topics = set(
        brand_topics.loc[brand_topics["source_type"] == "brand_official", "topic_reduced"]
    ) - {-1}
    pre_topics_df = brand_topics.loc[t_reddit & (t_dates < split_date)]
    post_topics_df = brand_topics.loc[t_reddit & (t_dates >= split_date)]
    pre_hits = pre_topics_df["topic_reduced"].isin(official_topics).sum()
    post_hits = post_topics_df["topic_reduced"].isin(official_topics).sum()
    n1, n2 = len(pre_topics_df), len(post_topics_df)
    p1, p2 = pre_hits / n1, post_hits / n2
    p_pool = (pre_hits + post_hits) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p2 - p1) / se if se > 0 else np.nan
    topic_p = 2 * (1 - stats.norm.cdf(abs(z_stat))) if se > 0 else np.nan

    # --- 3. Emotion gap: bootstrap CI ---
    official_emo_vec = brand_sent_emo.loc[
        brand_sent_emo["source_type"] == "brand_official", EMOTIONS
    ].mean().values
    pre_emo_matrix = brand_sent_emo.loc[se_pre, EMOTIONS].dropna().values
    post_emo_matrix = brand_sent_emo.loc[se_post, EMOTIONS].dropna().values
    boot_pre_emo, boot_post_emo = bootstrap_gap_ci(
        official_emo_vec, pre_emo_matrix, post_emo_matrix, emotion_gap_metric
    )
    emo_ci_pre = np.percentile(boot_pre_emo, [2.5, 97.5])
    emo_ci_post = np.percentile(boot_post_emo, [2.5, 97.5])
    emo_diff = boot_post_emo - boot_pre_emo
    emo_boot_p = 2 * min((emo_diff > 0).mean(), (emo_diff < 0).mean())

    # --- 4. Semantic similarity: bootstrap CI ---
    official_emb = embeddings[official_mask.values]
    official_centroid = official_emb.mean(axis=0, keepdims=True)
    pre_emb_matrix = embeddings[pre_mask.values]
    post_emb_matrix = embeddings[post_mask.values]
    boot_pre_sim, boot_post_sim = bootstrap_gap_ci(
        official_centroid, pre_emb_matrix, post_emb_matrix, semantic_sim_metric
    )
    sim_ci_pre = np.percentile(boot_pre_sim, [2.5, 97.5])
    sim_ci_post = np.percentile(boot_post_sim, [2.5, 97.5])
    sim_diff = boot_post_sim - boot_pre_sim
    sim_boot_p = 2 * min((sim_diff > 0).mean(), (sim_diff < 0).mean())

    def sig_marker(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    results.append({
        "brand": brand, "n_pre": n_pre, "n_post": n_post,
        "sentiment_mannwhitney_p": sent_p, "sentiment_sig": sig_marker(sent_p),
        "topic_overlap_pre_pct": round(p1, 4), "topic_overlap_post_pct": round(p2, 4),
        "topic_overlap_ztest_p": topic_p, "topic_overlap_sig": sig_marker(topic_p),
        # numeric CI bounds (for plotting) alongside display strings
        "emotion_gap_pre": boot_pre_emo.mean(), "emotion_gap_post": boot_post_emo.mean(),
        "emotion_gap_pre_ci_low": emo_ci_pre[0], "emotion_gap_pre_ci_high": emo_ci_pre[1],
        "emotion_gap_post_ci_low": emo_ci_post[0], "emotion_gap_post_ci_high": emo_ci_post[1],
        "emotion_gap_pre_95CI": f"[{emo_ci_pre[0]:.3f}, {emo_ci_pre[1]:.3f}]",
        "emotion_gap_post_95CI": f"[{emo_ci_post[0]:.3f}, {emo_ci_post[1]:.3f}]",
        "emotion_gap_bootstrap_p": emo_boot_p, "emotion_gap_sig": sig_marker(emo_boot_p),
        "semantic_sim_pre": boot_pre_sim.mean(), "semantic_sim_post": boot_post_sim.mean(),
        "semantic_sim_pre_ci_low": sim_ci_pre[0], "semantic_sim_pre_ci_high": sim_ci_pre[1],
        "semantic_sim_post_ci_low": sim_ci_post[0], "semantic_sim_post_ci_high": sim_ci_post[1],
        "semantic_sim_pre_95CI": f"[{sim_ci_pre[0]:.3f}, {sim_ci_pre[1]:.3f}]",
        "semantic_sim_post_95CI": f"[{sim_ci_post[0]:.3f}, {sim_ci_post[1]:.3f}]",
        "semantic_sim_bootstrap_p": sim_boot_p, "semantic_sim_sig": sig_marker(sim_boot_p),
    })

    print(f"  Sentiment: Mann-Whitney p={sent_p:.4f} ({sig_marker(sent_p)})")
    print(f"  Topic overlap: {p1:.1%} -> {p2:.1%}, z-test p={topic_p:.4f} ({sig_marker(topic_p)})")
    print(f"  Emotion gap: 95% CI pre {emo_ci_pre.round(3)}, post {emo_ci_post.round(3)}, "
          f"bootstrap p={emo_boot_p:.4f} ({sig_marker(emo_boot_p)})")
    print(f"  Semantic sim: 95% CI pre {sim_ci_pre.round(3)}, post {sim_ci_post.round(3)}, "
          f"bootstrap p={sim_boot_p:.4f} ({sig_marker(sim_boot_p)})")

results_df = pd.DataFrame(results)
results_df.to_csv("transition_significance_tests.csv", index=False)
print("\n\n=== FULL RESULTS TABLE ===")
print(results_df.to_string(index=False))
print("\nSaved: transition_significance_tests.csv")
print("\nSignificance key: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
print("\nIMPORTANT: report exact p-values in your dissertation, not just stars --")
print("examiners generally prefer seeing the actual number.")
