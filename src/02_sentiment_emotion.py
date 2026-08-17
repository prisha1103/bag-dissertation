"""
Day 2-3: Sentiment (RoBERTa) analysis.
NOTE: emotion analysis is now a SEPARATE step (02d_emotion_transformer.py)
using a transformer classifier -- NRCLex was abandoned after it turned out
to be broken in this environment. Don't add NRCLex back in here.

Run this in Colab with a GPU runtime (Runtime > Change runtime type > T4 GPU).

!pip install -q transformers sentencepiece

Input:  bag_corpus_clean.csv
Output: bag_sentiment_emotion.csv (row-level scores)
        sentiment_summary.csv (per brand x source_type)
"""

import pandas as pd
import torch
from transformers import pipeline

df = pd.read_csv("bag_corpus_clean.csv")
df["text"] = df["text"].astype(str)

device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

# ---------------------------------------------------------------------------
# 1. Sentiment: cardiffnlp RoBERTa (trained for social-media-style text,
#    which fits Reddit well and is a standard, citable choice)
# ---------------------------------------------------------------------------
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device,
    truncation=True,
    max_length=512,
)

def batch_sentiment(texts, batch_size=64):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = sentiment_pipe(batch)
        results.extend(out)
        if i % (batch_size * 20) == 0:
            print(f"  sentiment: {i}/{len(texts)}")
    return results

sent_results = batch_sentiment(df["text"].tolist())
df["sentiment_label"] = [r["label"] for r in sent_results]
df["sentiment_score"] = [r["score"] for r in sent_results]

# map to a single -1..+1 scale for easy comparison / composite scoring later
label_map = {"negative": -1, "neutral": 0, "positive": 1}
df["sentiment_signed"] = df["sentiment_label"].str.lower().map(label_map) * df["sentiment_score"]

# ---------------------------------------------------------------------------
# 2. Save row-level + summary tables
#    (emotion columns get added by 02d_emotion_transformer.py, run next)
# ---------------------------------------------------------------------------
df.to_csv("bag_sentiment_emotion.csv", index=False)

sentiment_summary = (
    df.groupby(["brand", "source_type"])["sentiment_signed"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
sentiment_summary.to_csv("sentiment_summary.csv", index=False)

print("\nSentiment summary (mean signed sentiment, -1 to +1):")
print(sentiment_summary)
print("\nSaved: bag_sentiment_emotion.csv, sentiment_summary.csv")
print("Now run 02d_emotion_transformer.py next to add emotion columns.")
