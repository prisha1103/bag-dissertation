"""
Emotion analysis via transformer classifier -- replaces the unreliable
NRCLex dependency entirely. Same pattern as your sentiment pipeline, which
you already confirmed works.

Model: j-hartmann/emotion-english-distilroberta-base
Trained on multiple emotion datasets, widely used and well-documented.
Returns 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.

Methodology note to write up: NRC Emotion Lexicon (via the nrclex package)
was initially planned but abandoned after the installed package version
was found to be missing core functionality (affect_frequencies and related
methods were not present on the installed NRCLex object, confirmed via
direct attribute inspection). A transformer-based emotion classifier was
used instead, consistent with the transformer-based approach already used
for sentiment analysis in this pipeline.

Run in Colab, GPU runtime.
Input:  bag_sentiment_emotion.csv (has text + sentiment columns)
Output: bag_sentiment_emotion.csv (overwritten with emotion columns added)
        emotion_summary.csv (brand x source_type x emotion)
"""

import pandas as pd
import torch
from transformers import pipeline

df = pd.read_csv("bag_sentiment_emotion.csv")
df["text"] = df["text"].astype(str)

# drop any leftover broken NRC columns from the earlier failed attempt
OLD_NRC_COLS = ["fear", "anger", "anticip", "anticipation", "trust", "surprise",
                 "positive", "negative", "sadness", "disgust", "joy"]
df = df.drop(columns=[c for c in OLD_NRC_COLS if c in df.columns], errors="ignore")

device = 0 if torch.cuda.is_available() else -1
emotion_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device,
    top_k=None,          # return scores for ALL emotions, not just top 1
    truncation=True,
    max_length=512,
)

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def batch_emotions(texts, batch_size=64):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out = emotion_pipe(batch)  # list of lists of {label, score} dicts
        for item_scores in out:
            row = {d["label"]: d["score"] for d in item_scores}
            all_scores.append(row)
        if i % (batch_size * 20) == 0:
            print(f"  emotion: {i}/{len(texts)}")
    return all_scores

print(f"Running emotion classification on {len(df)} texts...")
emotion_rows = batch_emotions(df["text"].tolist())
emotion_df = pd.DataFrame(emotion_rows)[EMOTIONS]

print("\nSanity check -- emotion means (should NOT all be 0 or identical):")
print(emotion_df.mean().round(4))

df = pd.concat([df.reset_index(drop=True), emotion_df], axis=1)
df.to_csv("bag_sentiment_emotion.csv", index=False)

emotion_summary = df.groupby(["brand", "source_type"])[EMOTIONS].mean().reset_index()
emotion_summary.to_csv("emotion_summary.csv", index=False)

print("\nEmotion summary by brand x source_type:")
print(emotion_summary.round(4).to_string(index=False))
print("\nSaved: bag_sentiment_emotion.csv, emotion_summary.csv")
print("Now go re-run 04_composite_bag_score.py -- update its EMOTIONS list to:")
print(EMOTIONS)
