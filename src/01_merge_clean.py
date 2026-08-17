"""
Day 1: Merge Reddit (consumer) + Brand-Official (brand) data into one
clean, analysis-ready corpus for the BAG pipeline.

Run this locally or in Colab (no GPU needed for this step).
Input:
  - all_brands_combined.csv   (Reddit data, from run_collection.py)
  - Final_Brand_data.xlsx     (manually filled brand-official copy)
Output:
  - bag_corpus_clean.csv      (single clean dataframe, ready for NLP)
"""

import re
import pandas as pd

SCHEMA_COLUMNS = ["brand", "category", "source_type", "date", "text", "rating", "url"]

# ---------------------------------------------------------------------------
# 1. Load both sources
# ---------------------------------------------------------------------------
reddit = pd.read_csv("all_brands_combined.csv")

official = pd.read_excel("Final_Brand_data.xlsx", sheet_name="Brand Official Content")
official = official.dropna(subset=["text"])
official = official[official["text"].astype(str).str.strip() != ""]

print(f"Loaded: {len(reddit)} Reddit rows, {len(official)} brand-official rows")

# ---------------------------------------------------------------------------
# 2. Basic text cleaning
# ---------------------------------------------------------------------------
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = re.sub(r"http\S+|www\.\S+", " ", t)          # strip URLs
    t = re.sub(r"&amp;|&gt;|&lt;", " ", t)            # stray HTML entities
    t = re.sub(r"\s+", " ", t).strip()                # collapse whitespace
    return t

reddit["text"] = reddit["text"].apply(clean_text)
official["text"] = official["text"].apply(clean_text)

# ---------------------------------------------------------------------------
# 3. Drop noise: exact duplicates + very short, low-signal text
#    (single-word replies, emoji-only reactions, etc. add no value to
#    sentiment/emotion/topic/semantic analysis and just inflate row counts)
# ---------------------------------------------------------------------------
MIN_CHARS = 15

before = len(reddit)
reddit = reddit.drop_duplicates(subset=["text"])
reddit = reddit[reddit["text"].str.len() >= MIN_CHARS]
print(f"Reddit: dropped {before - len(reddit)} rows (duplicates / <{MIN_CHARS} chars) "
      f"-> {len(reddit)} remaining")

# ---------------------------------------------------------------------------
# 4. Combine into single schema
# ---------------------------------------------------------------------------
combined = pd.concat([reddit, official], ignore_index=True).reindex(columns=SCHEMA_COLUMNS)
combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=True)

print("\nFinal corpus composition:")
print(combined.groupby(["brand", "source_type"]).size())
print(f"\nTotal rows: {len(combined)}")

combined.to_csv("bag_corpus_clean.csv", index=False)
print("\nSaved -> bag_corpus_clean.csv")
