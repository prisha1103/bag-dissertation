import os
import sys
import pandas as pd

from config import BRANDS, SCHEMA_COLUMNS
from reddit_scraper import collect_reddit_data
from trustpilot_scraper import collect_trustpilot_data

RAW_DIR = "data_raw"
os.makedirs(RAW_DIR, exist_ok=True)


def collect_brand(brand_name):
    print("\n=== Collecting " + brand_name + " ===")
    frames = []
    try:
        frames.append(collect_reddit_data(brand_name))
    except Exception as e:
        print("[" + brand_name + "] Reddit FAILED: " + str(e))
    try:
        frames.append(collect_trustpilot_data(brand_name))
    except Exception as e:
        print("[" + brand_name + "] Trustpilot FAILED: " + str(e))
    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    brand_df = pd.concat(frames, ignore_index=True).reindex(columns=SCHEMA_COLUMNS)
    out_path = RAW_DIR + "/" + brand_name.replace(" ", "_") + "_combined.csv"

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if len(brand_df) < len(existing):
            print("[" + brand_name + "] NEW run has FEWER rows (" + str(len(brand_df)) +
                  ") than existing file (" + str(len(existing)) + ") -- keeping existing file, not overwriting")
            return existing

    brand_df.to_csv(out_path, index=False)
    print("[" + brand_name + "] Saved " + str(len(brand_df)) + " rows -> " + out_path)
    return brand_df


def main():
    target_brands = [sys.argv[1]] if len(sys.argv) > 1 else list(BRANDS.keys())
    all_frames = [collect_brand(b) for b in target_brands]
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(RAW_DIR + "/all_brands_combined.csv", index=False)
    print("\n=== DONE ===")
    print("Total rows: " + str(len(combined)))
    print(combined.groupby(["brand", "source_type"]).size())


if __name__ == "__main__":
    main()
