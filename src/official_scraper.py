"""
Brand-official language collector -- scrapes brand website copy
(About pages, product descriptions, campaign/editorial pages) as the
"brand voice" side of the BAG comparison.

You must fill in `official_urls` per brand in config.py first.
"""

import time
import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

from config import BRANDS

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)


def _extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    paragraphs = soup.find_all(["p", "h1", "h2", "h3"])
    texts = []
    for p in paragraphs:
        t = p.get_text(strip=True)
        if t and len(t.split()) >= 4:
            texts.append(t)
    return " ".join(texts)


def collect_official_data(brand_name):
    if brand_name not in BRANDS:
        raise ValueError("Unknown brand: " + brand_name)

    brand_cfg = BRANDS[brand_name]
    urls = brand_cfg.get("official_urls", [])

    if not urls:
        print("[" + brand_name + "] No official_urls configured yet -- skipping.")
        return pd.DataFrame(columns=["brand", "category", "source_type", "date", "text", "rating", "url"])

    rows = []
    for url in urls:
        try:
            resp = scraper.get(url, timeout=20)
            if resp.status_code != 200:
                print("[" + brand_name + "] " + url + " failed (status " + str(resp.status_code) + ")")
                continue
            text = _extract_main_text(resp.text)
            if text:
                rows.append({
                    "brand": brand_name, "category": brand_cfg["category"],
                    "source_type": "brand_official", "date": None,
                    "text": text, "rating": None, "url": url,
                })
            else:
                print("[" + brand_name + "] " + url + " returned no usable text")
        except Exception as e:
            print("[" + brand_name + "] " + url + " FAILED: " + str(e))
        time.sleep(2)

    df = pd.DataFrame(rows)
    print("[" + brand_name + "] Official: collected " + str(len(df)) + " rows from " + str(len(urls)) + " URLs")
    return df


if __name__ == "__main__":
    import sys
    brand = sys.argv[1] if len(sys.argv) > 1 else "Supreme"
    df = collect_official_data(brand)
    df.to_csv("data_raw/" + brand + "_official.csv", index=False)
    print(df.head())
