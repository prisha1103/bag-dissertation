import time
import random
import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

from config import BRANDS, MAX_TRUSTPILOT_PAGES_PER_BRAND

scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})


def collect_trustpilot_data(brand_name):
    if brand_name not in BRANDS:
        raise ValueError("Unknown brand: " + brand_name)
    brand_cfg = BRANDS[brand_name]
    slug = brand_cfg["trustpilot_slug"]
    rows = []
    for page in range(1, MAX_TRUSTPILOT_PAGES_PER_BRAND + 1):
        url = "https://www.trustpilot.com/review/" + slug + "?page=" + str(page)
        resp = scraper.get(url, timeout=20)
        if resp.status_code != 200:
            print("[" + brand_name + "] stopped at page " + str(page) + " (status " + str(resp.status_code) + ")")
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        review_cards = soup.find_all(attrs={"data-service-review-card-paper": True})
        if not review_cards:
            print("[" + brand_name + "] no more reviews at page " + str(page))
            break
        for card in review_cards:
            text_el = card.find(attrs={"data-service-review-text-typography": True})
            rating_el = card.find("img", alt=lambda a: a and "Rated" in a)
            date_el = card.find("time")
            text = text_el.get_text(strip=True) if text_el else None
            if not text:
                continue
            rating = None
            if rating_el and "Rated" in rating_el["alt"]:
                try:
                    rating = int(rating_el["alt"].split()[1])
                except (IndexError, ValueError):
                    pass
            date = date_el["datetime"] if date_el and date_el.has_attr("datetime") else None
            rows.append({
                "brand": brand_name, "category": brand_cfg["category"],
                "source_type": "trustpilot", "date": date, "text": text,
                "rating": rating, "url": url,
            })
        time.sleep(random.uniform(1.5, 3.0))
    df = pd.DataFrame(rows)
    print("[" + brand_name + "] Trustpilot: collected " + str(len(df)) + " rows")
    return df


if __name__ == "__main__":
    import sys
    brand = sys.argv[1] if len(sys.argv) > 1 else "Supreme"
    df = collect_trustpilot_data(brand)
    df.to_csv("data_raw/" + brand + "_trustpilot.csv", index=False)
    print(df.head())
