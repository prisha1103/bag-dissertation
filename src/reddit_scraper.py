import time
import datetime
import requests
import pandas as pd

from config import BRANDS, MAX_REDDIT_POSTS_PER_BRAND, MAX_REDDIT_COMMENTS_PER_POST

BASE_URL = "https://arctic-shift.photon-reddit.com"
PAGE_SIZE = 100


def _get_with_retry(url, params, max_retries=4):
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp
        wait = 2 ** attempt  # 1, 2, 4, 8 seconds
        print("  request failed (" + str(resp.status_code) + "), retrying in " + str(wait) + "s...")
        time.sleep(wait)
    return resp  # give up after max_retries, return last (failed) response


def _fetch_posts(term, subreddit):
    posts = []
    before = None
    while len(posts) < MAX_REDDIT_POSTS_PER_BRAND:
        params = {"query": term, "sort": "desc", "limit": PAGE_SIZE}
        if subreddit:
            params["subreddit"] = subreddit
        if before:
            params["before"] = before
        resp = _get_with_retry(BASE_URL + "/api/posts/search", params)
        if resp.status_code != 200:
            print("  [posts] gave up after retries, stopping this subreddit")
            break
        batch = resp.json().get("data", [])
        if not batch:
            break
        posts.extend(batch)
        before = batch[-1]["created_utc"]
        time.sleep(1.5)  # slower pace, be nice to the free API
    return posts[:MAX_REDDIT_POSTS_PER_BRAND]


def _fetch_comments_for_post(post_id):
    params = {"link_id": post_id, "limit": MAX_REDDIT_COMMENTS_PER_POST}
    resp = _get_with_retry(BASE_URL + "/api/comments/search", params)
    if resp.status_code != 200:
        return []
    return resp.json().get("data", [])


def collect_reddit_data(brand_name):
    if brand_name not in BRANDS:
        raise ValueError("Unknown brand: " + brand_name)
    brand_cfg = BRANDS[brand_name]
    rows = []
    for term in brand_cfg["reddit_search_terms"]:
        for sub in brand_cfg["reddit_subreddits"]:
            subreddit = None if sub == "all" else sub
            print("[" + brand_name + "] searching '" + term + "' in r/" + sub + "...")
            posts = _fetch_posts(term, subreddit)
            for post in posts:
                post_date = datetime.datetime.utcfromtimestamp(post["created_utc"]).isoformat()
                post_text = (post.get("title", "") + " " + post.get("selftext", "")).strip()
                if post_text:
                    rows.append({
                        "brand": brand_name, "category": brand_cfg["category"],
                        "source_type": "reddit", "date": post_date, "text": post_text,
                        "rating": post.get("score", 0),
                        "url": "https://reddit.com/comments/" + post["id"],
                    })
                comments = _fetch_comments_for_post(post["id"])
                for c in comments:
                    body = c.get("body", "")
                    if body and body not in ("[deleted]", "[removed]"):
                        rows.append({
                            "brand": brand_name, "category": brand_cfg["category"],
                            "source_type": "reddit",
                            "date": datetime.datetime.utcfromtimestamp(c["created_utc"]).isoformat(),
                            "text": body, "rating": c.get("score", 0),
                            "url": "https://reddit.com/comments/" + post["id"],
                        })
    df = pd.DataFrame(rows)
    print("[" + brand_name + "] Reddit: collected " + str(len(df)) + " rows")
    return df


if __name__ == "__main__":
    import sys
    import os
    brand = sys.argv[1] if len(sys.argv) > 1 else "Supreme"
    df = collect_reddit_data(brand)
    os.makedirs("data_raw", exist_ok=True)
    df.to_csv("data_raw/" + brand + "_reddit.csv", index=False)
    print(df.head())
