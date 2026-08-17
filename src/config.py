BRANDS = {
    "Chanel": {
        "reddit_subreddits": ["femalefashionadvice", "malefashionadvice"],
        "reddit_search_terms": ["Chanel"],
        "trustpilot_slug": "chanel.com",
        "official_urls": [],
        "category": "luxury",
    },
    "Dior": {
        "reddit_subreddits": ["femalefashionadvice", "malefashionadvice"],
        "reddit_search_terms": ["Dior"],
        "trustpilot_slug": "dior.com",
        "official_urls": [],
        "category": "luxury",
    },
    "Gucci": {
        "reddit_subreddits": ["femalefashionadvice", "malefashionadvice"],
        "reddit_search_terms": ["Gucci"],
        "trustpilot_slug": "gucci.com",
        "official_urls": [],
        "category": "luxury",
    },
    "Supreme": {
        "reddit_subreddits": ["Supreme", "SupremeClothing"],
        "reddit_search_terms": ["Supreme"],
        "trustpilot_slug": "supremenewyork.com",
        "official_urls": [],
        "category": "streetwear",
    },
    "Off-White": {
        "reddit_subreddits": ["streetwear", "malefashionadvice"],
        "reddit_search_terms": ["Off-White", "OffWhite"],
        "trustpilot_slug": "off---white.com",
        "official_urls": [],
        "category": "streetwear",
    },
    "Palace": {
        "reddit_subreddits": ["streetwear", "sneakers", "malefashionadvice"],
        "reddit_search_terms": ["Palace Skateboards", "Palace"],
        "trustpilot_slug": "palaceskateboards.com",
        "official_urls": [],
        "category": "streetwear",
    },
}

SCHEMA_COLUMNS = [
    "brand", "category", "source_type", "date", "text", "rating", "url",
]

MAX_REDDIT_POSTS_PER_BRAND = 500
MAX_REDDIT_COMMENTS_PER_POST = 20
MAX_TRUSTPILOT_PAGES_PER_BRAND = 20
