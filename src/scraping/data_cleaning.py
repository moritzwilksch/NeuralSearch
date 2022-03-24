import os
import re

import polars as pl
from pymongo import MongoClient

db = MongoClient(
    f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME')}:"
    f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD')}@"
    f"{os.getenv('MONGO_IP')}:27017",
    authSource="admin",
)["ibkr"]


articles = db.articles.aggregate(
    [
        {
            "$group": {
                "_id": "$url",
                "body": {"$first": "$body"},
                "title": {"$first": "$title"},
            }
        },
    ],
)


def clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub("简体中文 繁體中文 Français Deutsch Italiano 日本語 Русский Español", "", text)
    text = re.sub("简体中文 繁體中文 Français Magyar Italiano 日本語 Русский Español ", "", text)
    text = re.sub("\s+", " ", text)
    return text


articles = [
    {"_id": a.get("_id"), "title": a.get("title"), "body": clean(a.get("body"))}
    for a in articles
]


df = pl.from_records(articles).rename({"_id": "url"})
df.write_parquet("data/ibkr_articles.parquet")

# print(f"\n{'-'*80}\n".join(articles))

# with open("data/ibkr_articles.txt", "w") as f:
#     f.write("\n".join(articles))
