from pymongo import MongoClient
import re
import os

db = MongoClient(
    f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME')}:"
    f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD')}@"
    f"{os.getenv('MONGO_IP')}:27017",
    authSource="admin",
)["ibkr"]


articles = db.articles.aggregate(
    [{"$group": {"_id": "$url", "body": {"$first": "$body"}}}]
)


def clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub("简体中文 繁體中文 Français Deutsch Italiano 日本語 Русский Español", "", text)
    text = re.sub("简体中文 繁體中文 Français Magyar Italiano 日本語 Русский Español ", "", text)
    text = re.sub("\s+", " ", text)
    return text


articles = [clean(a.get("body")) for a in articles]
print(f"\n{'-'*80}\n".join(articles))

with open("data/ibkr_articles.txt", "w") as f:
    f.write("\n".join(articles))
