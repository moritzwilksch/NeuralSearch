import os
import time

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient


class ArticleLinkCollector:
    def __init__(
        self, base_url: str, start_page_idx: int = 0, end_page_index: int = 15
    ) -> None:

        self.db = MongoClient(
            f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME')}:"
            f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD')}@"
            f"{os.getenv('MONGO_IP')}:27017",
            authSource="admin",
        )["ibkr"]

        self.base_url = base_url
        self.start_page_idx = start_page_idx
        self.end_page_index = end_page_index

    def collect_one_page(self, page_idx: int) -> None:
        r = requests.get(self.base_url, params={"page": page_idx})
        soup = BeautifulSoup(r.text, "lxml")
        links = soup.find_all("a")
        links = [
            l
            for l in links
            if l.attrs.get("href").startswith("/article/")
            and not l.text.startswith("KB")
        ]
        self.db.articles.insert_many(
            [
                {
                    "url": f"https://ibkr.info{l.attrs.get('href')}",
                    "title": l.text,
                    "body": None,
                }
                for l in links
            ]
        )
        return len(links)

    def run(self):
        for i in range(self.start_page_idx, self.end_page_index + 1):
            print(f"Collecting page {i}...")
            self.collect_one_page(i)
            time.sleep(3)


class ArticleBodyCollector:
    def __init__(self) -> None:
        self.db = MongoClient(
            f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME')}:"
            f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD')}@"
            f"{os.getenv('MONGO_IP')}:27017",
            authSource="admin",
        )["ibkr"]
        self.articles = self.db.articles.find({"body": None})

    def run(self):
        for article in self.articles:
            print(f"Collecting article {article['url']}...")
            r = requests.get(article["url"])
            soup = BeautifulSoup(r.text, "lxml")
            body = soup.find("div", {"class": "node"}).text
            # print(body)
            self.db.articles.update_one(
                {"_id": article["_id"]}, {"$set": {"body": body}}
            )
            time.sleep(2)
