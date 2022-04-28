from datetime import datetime
from elasticsearch import Elasticsearch
import toml
import polars as pl
from rich import print

config = toml.load("elasticsearch/env.toml")

es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="elasticsearch/http_ca.crt",
    basic_auth=(config.get("ES_USER"), config.get("ES_PASSWD")),
)

resp = es.search(index="ibkr-docs", query={"match": {"body": "inactivity fee"}})
print(pl.from_dicts([x["_source"] for x in resp["hits"]["hits"]]))
