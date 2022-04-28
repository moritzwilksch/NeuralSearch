#%%
from datetime import datetime
from elasticsearch import Elasticsearch
import toml
import polars as pl
from rich.progress import track


config = toml.load("elasticsearch/env.toml")

es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="elasticsearch/http_ca.crt",
    basic_auth=(config.get("ES_USER"), config.get("ES_PASSWD")),
)

df = pl.read_parquet("data/ibkr_articles.parquet")

for idx, row in track(enumerate(df.to_dicts()), description="Inserting documents", total=df.height):
    es.index(index="ibkr-docs", id=idx, document=row)
