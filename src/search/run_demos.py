from lib2to3.pytree import Base

import polars as pl

from src.search.models import TfidfSearchModel

df = pl.read_parquet("data/ibkr_articles.parquet")
model = TfidfSearchModel()
model.fit(df)
results = model.search("inactivity fee")
print(results)
