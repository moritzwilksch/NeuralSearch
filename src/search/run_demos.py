#%%
from operator import mod

from rich.console import Console
from rich.panel import Panel

c = Console()

import polars as pl

from src.search.models import (
    AutoEncoderModel,
    SentenceEncoderModel,
    SpacyEmbeddingModel,
    TfidfSearchModel,
)

df = pl.read_parquet("data/ibkr_articles.parquet")

QUERY = "inactivity fee"
# model = TfidfSearchModel()
# model.fit(df)
# results = model.search(QUERY)


# model = SpacyEmbeddingModel()
# model.fit(df)
# results = model.search(QUERY)
# print("done")

# model = AutoEncoderModel()
# model.fit(df)
# results = model.search(QUERY)
# print(results)


model = SentenceEncoderModel()
model.fit(df)
results = model.search(QUERY)
print(results)

for url, title, body in results.rows():
    c.print(Panel(f"{title}", style="bold"))
    c.print(body, style="dim")
    c.print()
