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

# model = TfidfSearchModel()
# model.fit(df)
# results = model.search("inactivity fee")


# model = SpacyEmbeddingModel()
# model.fit(df)
# results = model.search("inactivity fee")
# print("done")

# model = AutoEncoderModel()
# model.fit(df)
# results = model.search("inactivity fee")
# print(results)


model = SentenceEncoderModel()
model.fit(df)
results = model.search("inactivity fee")
print(results)

for url, title, body in results.rows():
    c.print(Panel(f"{title}", style="bold"))
    c.print(body, style="dim")
    c.print()
