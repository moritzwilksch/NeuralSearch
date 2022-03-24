from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseSearchModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        ...

    @abstractmethod
    def search(self, query: str) -> pl.DataFrame:
        ...

    def top_k(self, similarities, k):
        return np.argsort(similarities)[0][: -k - 1 : -1]


class TfidfSearchModel(BaseSearchModel):
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_df=0.7
        )

    def fit(self, data: pl.DataFrame) -> None:
        self.data = data
        self.vectorizer.fit(self.data["body"].to_list())
        self.vectorized_articles = self.vectorizer.transform(
            self.data["body"].to_list()
        )
        return self.vectorized_articles

    def search(self, query: str) -> pl.DataFrame:
        vectorized_query = self.vectorizer.transform([query])
        similarities = cosine_similarity(vectorized_query, self.vectorized_articles)
        top_idxs = self.top_k(similarities, 3)
        return self.data[top_idxs]
