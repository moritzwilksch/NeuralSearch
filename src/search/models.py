from abc import ABC, abstractmethod

import numpy as np
import polars as pl
import spacy
import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.search.neural_networks import AutoEncoder


class BaseSearchModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        self.data = data

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
        self.vectorized_articles = self.vectorizer.transform(self.data["body"].to_list())
        return self.vectorized_articles

    def search(self, query: str) -> pl.DataFrame:
        vectorized_query = self.vectorizer.transform([query])
        similarities = cosine_similarity(vectorized_query, self.vectorized_articles)
        top_idxs = self.top_k(similarities, 3)
        return self.data[top_idxs]


# -----------------------------------------------------------------------------


class SpacyEmbeddingModel(BaseSearchModel):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load("en_core_web_md")

    def fit(self, data: pl.DataFrame) -> None:
        self.data = data
        self.vectors = []
        for doc in self.nlp.pipe(
            data.select("body").to_series().to_list(),
            batch_size=64,
            n_process=-1,
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        ):
            self.vectors.append(doc.vector)

        return self.vectors

    def search(self, query: str) -> pl.DataFrame:
        query_vector = self.nlp(query).vector
        similarities = cosine_similarity([query_vector], self.vectors)
        top_idxs = self.top_k(similarities, 3)
        return self.data[top_idxs]


# -----------------------------------------------------------------------------


class AutoEncoderModel(BaseSearchModel):
    def __init__(self) -> None:
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            # analyzer="char_wb", ngram_range=(3, 5), max_df=0.7
            analyzer="word",
            ngram_range=(1, 2),
            max_df=0.7,
        )

    def fit(self, data: pl.DataFrame) -> None:
        super().fit(data)
        self.vectorizer.fit(self.data["body"].to_list())
        self.vectorized_articles = self.vectorizer.transform(
            self.data["body"].to_list()
        ).toarray()

        print(self.vectorized_articles.shape)
        self.nn = AutoEncoder(
            vocab_size=self.vectorized_articles.shape[-1], bottleneck_dim=64
        )
        self.nn.compile(tf.keras.optimizers.Adam(learning_rate=0.01), "huber")
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        self.nn.fit(
            self.vectorized_articles,
            self.vectorized_articles,
            batch_size=128,
            epochs=200,
            callbacks=[tb_callback],
        )
        self.vectors = self.nn(self.vectorized_articles, return_vector=True).numpy()

    def search(self, query: str) -> pl.DataFrame:
        self.vectorized_query = self.nn(
            self.vectorizer.transform([query]).toarray(), return_vector=True
        ).numpy()
        similarities = cosine_similarity(self.vectorized_query, self.vectors)
        top_idxs = self.top_k(similarities, 3)
        return self.data[top_idxs]


# -----------------------------------------------------------------------------


class SentenceEncoderModel(BaseSearchModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data: pl.DataFrame) -> None:
        super().fit(data)

        self.model = SentenceTransformer(
            "sentence-transformers/msmarco-distilbert-dot-v5"
        )
        print("loaded")

        self.doc_emb = self.model.encode(self.data["body"].to_list())

    def search(self, query: str) -> pl.DataFrame:
        query_emb = self.model.encode([query])
        scores = util.dot_score(query_emb, self.doc_emb)[0].cpu().tolist()
        top_idxs = self.top_k([scores], 3)
        return self.data[top_idxs]
