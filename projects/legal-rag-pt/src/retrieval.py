"""Retrievers: BM25 (lexical), dense (embeddings) and hybrid (Reciprocal Rank Fusion)."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

STOPWORDS = set(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre e ou que se "
    "ao aos à às é são ser foi como mais mas seu sua seus suas lhe há qual quais quando onde".split()
)


def load_chunks(path: str = "data/chunks.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", text) if t not in STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    def __init__(self, chunks: list[dict], model_name: str = "intfloat/multilingual-e5-base",
                 cache_dir: str = "data/cache"):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model_name = model_name
        self.name = f"dense:{model_name.split('/')[-1]}"
        self.is_e5 = "e5" in model_name.lower()
        self.model = SentenceTransformer(model_name)

        key = hashlib.md5(("|".join(c["chunk_id"] for c in chunks) + model_name).encode()).hexdigest()[:10]
        cache = Path(cache_dir) / f"emb_{key}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            prefix = "passage: " if self.is_e5 else ""
            self.emb = self.model.encode([prefix + c["text"] for c in chunks], batch_size=32,
                                         normalize_embeddings=True, show_progress_bar=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        prefix = "query: " if self.is_e5 else ""
        q = self.model.encode([prefix + query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over several ranked lists of chunk indices."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, retrievers: list, pool: int = 50):
        self.retrievers = retrievers
        self.pool = pool
        self.name = "hybrid:" + "+".join(r.name for r in retrievers)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        rankings = [[i for i, _ in r.search(query, self.pool)] for r in self.retrievers]
        return rrf(rankings)[:k]
