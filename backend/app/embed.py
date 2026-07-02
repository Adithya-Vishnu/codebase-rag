"""Embedding wrapper around BAAI/bge-small-en-v1.5 (384-dim).

The model is loaded lazily on first use (not at import) so that FastAPI
can start fast and routes that don't embed stay cheap.
"""
from functools import lru_cache

from .config import EMBEDDING_MODEL

# BGE models want this instruction prepended to *queries only* (not passages).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed passages (code chunks) in batches. Normalized for cosine search."""
    vectors = _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a search query with the BGE query instruction prepended."""
    vector = _model().encode(
        BGE_QUERY_INSTRUCTION + query, normalize_embeddings=True, show_progress_bar=False
    )
    return vector.tolist()


if __name__ == "__main__":
    vecs = embed_texts(["def add(a, b): return a + b", "SELECT * FROM users", "hello world"])
    print(f"{len(vecs)} vectors, dim={len(vecs[0])}")
    q = embed_query("how do I add two numbers?")
    print(f"query vector dim={len(q)}")
