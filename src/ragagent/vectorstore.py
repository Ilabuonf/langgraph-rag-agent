"""A small in-memory vector store with cosine similarity, persisted to disk with NumPy.

Deliberately dependency-free (no FAISS or vector DB) so the retrieval logic is easy to read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from ragagent.ingest import Chunk


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self, embeddings: Embeddings, embedding_id: str = "unknown"):
        self.embeddings = embeddings
        self.embedding_id = embedding_id
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = np.asarray(
            self.embeddings.embed_documents([c.text for c in chunks]), dtype=np.float32
        )
        vectors = _normalize(vectors)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks with their cosine similarity, best first."""
        if self.matrix is None or not self.chunks:
            return []
        q = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        scores = self.matrix @ (q / norm)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.matrix if self.matrix is not None else np.zeros((0, 0)))
        meta = {"embedding_id": self.embedding_id, "chunks": [c.to_dict() for c in self.chunks]}
        (path / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path, embeddings: Embeddings, embedding_id: str) -> VectorStore:
        path = Path(path)
        if not (path / "chunks.json").exists():
            raise FileNotFoundError(f"No index at {path}. Run `ragagent ingest` first.")
        meta = json.loads((path / "chunks.json").read_text())
        if meta["embedding_id"] != embedding_id:
            raise ValueError(
                f"Index was built with {meta['embedding_id']!r} embeddings but the current "
                f"provider uses {embedding_id!r}. Re-run `ragagent ingest`."
            )
        store = cls(embeddings, embedding_id)
        store.chunks = [Chunk.from_dict(c) for c in meta["chunks"]]
        matrix = np.load(path / "vectors.npy")
        store.matrix = matrix if matrix.size else None
        return store
