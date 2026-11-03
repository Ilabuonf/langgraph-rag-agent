import pytest

from ragagent.ingest import Chunk
from ragagent.providers import HashingEmbeddings
from ragagent.vectorstore import VectorStore


def test_search_ranks_relevant_document_first(store):
    results = store.search("How long must passwords be?", k=3)
    assert results[0][0].source == "security.md"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_store_returns_nothing():
    assert VectorStore(HashingEmbeddings()).search("anything") == []


def test_save_and_load_roundtrip(tmp_path, store):
    store.save(tmp_path / "index")
    loaded = VectorStore.load(tmp_path / "index", HashingEmbeddings(), "fake:hashing-1024")
    assert len(loaded) == len(store)
    query = "daily meal allowance abroad"
    assert [c.id for c, _ in loaded.search(query)] == [c.id for c, _ in store.search(query)]


def test_load_rejects_mismatched_embeddings(tmp_path):
    store = VectorStore(HashingEmbeddings(), "fake:hashing-1024")
    store.add([Chunk(id="a#0", text="hello world", source="a", chunk_index=0)])
    store.save(tmp_path)
    with pytest.raises(ValueError, match="Re-run"):
        VectorStore.load(tmp_path, HashingEmbeddings(), "openai:text-embedding-3-small")


def test_load_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope", HashingEmbeddings(), "fake:hashing-1024")
