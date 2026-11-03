from pathlib import Path

import pytest

from ragagent.ingest import build_chunks, load_documents
from ragagent.providers import HashingEmbeddings
from ragagent.vectorstore import VectorStore

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def store() -> VectorStore:
    store = VectorStore(HashingEmbeddings(), "fake:hashing-1024")
    store.add(build_chunks(load_documents(DATA / "docs")))
    return store
