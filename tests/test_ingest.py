import pytest

from ragagent.ingest import build_chunks, chunk_text, load_documents
from tests.conftest import DATA


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_has_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].startswith("word0") and chunks[-1].endswith("word499")


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    first, second = chunk_text(text, chunk_size=150, overlap=50)[:2]
    assert first.split()[-1] in second.split()


def test_chunks_never_split_words():
    text = " ".join(f"token{i}" for i in range(400))
    for chunk in chunk_text(text, chunk_size=120, overlap=30):
        assert all(word.startswith("token") for word in chunk.split())


def test_prefers_paragraph_breaks():
    text = ("A" * 60 + " sentence.") + "\n\n" + ("B" * 60 + " sentence.")
    assert chunk_text(text, chunk_size=100, overlap=0)[0].endswith("sentence.")


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=size, overlap=overlap)


def test_load_documents_and_chunk_ids():
    docs = load_documents(DATA / "docs")
    assert {source for source, _ in docs} == {
        "expenses.md",
        "onboarding.md",
        "remote_work.md",
        "security.md",
    }
    chunks = build_chunks(docs, chunk_size=400, overlap=50)
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.id == f"{c.source}#{c.chunk_index}" for c in chunks)


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_documents(DATA / "does-not-exist")
