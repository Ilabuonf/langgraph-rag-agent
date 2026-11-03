"""Model providers: OpenAI, Azure OpenAI, or an offline "fake" mode for demos and tests."""

from __future__ import annotations

import hashlib
import re

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ragagent.config import Settings

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i if in is it its my of on or our
    the this to us we what when where which who why will with you your should may must""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase content words, with a light plural strip so 'laptops' matches 'laptop'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings via the hashing trick.

    Needs no API key or download, so the whole pipeline runs offline and in CI.
    It is keyword-based, so use a real embedding model for semantic search.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def embedding_id(settings: Settings) -> str:
    """Identifier stored with the index, so an index is never queried with different embeddings."""
    if settings.provider == "fake":
        return "fake:hashing-1024"
    if settings.provider == "azure":
        return f"azure:{settings.azure_embedding_deployment}"
    return f"openai:{settings.embedding_model}"


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.provider == "fake":
        return HashingEmbeddings()
    if settings.provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_embedding_deployment,
            api_version=settings.azure_api_version,
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model)


def get_chat_model(settings: Settings) -> BaseChatModel | None:
    """Return a chat model, or None in fake mode (the graph then uses offline heuristics)."""
    if settings.provider == "fake":
        return None
    if settings.provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=settings.azure_chat_deployment,
            api_version=settings.azure_api_version,
            temperature=0,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.chat_model, temperature=0)
