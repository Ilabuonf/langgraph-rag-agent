"""The agent: a LangGraph state machine that retrieves, grades, self-corrects and answers.

    retrieve -> grade --(relevant docs)----------> generate -> END
                  |----(none, retries left)-----> rewrite -> retrieve
                  |----(none, no retries left)--> fallback -> END

With a chat model, grading and answering use structured outputs (Pydantic schemas), so the
answer always comes back with machine-checkable citations. Without one (fake mode) the same
graph runs on transparent keyword heuristics, which keeps tests and CI fully offline.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ragagent.ingest import Chunk
from ragagent.providers import tokenize
from ragagent.vectorstore import VectorStore

NOT_FOUND = "I couldn't find the answer in the provided documents."


class RAGState(TypedDict, total=False):
    question: str
    query: str
    documents: list[Chunk]
    relevant: list[Chunk]
    rewrites: int
    answer: str
    citations: list[str]


class Grade(BaseModel):
    """Whether a retrieved passage helps answer the question."""

    relevant: bool = Field(
        description="True if the passage contains information that answers the question."
    )


class Answer(BaseModel):
    """A grounded answer with the numbers of the passages it relies on."""

    answer: str = Field(description="Concise answer based only on the passages.")
    cited_passages: list[int] = Field(description="1-based numbers of the passages used.")


GRADE_PROMPT = """You are grading retrieved passages for a question-answering system.
Question: {question}

Passage:
{passage}

Does the passage contain information that helps answer the question?"""

REWRITE_PROMPT = """The search query below returned no relevant passages.
Rewrite it as a short, keyword-rich search query that keeps the same meaning.
Return only the new query.

Original question: {question}"""

ANSWER_PROMPT = """Answer the question using ONLY the numbered passages below.
If the passages do not contain the answer, say you don't know.
Cite the passages you used by their numbers.

{context}

Question: {question}"""


def keyword_overlap(question: str, text: str) -> float:
    """Share of the question's content words that appear in `text` (0..1)."""
    q = set(tokenize(question))
    if not q:
        return 0.0
    return len(q & set(tokenize(text))) / len(q)


def best_sentence(question: str, text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    sentences = [s for s in sentences if not s.startswith("#")] or sentences
    return max(sentences, key=lambda s: keyword_overlap(question, s))


def build_graph(
    store: VectorStore,
    llm: BaseChatModel | None = None,
    k: int = 4,
    max_rewrites: int = 1,
    min_overlap: float = 0.5,
):
    """Compile the agent graph. `llm=None` selects the offline heuristics."""

    def retrieve(state: RAGState) -> RAGState:
        query = state.get("query") or state["question"]
        docs = [chunk for chunk, score in store.search(query, k) if score > 0]
        return {"query": query, "documents": docs}

    def grade(state: RAGState) -> RAGState:
        question, docs = state["question"], state.get("documents", [])
        if llm is None:
            relevant = [d for d in docs if keyword_overlap(question, d.text) >= min_overlap]
        else:
            grader = llm.with_structured_output(Grade)
            relevant = [
                d
                for d in docs
                if grader.invoke(GRADE_PROMPT.format(question=question, passage=d.text)).relevant
            ]
        return {"relevant": relevant}

    def route(state: RAGState) -> str:
        if state.get("relevant"):
            return "generate"
        if llm is not None and state.get("rewrites", 0) < max_rewrites:
            return "rewrite"
        return "fallback"

    def rewrite(state: RAGState) -> RAGState:
        message = llm.invoke(REWRITE_PROMPT.format(question=state["question"]))
        return {"query": str(message.content).strip(), "rewrites": state.get("rewrites", 0) + 1}

    def generate(state: RAGState) -> RAGState:
        question, docs = state["question"], state["relevant"]
        if llm is None:
            top = max(docs, key=lambda d: keyword_overlap(question, d.text))
            return {"answer": f"{best_sentence(question, top.text)} [1]", "citations": [top.id]}

        context = "\n\n".join(f"[{i}] ({d.source})\n{d.text}" for i, d in enumerate(docs, 1))
        result = llm.with_structured_output(Answer).invoke(
            ANSWER_PROMPT.format(context=context, question=question)
        )
        # Keep only citation numbers that point to a real passage.
        numbers = dict.fromkeys(result.cited_passages)
        cited = [docs[i - 1].id for i in numbers if 1 <= i <= len(docs)]
        return {"answer": result.answer, "citations": cited}

    def fallback(state: RAGState) -> RAGState:
        return {"answer": NOT_FOUND, "citations": []}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route, {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def ask(app, question: str) -> RAGState:
    return app.invoke({"question": question, "rewrites": 0})
