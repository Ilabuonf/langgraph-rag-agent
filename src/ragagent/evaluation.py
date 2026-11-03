"""Offline evaluation: retrieval quality and answer grounding on a labelled question set.

Each line of the JSONL file is:
    {"question": "...", "expected_source": "file.md"}   # answerable
    {"question": "...", "expected_source": null}        # should be refused
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ragagent.graph import NOT_FOUND, ask
from ragagent.vectorstore import VectorStore


@dataclass
class EvalReport:
    n_answerable: int
    n_unanswerable: int
    hit_rate: float  # expected source in top-k retrieval
    mrr: float  # mean reciprocal rank of the expected source
    citation_accuracy: float  # answer cites the expected source
    refusal_accuracy: float  # unanswerable questions correctly refused

    def as_table(self) -> str:
        rows = [
            ("Answerable questions", str(self.n_answerable)),
            ("Unanswerable questions", str(self.n_unanswerable)),
            ("Retrieval hit rate@k", f"{self.hit_rate:.0%}"),
            ("Retrieval MRR", f"{self.mrr:.2f}"),
            ("Citation accuracy", f"{self.citation_accuracy:.0%}"),
            ("Refusal accuracy", f"{self.refusal_accuracy:.0%}"),
        ]
        width = max(len(r[0]) for r in rows)
        lines = [f"| {'Metric'.ljust(width)} | Value |", f"| {'-' * width} | ----- |"]
        lines += [f"| {name.ljust(width)} | {value} |" for name, value in rows]
        return "\n".join(lines)


def load_questions(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(app, store: VectorStore, questions: list[dict], k: int = 4) -> EvalReport:
    answerable = [q for q in questions if q.get("expected_source")]
    unanswerable = [q for q in questions if not q.get("expected_source")]

    hits, reciprocal_ranks, cited_ok = 0, 0.0, 0
    for item in answerable:
        sources = [chunk.source for chunk, _ in store.search(item["question"], k)]
        if item["expected_source"] in sources:
            hits += 1
            reciprocal_ranks += 1 / (sources.index(item["expected_source"]) + 1)
        result = ask(app, item["question"])
        if any(c.split("#")[0] == item["expected_source"] for c in result.get("citations", [])):
            cited_ok += 1

    refused = sum(1 for item in unanswerable if ask(app, item["question"])["answer"] == NOT_FOUND)

    def ratio(a: float, b: int) -> float:
        return a / b if b else 0.0

    return EvalReport(
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        hit_rate=ratio(hits, len(answerable)),
        mrr=ratio(reciprocal_ranks, len(answerable)),
        citation_accuracy=ratio(cited_ok, len(answerable)),
        refusal_accuracy=ratio(refused, len(unanswerable)),
    )
