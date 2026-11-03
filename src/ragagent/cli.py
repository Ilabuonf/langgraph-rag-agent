"""Command-line interface: ragagent ingest | ask | chat | eval | graph."""

from __future__ import annotations

import argparse
import sys

from ragagent.config import get_settings
from ragagent.evaluation import evaluate, load_questions
from ragagent.graph import ask, build_graph
from ragagent.ingest import build_chunks, load_documents
from ragagent.providers import embedding_id, get_chat_model, get_embeddings
from ragagent.vectorstore import VectorStore

DEFAULT_INDEX = ".index"


def _load_app(index: str):
    settings = get_settings()
    store = VectorStore.load(index, get_embeddings(settings), embedding_id(settings))
    return build_graph(store, get_chat_model(settings), k=settings.top_k), store, settings


def _print_result(result: dict) -> None:
    print(f"\n{result['answer']}")
    if result.get("citations"):
        print("\nSources:")
        for i, citation in enumerate(result["citations"], 1):
            print(f"  [{i}] {citation}")


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    docs = load_documents(args.path)
    chunks = build_chunks(docs, settings.chunk_size, settings.chunk_overlap)
    store = VectorStore(get_embeddings(settings), embedding_id(settings))
    store.add(chunks)
    store.save(args.index)
    print(f"Indexed {len(docs)} documents into {len(chunks)} chunks -> {args.index}/")


def cmd_ask(args: argparse.Namespace) -> None:
    app, _, _ = _load_app(args.index)
    _print_result(ask(app, args.question))


def cmd_chat(args: argparse.Namespace) -> None:
    app, _, _ = _load_app(args.index)
    print("Ask a question about your documents (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        _print_result(ask(app, question))


def cmd_eval(args: argparse.Namespace) -> None:
    app, store, settings = _load_app(args.index)
    report = evaluate(app, store, load_questions(args.questions), k=settings.top_k)
    print(f"Provider: {settings.provider}\n")
    print(report.as_table())


def cmd_graph(args: argparse.Namespace) -> None:
    from ragagent.providers import HashingEmbeddings

    print(build_graph(VectorStore(HashingEmbeddings())).get_graph().draw_mermaid())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragagent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index a folder or file (.md, .txt, .pdf)")
    p.add_argument("path")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive question loop")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="evaluate on a labelled JSONL question set")
    p.add_argument("questions")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("graph", help="print the agent graph as Mermaid")
    p.set_defaults(func=cmd_graph)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
