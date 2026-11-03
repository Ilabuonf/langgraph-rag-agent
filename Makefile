.PHONY: install test lint format demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

demo:
	ragagent ingest data/docs
	ragagent ask "When is business class allowed for flights?"
	ragagent eval data/eval/questions.jsonl
