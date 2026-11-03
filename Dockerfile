FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
# Build the index of the sample documents at image build time (offline provider).
RUN ragagent ingest data/docs

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENTRYPOINT ["ragagent"]
CMD ["eval", "data/eval/questions.jsonl"]
