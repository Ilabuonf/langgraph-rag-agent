from ragagent.evaluation import evaluate, load_questions
from ragagent.graph import build_graph
from tests.conftest import DATA


def test_offline_evaluation_meets_quality_bar(store):
    questions = load_questions(DATA / "eval" / "questions.jsonl")
    report = evaluate(build_graph(store), store, questions)
    assert report.n_answerable == 14 and report.n_unanswerable == 3
    # Regression guard: the offline baseline must not get worse.
    assert report.hit_rate >= 0.9
    assert report.citation_accuracy >= 0.8
    assert report.refusal_accuracy == 1.0
    assert "Retrieval MRR" in report.as_table()
