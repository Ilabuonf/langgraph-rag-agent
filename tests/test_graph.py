"""Agent tests. Offline mode needs no model; a stub chat model exercises the LLM path."""

from langchain_core.messages import AIMessage

from ragagent.graph import NOT_FOUND, Answer, Grade, ask, build_graph


class StubStructured:
    def __init__(self, parent, schema):
        self.parent, self.schema = parent, schema

    def invoke(self, prompt):
        self.parent.calls.append((self.schema.__name__, prompt))
        if self.schema is Grade:
            return Grade(relevant=self.parent.relevant)
        return Answer(answer="Stub answer.", cited_passages=self.parent.cited)


class StubLLM:
    """Mimics the parts of a LangChain chat model the graph uses."""

    def __init__(self, relevant=True, cited=(1,), rewrite_to="passwords length"):
        self.relevant, self.cited, self.rewrite_to = relevant, list(cited), rewrite_to
        self.calls = []

    def with_structured_output(self, schema):
        return StubStructured(self, schema)

    def invoke(self, prompt):
        self.calls.append(("rewrite", prompt))
        return AIMessage(content=self.rewrite_to)


def test_offline_answer_cites_the_right_document(store):
    result = ask(build_graph(store), "How long must passwords be?")
    assert "14 characters" in result["answer"]
    assert result["citations"][0].startswith("security.md")


def test_offline_refuses_unanswerable_question(store):
    result = ask(build_graph(store), "Who won the football World Cup in 2022?")
    assert result["answer"] == NOT_FOUND
    assert result["citations"] == []


def test_llm_path_uses_structured_grading_and_answer(store):
    llm = StubLLM(relevant=True, cited=[1, 1, 99])
    result = ask(build_graph(store, llm, k=2), "How long must passwords be?")
    assert result["answer"] == "Stub answer."
    # Duplicate and out-of-range citation numbers are dropped.
    assert len(result["citations"]) == 1
    kinds = [kind for kind, _ in llm.calls]
    assert kinds.count("Grade") == 2 and kinds.count("Answer") == 1


def test_llm_path_rewrites_query_then_falls_back(store):
    llm = StubLLM(relevant=False)
    result = ask(build_graph(store, llm, max_rewrites=1), "pwd rules?")
    assert result["answer"] == NOT_FOUND
    assert result["rewrites"] == 1
    assert result["query"] == "passwords length"


def test_graph_has_expected_nodes(store):
    nodes = set(build_graph(store).get_graph().nodes)
    assert {"retrieve", "grade", "rewrite", "generate", "fallback"} <= nodes
