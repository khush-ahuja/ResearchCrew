import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from run_eval import cited_claims  # noqa: E402

SOURCES = [{"id": 1, "facts": ["RAG retrieves documents."]}, {"id": 2, "facts": ["LoRA trains adapters."]}]


def test_cited_claims_pairs_each_sentence_with_its_sources():
    text = "RAG fetches documents first [1]. It is popular. LoRA is cheap [2][1]."
    claims = cited_claims(text, SOURCES)
    assert [c for c, _ in claims] == ["RAG fetches documents first .", "LoRA is cheap ."]
    assert "LoRA trains adapters." in claims[1][1] and "RAG retrieves documents." in claims[1][1]


def test_citations_to_unknown_sources_are_ignored():
    assert cited_claims("Made up [9].", SOURCES) == []
