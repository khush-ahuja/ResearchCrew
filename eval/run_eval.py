"""Real-model, real-web run of ResearchCrew with a citation audit.

    python eval/run_eval.py

For each question it runs the full crew (live web search + page reading) and records how the agents
behaved (sub-questions, pages read, sources, critic revisions, time, LLM calls, tokens). Then an
independent judge model checks every cited sentence in the final report against the facts the cited
sources contributed:
  supported_claims   share of cited sentences the judge says are backed by their cited sources
  invalid_citations  citations that pointed to a source number that does not exist (removed by report.py)
Writes eval/results.json and one Markdown report per question in examples/.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from researchcrew.config import Settings  # noqa: E402
from researchcrew.graph import run  # noqa: E402
from researchcrew.llm import get_llm, parse_json  # noqa: E402
from researchcrew.report import finalize, to_markdown  # noqa: E402
from researchcrew.tools import DuckDuckGoSearch, HttpFetch  # noqa: E402

QUESTIONS = [
    "What is retrieval-augmented generation and what are its main failure modes?",
    "How does LoRA fine-tuning reduce the cost of adapting large language models?",
    "What is the difference between HNSW and IVF indexes in vector databases?",
    "What are the main approaches to evaluating LLM agents?",
    "How do mixture-of-experts language models work?",
]

SUPPORT = """Do the source facts below support the claim? Judge only from the facts given.

Claim: {claim}

Source facts:
{facts}

Reply with JSON only: {{"supported": true or false}}"""

_SENT = re.compile(r"[^.!?\n]*\[\d+\][^.!?\n]*[.!?]?")
_CITE = re.compile(r"\[(\d+)\]")


class CountingLLM:
    def __init__(self, llm):
        self.llm, self.calls, self.tokens = llm, 0, 0

    def invoke(self, prompt):
        out = self.llm.invoke(prompt)
        self.calls += 1
        usage = (getattr(out, "response_metadata", {}) or {}).get("token_usage", {}) or {}
        self.tokens += usage.get("total_tokens", 0)
        return out


def cited_claims(text: str, sources: list[dict]) -> list[tuple[str, str]]:
    """(sentence without citation marks, facts of the sources it cites) for every cited sentence."""
    by_id = {s["id"]: s for s in sources}
    out = []
    for m in _SENT.finditer(text):
        sent = m.group(0).strip()
        ids = [int(i) for i in _CITE.findall(sent) if int(i) in by_id]
        if not ids:
            continue
        facts = "\n".join(f"- {f}" for i in dict.fromkeys(ids) for f in by_id[i]["facts"])
        out.append((_CITE.sub("", sent).strip(), facts))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", default="openai/gpt-oss-120b")
    p.add_argument("--max-sub", type=int, default=3)
    a = p.parse_args()

    settings = Settings()
    judge = get_llm(Settings(llm_provider=settings.llm_provider, llm_model=a.judge))
    (ROOT / "examples").mkdir(exist_ok=True)
    rows = []
    for qi, q in enumerate(QUESTIONS, 1):
        llm = CountingLLM(get_llm(settings))
        t0 = time.perf_counter()
        res = run(llm, q, DuckDuckGoSearch(), HttpFetch(), max_sub=a.max_sub)
        secs = time.perf_counter() - t0
        text, cited, invalid = finalize(res.get("draft", ""), res.get("sources", []))
        claims = cited_claims(text, cited)
        verdicts = [bool(parse_json(judge.invoke(SUPPORT.format(claim=c, facts=f)).content, {}).get("supported"))
                    for c, f in claims]
        trace = res.get("trace", [])
        rows.append({
            "question": q,
            "sub_questions": len(res.get("sub_questions", [])),
            "research_steps": sum(t["step"] == "research" for t in trace),
            "sources_cited": len(cited),
            "critic_revisions": sum(t["step"] == "write" for t in trace) - 1,
            "cited_sentences": len(claims),
            "supported_claims": round(sum(verdicts) / len(verdicts), 3) if verdicts else None,
            "invalid_citations": len(invalid),
            "critic_ok_at_end": bool((res.get("critique") or {}).get("ok", True)),
            "seconds": round(secs, 1), "llm_calls": llm.calls, "tokens": llm.tokens,
        })
        (ROOT / "examples" / f"report_{qi}.md").write_text(to_markdown(res), encoding="utf-8")
        print(json.dumps(rows[-1]))

    scored = [r for r in rows if r["supported_claims"] is not None]
    total_claims = sum(r["cited_sentences"] for r in scored)
    summary = {
        "questions": len(rows),
        "supported_claims": round(sum(r["supported_claims"] * r["cited_sentences"] for r in scored)
                                  / max(total_claims, 1), 3),
        "cited_sentences": total_claims,
        "invalid_citations_removed": sum(r["invalid_citations"] for r in rows),
        "avg_sources_cited": round(sum(r["sources_cited"] for r in rows) / len(rows), 1),
        "runs_with_critic_revision": sum(r["critic_revisions"] > 0 for r in rows),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
        "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / len(rows), 1),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
    }
    meta = {"model": f"{settings.llm_provider}/{settings.llm_model}", "judge": a.judge, "search": "DuckDuckGo",
            "max_sub_questions": a.max_sub}
    (ROOT / "eval" / "results.json").write_text(
        json.dumps({"config": meta, "summary": summary, "per_question": rows}, indent=2))
    print(json.dumps({"config": meta, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
