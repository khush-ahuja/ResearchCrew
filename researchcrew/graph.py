"""ResearchCrew: a multi-agent research workflow in LangGraph.

    plan ──Send×N──► research (parallel, one per sub-question) ──► write ──► critique ─┬─► END
                         ▲                                            ▲              │
                         └──────────── Send×follow-ups ───────────────┼──────────────┤  (missing info)
                                                                      └──────────────┘  (unsupported claims)

- planner     splits the question into searchable sub-questions
- researchers search the web, read the top pages and extract facts with their source URL (map step)
- writer      merges and de-duplicates notes, numbers the sources, writes a brief with [n] citations (reduce)
- critic      checks claims against sources; can request follow-up research or a rewrite (bounded loop)
"""
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from researchcrew import prompts
from researchcrew.llm import parse_json
from researchcrew.tools import Fetch, Search

NO_SOURCES = "I could not find reliable sources for this question."


class CrewState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    researched: Annotated[list[str], operator.add]
    notes: Annotated[list[dict], operator.add]
    sources: list[dict]
    draft: str
    critique: dict
    revisions: int
    trace: Annotated[list[dict], operator.add]


class ResearchTask(TypedDict):
    question: str
    sub_question: str


def _step(name: str, t0: float, detail: str) -> list[dict]:
    return [{"step": name, "detail": detail, "secs": round(time.perf_counter() - t0, 2)}]


def merge_sources(notes: list[dict]) -> list[dict]:
    """One numbered source per URL, with its facts merged and de-duplicated."""
    by_url: dict[str, dict] = {}
    for n in notes:
        if not n.get("facts"):
            continue
        src = by_url.setdefault(n["url"], {"url": n["url"], "title": n["title"], "facts": []})
        for f in n["facts"]:
            if f not in src["facts"]:
                src["facts"].append(f)
    return [{"id": i, **s} for i, s in enumerate(by_url.values(), 1)]


def format_sources(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['id']}] {s['title']} ({s['url']})\n" + "\n".join(f"- {f}" for f in s["facts"])
                       for s in sources)


def build_graph(llm, search: Search, fetch: Fetch, max_sub: int = 4, results_per_query: int = 5,
                pages_per_query: int = 3, max_revisions: int = 1):
    def ask(prompt: str) -> str:
        return llm.invoke(prompt).content

    def plan(state: CrewState):
        t0 = time.perf_counter()
        out = parse_json(ask(prompts.PLAN.format(question=state["question"], max_sub=max_sub)), {})
        subs = [s.strip() for s in out.get("sub_questions", []) if isinstance(s, str) and s.strip()][:max_sub]
        subs = subs or [state["question"]]
        return {"sub_questions": subs, "revisions": 0,
                "trace": _step("plan", t0, f"{len(subs)} sub-questions: " + " | ".join(subs))}

    def dispatch(state: CrewState):
        return [Send("research", {"question": state["question"], "sub_question": sq}) for sq in state["sub_questions"]]

    def research(task: ResearchTask):
        t0 = time.perf_counter()
        sq = task["sub_question"]
        try:
            results = search(sq, k=results_per_query)
        except Exception as e:  # network/API problems should not kill the whole run
            return {"researched": [sq], "trace": _step("research", t0, f"{sq!r}: search failed ({e})")}
        notes, read, seen = [], 0, set()
        for r in results:
            if read >= pages_per_query or r.url in seen:
                continue
            seen.add(r.url)
            try:
                text = fetch(r.url)
            except Exception:
                text = r.snippet  # fall back to the search snippet if the page can't be read
            if not text.strip():
                continue
            read += 1
            out = parse_json(ask(prompts.EXTRACT.format(sub_question=sq, title=r.title, text=text[:6000])), {})
            facts = [f.strip() for f in out.get("facts", []) if isinstance(f, str) and f.strip()]
            notes.append({"sub_question": sq, "url": r.url, "title": r.title, "facts": facts})
        n_facts = sum(len(n["facts"]) for n in notes)
        return {"notes": notes, "researched": [sq],
                "trace": _step("research", t0, f"{sq!r}: read {read} pages, {n_facts} facts")}

    def write(state: CrewState):
        t0 = time.perf_counter()
        sources = merge_sources(state.get("notes", []))
        if not sources:
            return {"sources": [], "draft": NO_SOURCES, "trace": _step("write", t0, "no sources")}
        crit = state.get("critique") or {}
        feedback = prompts.FEEDBACK.format(issues="\n".join(f"- {i}" for i in crit.get("issues", []))) \
            if crit and not crit.get("ok", True) and crit.get("issues") else ""
        draft = ask(prompts.WRITE.format(question=state["question"], sources=format_sources(sources),
                                         feedback=feedback)).strip()
        return {"sources": sources, "draft": draft,
                "trace": _step("write", t0, f"{len(sources)} sources" + (", revised" if feedback else ""))}

    def critique(state: CrewState):
        t0 = time.perf_counter()
        if not state["sources"]:
            return {"critique": {"ok": True, "issues": [], "follow_ups": []}}
        out = parse_json(ask(prompts.CRITIQUE.format(question=state["question"], sources=format_sources(
            state["sources"]), draft=state["draft"])), {"ok": True})
        done = set(state.get("researched", []))
        follow = [f for f in out.get("follow_ups", []) if isinstance(f, str) and f.strip() and f not in done][:2]
        crit = {"ok": bool(out.get("ok", True)), "issues": out.get("issues", []) or [], "follow_ups": follow}
        detail = "approved" if crit["ok"] else f"{len(crit['issues'])} issues, {len(follow)} follow-ups"
        return {"critique": crit, "revisions": state["revisions"] + (0 if crit["ok"] else 1),
                "trace": _step("critique", t0, detail)}

    def after_critique(state: CrewState):
        crit = state["critique"]
        if crit["ok"] or state["revisions"] > max_revisions:
            return END
        if crit["follow_ups"]:
            return [Send("research", {"question": state["question"], "sub_question": q}) for q in crit["follow_ups"]]
        return "write"

    g = StateGraph(CrewState)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("write", write)
    g.add_node("critique", critique)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", dispatch, ["research"])
    g.add_edge("research", "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", after_critique, ["research", "write", END])
    return g.compile()


def run(llm, question: str, search: Search, fetch: Fetch, **kw) -> dict:
    return build_graph(llm, search, fetch, **kw).invoke({"question": question})
