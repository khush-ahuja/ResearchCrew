from conftest import FakeSearch, ScriptedLLM

from researchcrew.graph import NO_SOURCES, merge_sources, run


def test_plan_fans_out_one_researcher_per_sub_question(search, fetch):
    llm = ScriptedLLM()
    out = run(llm, "Explain RAG", search, fetch)
    assert out["sub_questions"] == ["What is RAG?", "How is RAG evaluated?"]
    assert sorted(search.queries) == ["How is RAG evaluated?", "What is RAG?"]
    assert sorted(out["researched"]) == sorted(out["sub_questions"])
    assert sum(t["step"] == "research" for t in out["trace"]) == 2


def test_sources_are_deduplicated_across_researchers(search, fetch):
    out = run(ScriptedLLM(), "Explain RAG", search, fetch)
    urls = [s["url"] for s in out["sources"]]
    assert len(urls) == len(set(urls))
    assert [s["id"] for s in out["sources"]] == list(range(1, len(urls) + 1))
    rag = next(s for s in out["sources"] if s["url"] == "https://a.example/rag")
    assert rag["facts"] == ["RAG retrieves documents at query time.", "RAG grounds answers in sources."]


def test_same_url_is_fetched_once_per_researcher(search, fetch):
    run(ScriptedLLM(subs=["What is RAG?"]), "Explain RAG", search, fetch)
    assert fetch.fetched.count("https://a.example/rag") == 1


def test_unreadable_page_falls_back_to_search_snippet(search, fetch):
    out = run(ScriptedLLM(subs=["What is RAG?"]), "Explain RAG", search, fetch)
    broken = next(s for s in out["sources"] if s["url"] == "https://broken.example/x")
    assert broken["facts"] == ["RAG retrieves documents at query time."]


def test_failed_search_does_not_crash_the_run(fetch):
    search = FakeSearch(fail_on={"How is RAG evaluated?"})
    out = run(ScriptedLLM(), "Explain RAG", search, fetch)
    assert any("search failed" in t["detail"] for t in out["trace"])
    assert out["sources"]  # the other researcher still contributed


def test_no_sources_gives_honest_answer_without_writer(search, fetch):
    llm = ScriptedLLM(subs=["Something nobody wrote about"])
    out = run(llm, "Unknowable?", search, fetch)
    assert out["draft"] == NO_SOURCES
    assert llm.count("You are the writer") == 0


def test_critic_follow_ups_trigger_more_research_then_rewrite(search, fetch):
    llm = ScriptedLLM(critiques=[
        {"ok": False, "issues": ["cost not covered"], "follow_ups": ["What does RAG cost?"]},
        {"ok": True, "issues": [], "follow_ups": []},
    ])
    out = run(llm, "Explain RAG", search, fetch)
    assert "What does RAG cost?" in search.queries
    assert any(s["url"] == "https://c.example/cost" for s in out["sources"])
    assert llm.count("You are the writer") == 2
    assert out["draft"].endswith("Revised.")
    assert out["critique"]["ok"] is True
    assert [t["step"] for t in out["trace"]][-2:] == ["write", "critique"]


def test_critic_without_follow_ups_sends_draft_back_to_writer(search, fetch):
    llm = ScriptedLLM(critiques=[
        {"ok": False, "issues": ["claim [1] unsupported"], "follow_ups": []},
        {"ok": True, "issues": [], "follow_ups": []},
    ])
    run(llm, "Explain RAG", search, fetch)
    assert len(search.queries) == 2  # no new research
    assert llm.count("You are the writer") == 2
    assert "claim [1] unsupported" in [c for c in llm.calls if c.startswith("You are the writer")][1]


def test_revision_loop_is_bounded(search, fetch):
    llm = ScriptedLLM(critiques=[{"ok": False, "issues": ["never happy"], "follow_ups": []}])
    out = run(llm, "Explain RAG", search, fetch, max_revisions=1)
    assert llm.count("You are the writer") == 2
    assert out["critique"]["ok"] is False


def test_already_researched_follow_ups_are_ignored(search, fetch):
    llm = ScriptedLLM(critiques=[
        {"ok": False, "issues": ["thin"], "follow_ups": ["What is RAG?"]},
        {"ok": True, "issues": [], "follow_ups": []},
    ])
    run(llm, "Explain RAG", search, fetch)
    assert search.queries.count("What is RAG?") == 1


def test_unparseable_plan_falls_back_to_original_question(search, fetch):
    class BadPlanner(ScriptedLLM):
        def invoke(self, prompt):
            if prompt.startswith("You lead"):
                from types import SimpleNamespace
                return SimpleNamespace(content="sure! here are some ideas")
            return super().invoke(prompt)

    out = run(BadPlanner(), "What is RAG?", search, fetch)
    assert out["sub_questions"] == ["What is RAG?"]


def test_merge_sources_skips_empty_notes():
    notes = [{"url": "u1", "title": "t", "facts": []}, {"url": "u2", "title": "t2", "facts": ["f", "f"]}]
    assert merge_sources(notes) == [{"id": 1, "url": "u2", "title": "t2", "facts": ["f"]}]
