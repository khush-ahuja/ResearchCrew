from researchcrew.report import finalize, to_markdown

SOURCES = [{"id": i, "url": f"https://s{i}.example", "title": f"S{i}", "facts": ["x"]} for i in (1, 2, 3)]


def test_citations_renumbered_by_first_appearance_and_uncited_dropped():
    text, cited, invalid = finalize("A [3]. B [1][3]. C.", SOURCES)
    assert text == "A [1]. B [2][1]. C."
    assert [(s["id"], s["url"]) for s in cited] == [(1, "https://s3.example"), (2, "https://s1.example")]
    assert invalid == []


def test_citations_to_missing_sources_are_removed_and_reported():
    text, cited, invalid = finalize("Real [2]. Invented [9].", SOURCES)
    assert text == "Real [1]. Invented."
    assert invalid == [9]
    assert len(cited) == 1


def test_markdown_has_sources_trace_and_warnings():
    md = to_markdown({
        "question": "Q?", "draft": "Answer [2]. Bad [7].", "sources": SOURCES,
        "critique": {"ok": False, "issues": ["thin on cost"]},
        "trace": [{"step": "plan", "detail": "2 sub-questions", "secs": 0.5},
                  {"step": "write", "detail": "2 sources", "secs": 1.0}],
    })
    assert md.startswith("# Q?")
    assert "1. [S2](https://s2.example)" in md
    assert "S1" not in md
    assert "1 citation(s) to non-existent sources were removed" in md
    assert "thin on cost" in md
    assert "Total agent time: 1.5s" in md
