import json
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from researchcrew.tools import SearchResult  # noqa: E402

# A tiny offline "web": sub-question -> search results, url -> page text.
WEB = {
    "What is RAG?": [
        SearchResult("RAG explained", "https://a.example/rag", "snippet a"),
        SearchResult("RAG explained (mirror)", "https://a.example/rag", "dup"),
        SearchResult("Broken page", "https://broken.example/x", "Snippet: RAG retrieves documents at query time."),
    ],
    "How is RAG evaluated?": [
        SearchResult("RAG metrics", "https://b.example/eval", "snippet b"),
        SearchResult("RAG explained", "https://a.example/rag", "snippet a"),
    ],
    "What does RAG cost?": [SearchResult("Pricing", "https://c.example/cost", "snippet c")],
}
PAGES = {
    "https://a.example/rag": "FACT: RAG retrieves documents at query time.\nFACT: RAG grounds answers in sources.",
    "https://b.example/eval": "FACT: Faithfulness measures grounding.",
    "https://c.example/cost": "FACT: Embedding 1M tokens costs a few cents.",
}


class FakeSearch:
    def __init__(self, fail_on=()):
        self.fail_on, self.queries = set(fail_on), []
        self._lock = threading.Lock()

    def __call__(self, query, k=5):
        with self._lock:
            self.queries.append(query)
        if query in self.fail_on:
            raise RuntimeError("rate limited")
        return WEB.get(query, [])[:k]


class FakeFetch:
    def __init__(self):
        self.fetched = []
        self._lock = threading.Lock()

    def __call__(self, url):
        with self._lock:
            self.fetched.append(url)
        if url not in PAGES:
            raise ConnectionError("404")
        return PAGES[url]


class ScriptedLLM:
    """Deterministic stand-in for a chat model, keyed on which ResearchCrew prompt it receives."""

    def __init__(self, subs=("What is RAG?", "How is RAG evaluated?"), critiques=None, draft=None):
        self.subs = list(subs)
        self.critiques = list(critiques or [{"ok": True, "issues": [], "follow_ups": []}])
        self.draft = draft
        self.calls = []
        self._lock = threading.Lock()

    def invoke(self, prompt):
        with self._lock:
            self.calls.append(prompt)
        if prompt.startswith("You lead a small research team"):
            reply = json.dumps({"sub_questions": self.subs})
        elif prompt.startswith("You are a researcher"):
            text = prompt.split("Page text:\n", 1)[1]
            reply = json.dumps({"facts": re.findall(r"FACT: (.+)", text) or
                                ([text.split("\n")[0].removeprefix("Snippet: ")] if "Snippet:" in text else [])})
        elif prompt.startswith("You are the writer"):
            n = len(re.findall(r"^\[\d+\]", prompt, re.M))
            revised = "critic flagged" in prompt
            reply = self.draft or ("Direct answer [2]. " + " ".join(f"Point [{i}]." for i in range(1, n + 1))
                                   + (" Revised." if revised else ""))
        elif prompt.startswith("You are the critic"):
            reply = json.dumps(self.critiques.pop(0) if len(self.critiques) > 1 else self.critiques[0])
        else:
            raise AssertionError("unexpected prompt: " + prompt[:60])
        return SimpleNamespace(content=reply)

    def count(self, prefix):
        return sum(c.startswith(prefix) for c in self.calls)


@pytest.fixture
def search():
    return FakeSearch()


@pytest.fixture
def fetch():
    return FakeFetch()
