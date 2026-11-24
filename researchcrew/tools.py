"""Tools the research agents can call: web search and page fetching.

Both are small classes behind simple interfaces so tests can swap in offline fakes.
"""
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class Search(Protocol):
    def __call__(self, query: str, k: int = 5) -> list[SearchResult]: ...


class Fetch(Protocol):
    def __call__(self, url: str) -> str: ...


class DuckDuckGoSearch:
    """Free web search, no API key (uses the `ddgs` package)."""

    def __call__(self, query: str, k: int = 5) -> list[SearchResult]:
        from ddgs import DDGS

        rows = DDGS().text(query, max_results=k) or []
        return [SearchResult(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in rows if r.get("href")]


class TavilySearch:
    """Optional: Tavily's search API (set TAVILY_API_KEY). Often cleaner results for research."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, query: str, k: int = 5) -> list[SearchResult]:
        import httpx

        r = httpx.post("https://api.tavily.com/search", timeout=30,
                       json={"api_key": self.api_key, "query": query, "max_results": k})
        r.raise_for_status()
        return [SearchResult(x["title"], x["url"], x.get("content", "")) for x in r.json().get("results", [])]


class HttpFetch:
    """Downloads a page and returns its readable text (scripts, nav and footers removed)."""

    def __init__(self, max_chars: int = 6000, timeout: float = 15.0):
        self.max_chars, self.timeout = max_chars, timeout

    def __call__(self, url: str) -> str:
        import httpx
        from bs4 import BeautifulSoup

        r = httpx.get(url, timeout=self.timeout, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (ResearchCrew; +https://github.com/khush-ahuja/ResearchCrew)"})
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", "html"):
            return r.text[: self.max_chars]
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))
        return re.sub(r"[ \t]+", " ", text).strip()[: self.max_chars]
