"""ResearchCrew: a multi-agent research assistant built with LangGraph."""
import os

from researchcrew.config import Settings


def research(question: str, settings: Settings | None = None, **kw) -> dict:
    """Runs the full crew with real web search. Uses Tavily if TAVILY_API_KEY is set, else DuckDuckGo."""
    from researchcrew.graph import run
    from researchcrew.llm import get_llm
    from researchcrew.tools import DuckDuckGoSearch, HttpFetch, TavilySearch

    key = os.getenv("TAVILY_API_KEY")
    search = TavilySearch(key) if key else DuckDuckGoSearch()
    return run(get_llm(settings or Settings()), question, search, HttpFetch(), **kw)
