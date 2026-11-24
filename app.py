"""Streamlit UI that shows each agent's step as it happens.

    streamlit run app.py
"""
import os

import streamlit as st

from researchcrew.config import Settings
from researchcrew.graph import build_graph
from researchcrew.llm import get_llm
from researchcrew.report import to_markdown
from researchcrew.tools import DuckDuckGoSearch, HttpFetch, TavilySearch

st.set_page_config(page_title="ResearchCrew", page_icon="🔎", layout="wide")
st.title("🔎 ResearchCrew")
st.caption("A planner, parallel researchers, a writer and a critic answer your question with cited sources.")

question = st.text_input("Research question", placeholder="e.g. How do LLM agents use tools?")
max_sub = st.slider("Sub-questions", 2, 6, 4)

if st.button("Research", type="primary", disabled=not question.strip()):
    key = os.getenv("TAVILY_API_KEY")
    graph = build_graph(get_llm(Settings()), TavilySearch(key) if key else DuckDuckGoSearch(), HttpFetch(),
                        max_sub=max_sub)
    final: dict = {"question": question}
    with st.status("Agents at work...", expanded=True) as status:
        for update in graph.stream({"question": question}, stream_mode="updates"):
            for delta in update.values():
                for t in (delta or {}).get("trace", []):
                    st.write(f"**{t['step']}** ({t['secs']}s): {t['detail']}")
                for k, v in (delta or {}).items():
                    final[k] = final.get(k, []) + v if k in ("notes", "trace", "researched") else v
        status.update(label="Done", state="complete")
    st.markdown(to_markdown(final))
