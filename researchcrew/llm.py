"""Chat-model factory plus a forgiving JSON parser for model outputs."""
import json
import re


def get_llm(settings):
    p = settings.llm_provider.lower()
    if p == "groq":
        from langchain_groq import ChatGroq
        extra = {"reasoning_effort": "low"} if "gpt-oss" in settings.llm_model else {}
        return ChatGroq(model=settings.llm_model, temperature=settings.temperature, max_retries=6, **extra)
    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm_model, temperature=settings.temperature)
    if p == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=settings.llm_model, temperature=settings.temperature)
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider}")


def parse_json(text: str, default):
    """Extracts the first JSON object/array from a model reply; returns `default` on failure."""
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return default
