"""Settings, read from environment variables (or a .env file)."""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    # "groq", "openai", or "ollama" (fully local)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-20b"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))
