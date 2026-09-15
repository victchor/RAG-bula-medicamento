import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    anthropic_api_key: str | None
    claude_model: str
    google_api_key: str | None
    gemini_model: str
    llm_temperature: float
    llm_max_tokens: int


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider not in ("anthropic", "google"):
        raise RuntimeError(f"LLM_PROVIDER inválido: {provider!r}. Use 'anthropic' ou 'google'.")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if provider == "anthropic" and not anthropic_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=anthropic mas ANTHROPIC_API_KEY não foi definida. "
            "Preencha o .env (https://console.anthropic.com/settings/keys)."
        )

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if provider == "google" and not google_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=google mas GOOGLE_API_KEY não foi definida. "
            "Gere uma chave gratuita em https://aistudio.google.com/apikey e preencha o .env."
        )

    return Settings(
        llm_provider=provider,
        anthropic_api_key=anthropic_api_key,
        claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        google_api_key=google_api_key,
        # gemini-3.1-flash-lite tem cota gratuita bem maior que os outros
        # modelos Gemini e sem overhead de "thinking" — trocar o padrão
        # para um modelo "melhor" pode reintroduzir limite de cota baixo.
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
    )
