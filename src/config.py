import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def require_real_secret(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} environment variable is not set")
    if value.startswith("sk-your") or "your_" in value or "your-" in value:
        raise ValueError(f"{name} contains a placeholder value")
    return value


def get_neo4j_uri() -> str:
    uri = os.getenv("NEO4J_URI")
    if not uri:
        raise ValueError("NEO4J_URI environment variable is not set")
    return uri


def get_neo4j_user() -> str:
    user = os.getenv("NEO4J_USER")
    if not user:
        raise ValueError("NEO4J_USER environment variable is not set")
    return user


def get_neo4j_password() -> str:
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable is not set")
    return password


def get_openai_api_key() -> str:
    return require_real_secret(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY")


def get_llm_api_key() -> str:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    return require_real_secret(api_key, "LLM_API_KEY")


def get_llm_base_url() -> str | None:
    return os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def get_llm_model() -> str | None:
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")


def get_embedding_api_key() -> str:
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    return require_real_secret(api_key, "EMBEDDING_API_KEY")


def get_embedding_base_url() -> str | None:
    return os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embedding_dim() -> int:
    return int(os.getenv("EMBEDDING_DIM", "1024"))


def get_embedding_dimensions() -> int | None:
    dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    if dimensions:
        return int(dimensions)

    model = get_embedding_model()
    if model in {"embedding-3", "text-embedding-3-small", "text-embedding-3-large"}:
        return get_embedding_dim()

    return None


def get_reranker_api_key() -> str:
    api_key = (
        os.getenv("RERANKER_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    return require_real_secret(api_key, "RERANKER_API_KEY")


def get_reranker_base_url() -> str | None:
    return os.getenv("RERANKER_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def get_reranker_model() -> str:
    return os.getenv("RERANKER_MODEL") or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-nano"


def suppress_neo4j_notifications() -> bool:
    value = os.getenv("SUPPRESS_NEO4J_NOTIFICATIONS", "true").lower()
    return value in {"1", "true", "yes", "on"}
