from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default

    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw}")


def _env_allowed_origins(raw: str | None) -> Tuple[str, ...]:
    if not raw:
        return ()
    values = tuple(item.strip() for item in raw.split(","))
    return tuple(item for item in values if item)


@dataclass(frozen=True)
class AppConfig:
    service_name: str = "rag-knowledge-assistant"
    api_key: str | None = None
    storage_path: str | None = None

    default_chunk_size: int = 800
    default_chunk_overlap: int = 120
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False

    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60

    request_body_limit_bytes: int = 1_048_576
    request_timeout_seconds: int = 15

    log_level: str = "INFO"
    docs_enabled: bool = True
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            service_name=_env("RAG_SERVICE_NAME", "rag-knowledge-assistant"),
            api_key=_env("RAG_API_KEY"),
            storage_path=_normalize_storage_path(_env("RAG_STORAGE_PATH")),
            default_chunk_size=_env_int("RAG_DEFAULT_CHUNK_SIZE", 800),
            default_chunk_overlap=_env_int("RAG_DEFAULT_CHUNK_OVERLAP", 120),
            default_embedding_model=_env(
                "RAG_DEFAULT_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            host=_env("RAG_HOST", "127.0.0.1"),
            port=_env_int("RAG_PORT", 8000),
            reload=_env_bool("RAG_RELOAD", False),
            rate_limit_requests=_env_int("RAG_RATE_LIMIT_REQUESTS", 120),
            rate_limit_window_seconds=_env_int("RAG_RATE_LIMIT_WINDOW_SECONDS", 60),
            request_body_limit_bytes=_env_int("RAG_MAX_REQUEST_BYTES", 1_048_576),
            request_timeout_seconds=_env_int("RAG_REQUEST_TIMEOUT_SECONDS", 15),
            log_level=_env("RAG_LOG_LEVEL", "INFO"),
            docs_enabled=_env_bool("RAG_DOCS_ENABLED", True),
            allowed_origins=_env_allowed_origins(_env("RAG_ALLOWED_ORIGINS")),
        )


def _normalize_storage_path(raw: str | None) -> str | None:
    if not raw:
        return None

    path = Path(raw).expanduser()
    if not path.suffix:
        return str(path.with_suffix(".json"))
    return str(path)
