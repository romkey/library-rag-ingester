"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

DISTANCES = ("cosine", "dot", "euclid", "manhattan")


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


def _required(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise ConfigError(f"{name} is required")
    return value.strip()


def _optional(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _int(name: str, value: str | None, default: int, minimum: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        number = int(value)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, not {value!r}") from None
    if number < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {number}")
    return number


def _bool(name: str, value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, not {value!r}")


def _choice(
    name: str, value: str | None, default: str, choices: tuple[str, ...]
) -> str:
    text = (value or "").strip().lower() or default
    if text not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return text


def _url(name: str, value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must start with http:// or https://, not {value!r}")
    return value


def _prefix(value: str | None, default: str) -> str:
    """An embedding prompt prefix, ready to prepend to text.

    Empty means the default and "none" means no prefix. A literal \\n becomes
    a newline (for instruction prompts), and a separating space is added
    unless the prefix already ends in whitespace.
    """
    if value is None or not value.strip():
        return default
    if value.strip().lower() == "none":
        return ""
    text = value.replace("\\n", "\n")
    return text if text[-1].isspace() else text + " "


@dataclass(frozen=True)
class Settings:
    # The library's catalog feed (GET /api/v1/catalog) and its API key.
    library_catalog_url: str
    library_api_key: str
    request_timeout_seconds: int = 60

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "library"
    qdrant_distance: str = "cosine"

    # Embeddings, from an OpenAI-compatible /embeddings API
    embedding_model: str = "nomic-ai/modernbert-embed-base"
    embedding_batch_size: int = 32
    embedding_context_length: int = 512
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    openai_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str | None = None

    # Scheduling
    poll_interval_seconds: int = 3600
    force_reingest: bool = False
    log_level: str = "INFO"

    @property
    def embedding_signature(self) -> str:
        """Identifies how stored vectors were made.

        Part of each book's content hash, so changing the model or the
        document prefix re-embeds every book. The query prefix isn't part
        of it: it doesn't affect stored vectors.
        """
        return f"{self.embedding_model}|{self.embedding_document_prefix}"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        log_level = (env.get("LOG_LEVEL") or cls.log_level).strip().upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        return cls(
            library_catalog_url=_url(
                "LIBRARY_CATALOG_URL",
                _required("LIBRARY_CATALOG_URL", env.get("LIBRARY_CATALOG_URL")),
            ),
            library_api_key=_required("LIBRARY_API_KEY", env.get("LIBRARY_API_KEY")),
            request_timeout_seconds=_int(
                "REQUEST_TIMEOUT_SECONDS",
                env.get("REQUEST_TIMEOUT_SECONDS"),
                cls.request_timeout_seconds,
                minimum=1,
            ),
            qdrant_url=_url(
                "QDRANT_URL", (env.get("QDRANT_URL") or cls.qdrant_url).strip()
            ),
            qdrant_api_key=_optional(env.get("QDRANT_API_KEY")),
            qdrant_collection=(
                env.get("QDRANT_COLLECTION") or cls.qdrant_collection
            ).strip(),
            qdrant_distance=_choice(
                "QDRANT_DISTANCE",
                env.get("QDRANT_DISTANCE"),
                cls.qdrant_distance,
                DISTANCES,
            ),
            embedding_model=(env.get("EMBEDDING_MODEL") or cls.embedding_model).strip(),
            embedding_batch_size=_int(
                "EMBEDDING_BATCH_SIZE",
                env.get("EMBEDDING_BATCH_SIZE"),
                cls.embedding_batch_size,
                minimum=1,
            ),
            embedding_context_length=_int(
                "EMBEDDING_CONTEXT_LENGTH",
                env.get("EMBEDDING_CONTEXT_LENGTH"),
                cls.embedding_context_length,
                minimum=1,
            ),
            embedding_query_prefix=_prefix(
                env.get("EMBEDDING_QUERY_PREFIX"), cls.embedding_query_prefix
            ),
            embedding_document_prefix=_prefix(
                env.get("EMBEDDING_DOCUMENT_PREFIX"), cls.embedding_document_prefix
            ),
            openai_base_url=_url(
                "OPENAI_BASE_URL",
                (env.get("OPENAI_BASE_URL") or cls.openai_base_url).strip(),
            ),
            openai_api_key=_optional(env.get("OPENAI_API_KEY")),
            poll_interval_seconds=_int(
                "POLL_INTERVAL_SECONDS",
                env.get("POLL_INTERVAL_SECONDS"),
                cls.poll_interval_seconds,
                minimum=0,
            ),
            force_reingest=_bool(
                "FORCE_REINGEST", env.get("FORCE_REINGEST"), cls.force_reingest
            ),
            log_level=log_level,
        )
