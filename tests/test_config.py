import pytest

from library_rag_ingester.config import ConfigError, Settings

REQUIRED = {
    "LIBRARY_CATALOG_URL": "https://library.example.com/api/v1/catalog",
    "LIBRARY_API_KEY": "lib_test",  # not-a-secret
}


def test_defaults() -> None:
    settings = Settings.from_env(REQUIRED)
    assert settings == Settings(
        library_catalog_url="https://library.example.com/api/v1/catalog",
        library_api_key="lib_test",  # not-a-secret
    )
    assert settings.embedding_document_prefix == "search_document: "


def test_reads_environment() -> None:
    env = {
        **REQUIRED,
        "REQUEST_TIMEOUT_SECONDS": "5",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "qk",  # not-a-secret
        "QDRANT_COLLECTION": "books",
        "QDRANT_DISTANCE": "Dot",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_BATCH_SIZE": "8",
        "EMBEDDING_CONTEXT_LENGTH": "8191",
        "EMBEDDING_QUERY_PREFIX": "none",
        "EMBEDDING_DOCUMENT_PREFIX": "Instruct: books\\nDoc:",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": "sk",  # not-a-secret
        "POLL_INTERVAL_SECONDS": "0",
        "FORCE_REINGEST": "yes",
        "LOG_LEVEL": "debug",
    }
    settings = Settings.from_env(env)
    assert settings.request_timeout_seconds == 5
    assert settings.qdrant_api_key == "qk"  # not-a-secret
    assert settings.qdrant_collection == "books"
    assert settings.qdrant_distance == "dot"
    assert settings.embedding_batch_size == 8
    assert settings.embedding_query_prefix == ""
    assert settings.embedding_document_prefix == "Instruct: books\nDoc: "
    assert settings.openai_api_key == "sk"  # not-a-secret
    assert settings.poll_interval_seconds == 0
    assert settings.force_reingest is True
    assert settings.log_level == "DEBUG"
    assert settings.embedding_signature == (
        "text-embedding-3-small|Instruct: books\nDoc: "
    )


def test_prefix_ending_in_whitespace_is_kept() -> None:
    settings = Settings.from_env({**REQUIRED, "EMBEDDING_QUERY_PREFIX": "q:\\n"})
    assert settings.embedding_query_prefix == "q:\n"


@pytest.mark.parametrize("name", ["LIBRARY_CATALOG_URL", "LIBRARY_API_KEY"])
def test_requires(name: str) -> None:
    env = {k: v for k, v in REQUIRED.items() if k != name}
    with pytest.raises(ConfigError, match=f"{name} is required"):
        Settings.from_env(env)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LIBRARY_CATALOG_URL", "library.example.com", "http://"),
        ("QDRANT_URL", "qdrant:6333", "http://"),
        ("REQUEST_TIMEOUT_SECONDS", "soon", "whole number"),
        ("EMBEDDING_BATCH_SIZE", "0", "at least 1"),
        ("POLL_INTERVAL_SECONDS", "-1", "at least 0"),
        ("QDRANT_DISTANCE", "near", "one of"),
        ("FORCE_REINGEST", "maybe", "true or false"),
        ("LOG_LEVEL", "loud", "logging level"),
    ],
)
def test_rejects_bad_values(name: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=f"{name}.*{message}|{message}.*{name}"):
        Settings.from_env({**REQUIRED, name: value})


def test_false_values() -> None:
    assert not Settings.from_env({**REQUIRED, "FORCE_REINGEST": "off"}).force_reingest
