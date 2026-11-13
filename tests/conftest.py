import pytest
from qdrant_client import QdrantClient

from library_rag_ingester.config import Settings
from library_rag_ingester.qdrant_store import QdrantStore
from tests.fakes import DIMENSION, FakeEmbedder


@pytest.fixture
def settings() -> Settings:
    return Settings(
        library_catalog_url="https://library.example.com/api/v1/catalog",
        library_api_key="test-key",  # not-a-secret
        embedding_batch_size=2,
    )


@pytest.fixture
def store() -> QdrantStore:
    """An empty in-memory Qdrant collection with 8-dimensional vectors."""
    s = QdrantStore(collection="test", client=QdrantClient(":memory:"))
    s.ensure_collection(vector_size=DIMENSION, distance="cosine")
    return s


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
