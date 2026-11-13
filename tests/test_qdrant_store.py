import logging

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct

from library_rag_ingester.documents import build_document
from library_rag_ingester.qdrant_store import QdrantStore
from tests.fakes import book, vector


def test_upsert_hashes_search_and_delete(store: QdrantStore) -> None:
    docs = [build_document(book(str(i)), "sig") for i in range(3)]
    assert store.upsert(docs, [vector(d.text) for d in docs]) == 3

    assert store.content_hashes() == {d.point_id: d.content_hash for d in docs}

    hits = store.search(vector(docs[1].text), limit=1)
    assert hits[0][1]["book_id"] == "1"
    assert hits[0][1]["text"] == "Title: Book 1"
    assert "ingested_at" in hits[0][1]

    assert store.delete([docs[0].point_id]) == 1
    assert set(store.content_hashes()) == {docs[1].point_id, docs[2].point_id}


def test_content_hashes_ignores_other_sources(store: QdrantStore) -> None:
    store.client.upsert(
        "test", [PointStruct(id=1, vector=[1.0] * 8, payload={"source": "other"})]
    )
    assert store.content_hashes() == {}


def test_content_hashes_pages(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("library_rag_ingester.qdrant_store._BATCH", 2)
    docs = [build_document(book(str(i)), "sig") for i in range(5)]
    store.upsert(docs, [vector(d.text) for d in docs])
    assert len(store.content_hashes()) == 5


def test_upsert_length_mismatch(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        store.upsert([build_document(book(), "sig")], [])


def test_existing_collection_with_other_size(store: QdrantStore) -> None:
    store.ensure_collection(vector_size=8, distance="cosine")  # same size: fine
    with pytest.raises(ValueError, match="vector size 8"):
        store.ensure_collection(vector_size=16, distance="cosine")


def test_named_vectors_are_not_checked() -> None:
    client = QdrantClient(":memory:")
    client.create_collection("named", vectors_config={})
    QdrantStore(collection="named", client=client).ensure_collection(
        vector_size=8, distance="dot"
    )


def test_payload_index_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = QdrantStore(collection="x", client=QdrantClient(":memory:"))

    def refuse(**kwargs: object) -> None:
        raise UnexpectedResponse(400, "Bad Request", b"nope", httpx.Headers())

    monkeypatch.setattr(store.client, "create_payload_index", refuse)
    with caplog.at_level(logging.WARNING):
        store.ensure_collection(vector_size=8, distance="euclid")
    assert "Payload index 'source' not created" in caplog.text
