import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from library_rag_ingester.catalog import Book, CatalogError, CatalogResponse
from library_rag_ingester.config import Settings
from library_rag_ingester.documents import point_id
from library_rag_ingester.embeddings import Kind
from library_rag_ingester.qdrant_store import QdrantStore
from library_rag_ingester.service import IngestionService
from tests.fakes import FakeCatalog, FakeEmbedder, book


def ok(*books: Book, etag: str | None = "v1") -> CatalogResponse:
    return CatalogResponse(books=list(books), etag=etag)


def service(
    settings: Settings,
    catalog: FakeCatalog,
    store: QdrantStore,
    embedder: FakeEmbedder,
    heartbeat_path: Path | None = None,
) -> IngestionService:
    return IngestionService(
        settings=settings,
        catalog=catalog,
        store=store,
        embedder=embedder,
        heartbeat_path=heartbeat_path,
    )


def test_first_sync_embeds_every_book_in_batches(
    settings: Settings, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    catalog = FakeCatalog(ok(book("1"), book("2"), book("3")))
    stats = service(settings, catalog, store, embedder).run_cycle()

    assert (stats.books, stats.embedded, stats.unchanged, stats.deleted) == (3, 3, 0, 0)
    assert [len(texts) for texts, _ in embedder.calls] == [2, 1]
    assert {kind for _, kind in embedder.calls} == {"document"}
    assert set(store.content_hashes()) == {point_id(i) for i in "123"}


def test_resync_embeds_only_changes_and_deletes_removed_books(
    settings: Settings, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    catalog = FakeCatalog(
        ok(book("1"), book("2"), book("3")),
        ok(book("1"), book("2", location="Shelf A1"), book("4"), etag="v2"),
    )
    svc = service(settings, catalog, store, embedder)
    svc.run_cycle()
    embedder.calls.clear()

    stats = svc.run_cycle()

    assert (stats.embedded, stats.unchanged, stats.deleted) == (2, 1, 1)
    assert sorted(embedder.embedded) == ["Title: Book 2", "Title: Book 4"]
    assert set(store.content_hashes()) == {point_id(i) for i in "124"}
    assert catalog.etags == [None, "v1"]


def test_not_modified_does_nothing(
    settings: Settings, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    catalog = FakeCatalog(ok(book("1")), CatalogResponse(books=None, etag="v1"))
    svc = service(settings, catalog, store, embedder)
    svc.run_cycle()
    embedder.calls.clear()

    stats = svc.run_cycle()

    assert stats.not_modified
    assert embedder.calls == []
    assert len(store.content_hashes()) == 1


def test_changed_model_reembeds_everything(
    settings: Settings, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    service(settings, FakeCatalog(ok(book("1"))), store, embedder).run_cycle()
    embedder.calls.clear()

    other = replace(settings, embedding_model="another-model")
    stats = service(other, FakeCatalog(ok(book("1"))), store, embedder).run_cycle()

    assert stats.embedded == 1


def test_force_reingest_ignores_etag_and_hashes_once(
    settings: Settings, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    service(settings, FakeCatalog(ok(book("1"))), store, embedder).run_cycle()
    forced = replace(settings, force_reingest=True)
    catalog = FakeCatalog(ok(book("1")), ok(book("1"), etag="v2"))
    svc = service(forced, catalog, store, embedder)

    assert svc.run_cycle().embedded == 1
    assert svc.run_cycle().embedded == 0
    assert catalog.etags == [None, "v1"]


def test_empty_catalog_deletes_nothing(
    settings: Settings,
    store: QdrantStore,
    embedder: FakeEmbedder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = FakeCatalog(ok(book("1")), ok(), ok())
    svc = service(settings, catalog, store, embedder)
    svc.run_cycle()

    with caplog.at_level(logging.WARNING):
        assert svc.run_cycle().deleted == 0
    assert "Catalog is empty; keeping the 1 stored books" in caplog.text
    assert len(store.content_hashes()) == 1


def test_empty_catalog_and_empty_store(
    settings: Settings, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    assert service(settings, FakeCatalog(ok()), store, embedder).run_cycle().books == 0


def test_duplicate_books_are_stored_once(
    settings: Settings,
    store: QdrantStore,
    embedder: FakeEmbedder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = FakeCatalog(ok(book("1"), book("1", title="Later")))
    stats = service(settings, catalog, store, embedder).run_cycle()
    assert stats.books == 1
    assert "appears twice" in caplog.text


def test_stop_mid_sync_skips_deletes_and_keeps_etag(
    settings: Settings, store: QdrantStore
) -> None:
    class StoppingEmbedder(FakeEmbedder):
        """Asks the service to stop as soon as it embeds anything."""

        def embed(
            self, texts: Sequence[str], kind: Kind = "document"
        ) -> list[list[float]]:
            svc.stop()
            return super().embed(texts, kind)

    catalog = FakeCatalog(
        ok(book("1")), ok(book("2"), book("3"), book("4"), etag="v2"), ok(book("2"))
    )
    service(settings, catalog, store, FakeEmbedder()).run_cycle()
    svc = service(settings, catalog, store, StoppingEmbedder())
    svc._etag = "v1"  # as if this service had run the first cycle

    stats = svc.run_cycle()

    assert stats.stopped
    assert stats.embedded == 2  # one batch, then the stop
    assert point_id("1") in store.content_hashes()  # not deleted
    svc.run_cycle()
    assert catalog.etags == [None, "v1", "v1"]  # the stopped cycle's etag unused


def test_run_forever_retries_failures_and_beats(
    settings: Settings,
    store: QdrantStore,
    embedder: FakeEmbedder,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beat = tmp_path / "heartbeat"
    calls = 0

    class FlakyCatalog(FakeCatalog):
        def fetch(self, etag: str | None = None) -> CatalogResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise CatalogError("library down")
            svc.stop()
            return ok(book("1"))

    svc = service(settings, FlakyCatalog(), store, embedder, heartbeat_path=beat)
    waits: list[float | None] = []

    def fake_wait(timeout: float | None = None) -> bool:
        waits.append(timeout)
        return False

    # Don't really sleep.
    monkeypatch.setattr(svc._stop, "wait", fake_wait)

    svc.run_forever()

    assert calls == 2
    assert "Sync failed, will retry: library down" in caplog.text
    assert beat.exists()
    # 3600 s in one-minute steps between the failed cycle and the next.
    assert waits == [60.0] * 60
