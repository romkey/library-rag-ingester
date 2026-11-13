"""The sync cycle: catalog in, changed books embedded, vanished books removed."""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from library_rag_ingester import heartbeat
from library_rag_ingester.catalog import Book, CatalogError, CatalogResponse
from library_rag_ingester.config import Settings
from library_rag_ingester.documents import Document, build_document
from library_rag_ingester.embeddings import Embedder, EmbeddingError
from library_rag_ingester.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# A cycle that fails with one of these is logged and retried next interval.
CYCLE_ERRORS = (
    CatalogError,
    EmbeddingError,
    httpx.HTTPError,
    ResponseHandlingException,
    UnexpectedResponse,
)

_HEARTBEAT_SECONDS = 60


class Catalog(Protocol):
    def fetch(self, etag: str | None = None) -> CatalogResponse: ...


@dataclass
class SyncStats:
    books: int = 0
    embedded: int = 0
    unchanged: int = 0
    deleted: int = 0
    not_modified: bool = False
    stopped: bool = False


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        catalog: Catalog,
        store: QdrantStore,
        embedder: Embedder,
        heartbeat_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._store = store
        self._embedder = embedder
        self._heartbeat_path = heartbeat_path
        self._stop = threading.Event()
        self._etag: str | None = None
        self._force = settings.force_reingest

    def stop(self) -> None:
        """Finish the current batch, skip deletes, and leave run_forever."""
        self._stop.set()

    def _beat(self) -> None:
        if self._heartbeat_path is not None:
            heartbeat.touch(self._heartbeat_path)

    def run_forever(self) -> None:
        """Sync every POLL_INTERVAL_SECONDS until stop() is called."""
        while not self._stop.is_set():
            try:
                self.run_cycle()
            except CYCLE_ERRORS as exc:
                logger.error("Sync failed, will retry: %s", exc)
            self._sleep(self._settings.poll_interval_seconds)

    def _sleep(self, seconds: int) -> None:
        remaining = float(seconds)
        while remaining > 0 and not self._stop.is_set():
            self._beat()
            step = min(remaining, _HEARTBEAT_SECONDS)
            self._stop.wait(step)
            remaining -= step

    def run_cycle(self) -> SyncStats:
        self._beat()
        stats = SyncStats()
        response = self._catalog.fetch(None if self._force else self._etag)
        if response.books is None:
            logger.info("Catalog unchanged since the last sync")
            stats.not_modified = True
            return stats

        docs = self._documents(response.books)
        stats.books = len(docs)
        stored = self._store.content_hashes()
        changed = [
            doc
            for doc in docs
            if self._force or stored.get(doc.point_id) != doc.content_hash
        ]
        stats.unchanged = len(docs) - len(changed)

        for start in range(0, len(changed), self._settings.embedding_batch_size):
            if self._stop.is_set():
                stats.stopped = True
                logger.info("Stopped mid-sync; nothing deleted")
                return stats
            self._beat()
            batch = changed[start : start + self._settings.embedding_batch_size]
            vectors = self._embedder.embed([doc.text for doc in batch], "document")
            stats.embedded += self._store.upsert(batch, vectors)

        stats.deleted = self._prune(docs, stored)
        # Only now: a failed cycle must re-fetch and retry next time.
        self._etag = response.etag
        self._force = False
        logger.info(
            "Synced %d books: %d embedded, %d unchanged, %d deleted",
            stats.books,
            stats.embedded,
            stats.unchanged,
            stats.deleted,
        )
        return stats

    def _documents(self, books: Sequence[Book]) -> list[Document]:
        signature = self._settings.embedding_signature
        by_id: dict[str, Document] = {}
        for book in books:
            doc = build_document(book, signature)
            if doc.book_id in by_id:
                logger.warning("Book %s appears twice in the catalog", doc.book_id)
            by_id[doc.book_id] = doc
        return list(by_id.values())

    def _prune(self, docs: list[Document], stored: dict[str, str]) -> int:
        """Delete stored books that are no longer in the catalog."""
        if not docs:
            # An empty catalog is more likely a broken library than an empty
            # one; don't wipe the collection on its say-so.
            if stored:
                logger.warning(
                    "Catalog is empty; keeping the %d stored books", len(stored)
                )
            return 0
        current = {doc.point_id for doc in docs}
        stale = [pid for pid in stored if pid not in current]
        if stale:
            logger.info("Deleting %d books no longer in the catalog", len(stale))
        return self._store.delete(stale) if stale else 0
