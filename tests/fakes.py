"""Stand-ins for the catalog and the embeddings API."""

import hashlib
from collections.abc import Sequence

from library_rag_ingester.catalog import Book, CatalogResponse
from library_rag_ingester.embeddings import Kind

DIMENSION = 8


class FakeEmbedder:
    """Deterministic vectors from a hash of each text; records what it saw."""

    dimension = DIMENSION

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Kind]] = []

    def embed(self, texts: Sequence[str], kind: Kind = "document") -> list[list[float]]:
        self.calls.append((list(texts), kind))
        return [vector(text) for text in texts]

    @property
    def embedded(self) -> list[str]:
        return [text for texts, _ in self.calls for text in texts]


def vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 + 0.01 for b in digest[:DIMENSION]]


class FakeCatalog:
    """Serves queued responses; records the ETag each fetch sent."""

    def __init__(self, *responses: CatalogResponse) -> None:
        self.responses = list(responses)
        self.etags: list[str | None] = []

    def fetch(self, etag: str | None = None) -> CatalogResponse:
        self.etags.append(etag)
        return self.responses.pop(0)

    def __enter__(self) -> FakeCatalog:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


def book(
    book_id: str = "1", *, title: str | None = None, location: str | None = None
) -> Book:
    return Book(
        id=book_id,
        title=title or f"Book {book_id}",
        authors=("Ada Lovelace",),
        location=location,
        text=f"Title: Book {book_id}",
    )
