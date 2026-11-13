"""The library's catalog feed: fetching it and parsing its records.

The feed is `GET /api/v1/catalog` on the PDX Hackerspace library app
(https://github.com/pdxhackerspace/library): JSON Lines, one book per line,
authenticated with `Authorization: Bearer <API key>`, with an `ETag` that
answers `If-None-Match` with 304. The record fields come from
`app/services/books/catalog_feed.rb` there; check it rather than guessing.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from library_rag_ingester import __version__

logger = logging.getLogger(__name__)


class CatalogError(Exception):
    """The catalog couldn't be fetched, or its contents can't be used."""


@dataclass(frozen=True)
class Book:
    """One catalog record. Strings the feed leaves out or nulls are None."""

    id: str
    title: str
    url: str | None = None
    subtitle: str | None = None
    authors: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    isbns: tuple[str, ...] = ()
    publisher: str | None = None
    published_on: str | None = None
    language: str | None = None
    page_count: int | None = None
    location: str | None = None
    copies_count: int | None = None
    description: str | None = None
    notes: str | None = None
    ebook_url: str | None = None
    cover_url: str | None = None
    metadata_source: str | None = None
    source_url: str | None = None
    added_on: str | None = None
    added_at: str | None = None
    updated_at: str | None = None
    # Ready-to-embed text the feed builds from the fields above.
    text: str | None = None

    @property
    def full_title(self) -> str:
        return f"{self.title}: {self.subtitle}" if self.subtitle else self.title


@dataclass(frozen=True)
class CatalogResponse:
    """A fetch's result. `books` is None when the catalog is unchanged (304)."""

    books: list[Book] | None
    etag: str | None


# Any: a record is JSON from the feed; each field is checked as it's read.
def _str(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _strs(record: dict[str, Any], key: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(str(v).strip() for v in value if v is not None and str(v).strip())


def book_from_record(record: object) -> Book:
    """A Book from one feed record; CatalogError if it has no id or title."""
    if not isinstance(record, dict):
        raise CatalogError(f"catalog record is {type(record).__name__}, not an object")
    book_id = _str(record, "id")
    if book_id is None:
        raise CatalogError("catalog record has no id")
    title = _str(record, "title")
    if title is None:
        raise CatalogError(f"catalog record {book_id} has no title")
    return Book(
        id=book_id,
        title=title,
        url=_str(record, "url"),
        subtitle=_str(record, "subtitle"),
        authors=_strs(record, "authors"),
        subjects=_strs(record, "subjects"),
        isbns=_strs(record, "isbns"),
        publisher=_str(record, "publisher"),
        published_on=_str(record, "published_on"),
        language=_str(record, "language"),
        page_count=_int(record, "page_count"),
        location=_str(record, "location"),
        copies_count=_int(record, "copies_count"),
        description=_str(record, "description"),
        notes=_str(record, "notes"),
        ebook_url=_str(record, "ebook_url"),
        cover_url=_str(record, "cover_url"),
        metadata_source=_str(record, "metadata_source"),
        source_url=_str(record, "source_url"),
        added_on=_str(record, "added_on"),
        added_at=_str(record, "added_at"),
        updated_at=_str(record, "updated_at"),
        text=_str(record, "text"),
    )


def parse_catalog(body: str) -> list[Book]:
    """Books from the feed body.

    The feed is JSON Lines. A JSON array of records is accepted too, for a
    catalog exported to a plain JSON file. Any unreadable record fails the
    whole parse: a partial catalog would delete the missing books.
    """
    stripped = body.strip()
    if stripped.startswith("["):
        try:
            records = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"catalog isn't valid JSON: {exc}") from None
        return [book_from_record(record) for record in records]

    books: list[Book] = []
    for number, line in enumerate(body.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CatalogError(
                f"catalog line {number} isn't valid JSON: {exc}"
            ) from None
        books.append(book_from_record(record))
    return books


class CatalogClient:
    """Fetches the catalog feed with the library API key."""

    def __init__(
        self,
        url: str,
        api_key: str,
        timeout_seconds: int = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = url
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/jsonl, application/json",
                "User-Agent": f"library-rag-ingester/{__version__}",
            },
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> CatalogClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def fetch(self, etag: str | None = None) -> CatalogResponse:
        """The whole catalog, or books=None if it hasn't changed since `etag`."""
        headers = {"If-None-Match": etag} if etag else {}
        try:
            response = self._client.get(self._url, headers=headers)
        except httpx.HTTPError as exc:
            raise CatalogError(f"couldn't fetch the catalog: {exc}") from exc

        if response.status_code == httpx.codes.NOT_MODIFIED:
            return CatalogResponse(books=None, etag=etag)
        if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise CatalogError(
                f"the library refused the API key (HTTP {response.status_code}); "
                "check LIBRARY_API_KEY"
            )
        if response.is_error:
            raise CatalogError(f"the catalog returned HTTP {response.status_code}")

        books = parse_catalog(response.text)
        logger.info("Fetched %d books from the catalog", len(books))
        return CatalogResponse(books=books, etag=response.headers.get("ETag"))
