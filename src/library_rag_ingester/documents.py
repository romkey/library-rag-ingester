"""Books to Qdrant documents: the text to embed and the payload to store.

Pure functions, no I/O. Changing the text or payload layout means bumping
DOCUMENT_SCHEMA_VERSION so every stored book is rebuilt on the next sync.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from library_rag_ingester.catalog import Book

DOCUMENT_SCHEMA_VERSION = 1
SOURCE = "library"
CHUNK_TYPE = "book"

# Labels and order follow the feed's own `text` field (TEXT_FIELDS in the
# library's catalog_feed.rb), used when a record doesn't carry one.
_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("Title", "full_title"),
    ("Authors", "authors"),
    ("Subjects", "subjects"),
    ("Publisher", "publisher"),
    ("Published", "published_on"),
    ("Language", "language"),
    ("Pages", "page_count"),
    ("ISBN", "isbns"),
    ("Location", "location"),
    ("Added to library", "added_on"),
    ("Description", "description"),
    ("Notes", "notes"),
)


# Any: payload values are JSON (strings, numbers, lists, null).
type Payload = dict[str, Any]


@dataclass(frozen=True)
class Document:
    """One point: a book's embedding text and payload."""

    book_id: str
    text: str
    payload: Payload
    content_hash: str

    @property
    def point_id(self) -> str:
        return point_id(self.book_id)


def point_id(book_id: str) -> str:
    """The book's Qdrant point id; stable, so re-syncs overwrite in place."""
    return str(uuid5(NAMESPACE_URL, f"{SOURCE}:{CHUNK_TYPE}:{book_id}"))


def build_text(book: Book) -> str:
    """The feed's ready-to-embed text, or the same layout built from fields."""
    if book.text:
        return book.text
    lines: list[str] = []
    for label, attribute in _TEXT_FIELDS:
        value = getattr(book, attribute)
        if isinstance(value, tuple):
            value = ", ".join(value)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def build_payload(book: Book) -> Payload:
    published_year = (book.published_on or "")[:4]
    return {
        "source": SOURCE,
        "chunk_type": CHUNK_TYPE,
        "book_id": book.id,
        "url": book.url,
        "title": book.title,
        "subtitle": book.subtitle,
        "full_title": book.full_title,
        "authors": list(book.authors),
        "subjects": list(book.subjects),
        "isbns": list(book.isbns),
        "publisher": book.publisher,
        "published_on": book.published_on,
        "published_year": published_year if published_year.isdigit() else None,
        "language": book.language,
        "page_count": book.page_count,
        "location": book.location,
        "copies_count": book.copies_count,
        "description": book.description,
        "notes": book.notes,
        "ebook_url": book.ebook_url,
        "cover_url": book.cover_url,
        "metadata_source": book.metadata_source,
        "source_url": book.source_url,
        "added_on": book.added_on,
        "added_at": book.added_at,
        "updated_at": book.updated_at,
    }


def content_hash(text: str, payload: Payload, embedding_signature: str) -> str:
    """Changes whenever the stored point would: text, payload or embedding setup."""
    material = json.dumps(
        [DOCUMENT_SCHEMA_VERSION, embedding_signature, text, payload],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def build_document(book: Book, embedding_signature: str) -> Document:
    text = build_text(book)
    payload = build_payload(book)
    return Document(
        book_id=book.id,
        text=text,
        payload=payload,
        content_hash=content_hash(text, payload, embedding_signature),
    )
