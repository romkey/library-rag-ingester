from dataclasses import replace

from library_rag_ingester.catalog import Book
from library_rag_ingester.documents import (
    build_document,
    build_payload,
    build_text,
    point_id,
)

BOOK = Book(
    id="7",
    title="The Pragmatic Programmer",
    subtitle="From Journeyman to Master",
    authors=("David Thomas", "Andrew Hunt"),
    subjects=("Programming",),
    isbns=("9780201616224",),
    published_on="1999-10-30",
    page_count=352,
    location="Shelf B2",
    description="Classic advice.",
)


def test_text_from_feed_is_used_as_is() -> None:
    assert build_text(replace(BOOK, text="From the feed")) == "From the feed"


def test_text_built_like_the_feed_when_missing() -> None:
    assert build_text(BOOK) == (
        "Title: The Pragmatic Programmer: From Journeyman to Master\n"
        "Authors: David Thomas, Andrew Hunt\n"
        "Subjects: Programming\n"
        "Published: 1999-10-30\n"
        "Pages: 352\n"
        "ISBN: 9780201616224\n"
        "Location: Shelf B2\n"
        "Description: Classic advice."
    )


def test_payload() -> None:
    payload = build_payload(BOOK)
    assert payload["source"] == "library"
    assert payload["book_id"] == "7"
    assert payload["authors"] == ["David Thomas", "Andrew Hunt"]
    assert payload["published_year"] == "1999"
    assert build_payload(replace(BOOK, published_on=None))["published_year"] is None


def test_point_id_is_stable_per_book() -> None:
    assert point_id("7") == point_id("7") != point_id("8")
    assert build_document(BOOK, "sig").point_id == point_id("7")


def test_content_hash_tracks_text_payload_and_signature() -> None:
    base = build_document(BOOK, "sig").content_hash
    assert build_document(BOOK, "sig").content_hash == base
    assert build_document(BOOK, "other").content_hash != base
    assert (
        build_document(replace(BOOK, location="Shelf A1"), "sig").content_hash != base
    )
    assert build_document(replace(BOOK, cover_url="c"), "sig").content_hash != base
