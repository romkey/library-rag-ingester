import json

import httpx
import pytest

from library_rag_ingester.catalog import (
    Book,
    CatalogClient,
    CatalogError,
    book_from_record,
    parse_catalog,
)

URL = "https://library.example.com/api/v1/catalog"

# A record as the library's Books::CatalogFeed writes it.
RECORD = {
    "id": 7,
    "url": "https://library.example.com/books/7",
    "title": "The Pragmatic Programmer",
    "subtitle": "From Journeyman to Master",
    "authors": ["David Thomas", "Andrew Hunt"],
    "subjects": ["Programming"],
    "isbns": ["9780201616224"],
    "publisher": "Addison-Wesley",
    "published_on": "1999-10-30",
    "language": "en",
    "page_count": 352,
    "location": "Shelf B2",
    "copies_count": 2,
    "description": "Classic advice.",
    "notes": None,
    "ebook_url": None,
    "cover_url": "https://library.example.com/rails/active_storage/blobs/x/c.jpg",
    "metadata_source": "open_library",
    "source_url": None,
    "added_on": "2025-03-04",
    "added_at": "2025-03-04T10:00:00-08:00",
    "updated_at": "2026-11-01T12:00:00-08:00",
    "text": "Title: The Pragmatic Programmer: From Journeyman to Master",
}


def jsonl(*records: object) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def test_book_from_record() -> None:
    book = book_from_record(RECORD)
    assert book.id == "7"
    assert book.authors == ("David Thomas", "Andrew Hunt")
    assert book.page_count == 352
    assert book.notes is None
    assert book.full_title == "The Pragmatic Programmer: From Journeyman to Master"


def test_book_from_sparse_record() -> None:
    book = book_from_record(
        {
            "id": "x",
            "title": " T ",
            "authors": "nope",
            "page_count": True,
            "isbns": [None, " "],
        }
    )
    assert book == Book(id="x", title="T")
    assert book.full_title == "T"


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ([1], "not an object"),
        ({"title": "T"}, "no id"),
        ({"id": 1, "title": ""}, "no title"),
    ],
)
def test_book_from_bad_record(record: object, message: str) -> None:
    with pytest.raises(CatalogError, match=message):
        book_from_record(record)


def test_parse_jsonl_skips_blank_lines() -> None:
    body = jsonl(RECORD, {**RECORD, "id": 8}) + "\n"
    assert [b.id for b in parse_catalog(body)] == ["7", "8"]


def test_parse_json_array() -> None:
    assert [b.id for b in parse_catalog(json.dumps([RECORD]))] == ["7"]


def test_parse_empty() -> None:
    assert parse_catalog("") == []


def test_bad_jsonl_line_fails_the_parse() -> None:
    with pytest.raises(CatalogError, match="line 2"):
        parse_catalog(jsonl(RECORD) + "{not json\n")


def test_bad_json_array_fails_the_parse() -> None:
    with pytest.raises(CatalogError, match="valid JSON"):
        parse_catalog("[{")


def client(handler: httpx.MockTransport) -> CatalogClient:
    return CatalogClient(
        URL, "lib_secret", timeout_seconds=5, transport=handler
    )  # not-a-secret


def test_fetch_sends_key_and_returns_books_and_etag() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=jsonl(RECORD), headers={"ETag": 'W/"v1"'})

    with client(httpx.MockTransport(handle)) as c:
        response = c.fetch()
    assert response.etag == 'W/"v1"'
    assert response.books is not None
    assert [b.id for b in response.books] == ["7"]
    assert seen[0].headers["Authorization"] == "Bearer lib_secret"  # not-a-secret
    assert "If-None-Match" not in seen[0].headers
    assert str(seen[0].url) == URL


def test_fetch_not_modified() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["If-None-Match"] == 'W/"v1"'
        return httpx.Response(304)

    response = client(httpx.MockTransport(handle)).fetch('W/"v1"')
    assert response.books is None
    assert response.etag == 'W/"v1"'


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "LIBRARY_API_KEY"), (403, "LIBRARY_API_KEY"), (500, "HTTP 500")],
)
def test_fetch_errors(status: int, message: str) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status))
    with pytest.raises(CatalogError, match=message):
        client(transport).fetch()


def test_fetch_connection_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(CatalogError, match="couldn't fetch"):
        client(httpx.MockTransport(handle)).fetch()
