import io

from library_rag_ingester.documents import build_document
from library_rag_ingester.embeddings import Kind
from library_rag_ingester.qdrant_store import QdrantStore
from library_rag_ingester.query import format_hits, search
from tests.fakes import FakeEmbedder, book, vector


def test_search_embeds_as_query(store: QdrantStore, embedder: FakeEmbedder) -> None:
    docs = [build_document(book(str(i)), "sig") for i in range(3)]
    store.upsert(docs, [vector(d.text) for d in docs])

    # The fake embeds a text the same way whatever its kind, so querying
    # with a stored text finds that book first.
    hits = search(store, embedder, "Title: Book 2", limit=2)
    assert hits[0][1]["book_id"] == "2"
    assert len(hits) == 2
    kinds: list[Kind] = [kind for _, kind in embedder.calls]
    assert kinds == ["query"]


def test_format_hits() -> None:
    out = io.StringIO()
    payload = {
        "full_title": "Make: Electronics",
        "authors": ["Charles Platt"],
        "location": "Shelf A1",
        "url": "https://library.example.com/books/3",
        "text": "Title: Make: Electronics",
    }
    format_hits([(0.91234, payload), (0.5, {})], out, show_text=True, width=60)
    assert out.getvalue() == (
        " 1. 0.912  Make: Electronics (Charles Platt)\n"
        "      Location: Shelf A1\n"
        "      https://library.example.com/books/3\n"
        "      Title: Make: Electronics\n"
        "\n"
        " 2. 0.500  ?\n"
        "\n"
    )


def test_format_no_hits() -> None:
    out = io.StringIO()
    format_hits([], out)
    assert out.getvalue() == "No results.\n"
