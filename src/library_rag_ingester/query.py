"""`library-rag-ingester query`: semantic search over the collection.

library-rag-ingester query "intro to soldering"
library-rag-ingester query --limit 10 --show-text "machine learning"
"""

import textwrap
from typing import Any, TextIO

from library_rag_ingester.embeddings import Embedder
from library_rag_ingester.qdrant_store import QdrantStore

# Any: point payloads are free-form JSON stored by this app.
type Hit = tuple[float, dict[str, Any]]


def search(
    store: QdrantStore,
    embedder: Embedder,
    text: str,
    *,
    limit: int = 5,
    min_score: float | None = None,
) -> list[Hit]:
    vector = embedder.embed([text], "query")[0]
    return store.search(vector, limit=limit, min_score=min_score)


def format_hits(
    hits: list[Hit], out: TextIO, *, show_text: bool = False, width: int = 100
) -> None:
    if not hits:
        out.write("No results.\n")
        return
    for rank, (score, payload) in enumerate(hits, 1):
        title = payload.get("full_title") or payload.get("title") or "?"
        authors = ", ".join(payload.get("authors") or [])
        line = f"{rank:>2}. {score:.3f}  {title}"
        if authors:
            line += f" ({authors})"
        out.write(line + "\n")
        if payload.get("location"):
            out.write(f"      Location: {payload['location']}\n")
        if payload.get("url"):
            out.write(f"      {payload['url']}\n")
        if show_text and payload.get("text"):
            out.write(
                textwrap.fill(
                    str(payload["text"]),
                    width=width,
                    initial_indent="      ",
                    subsequent_indent="      ",
                )
                + "\n"
            )
        out.write("\n")
