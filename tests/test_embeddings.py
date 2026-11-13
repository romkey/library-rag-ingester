import json
from collections.abc import Callable

import httpx
import pytest

from library_rag_ingester.embeddings import (
    EmbeddingError,
    OpenAIEmbedder,
    parse_embeddings,
    prepare,
)

type Handler = Callable[[httpx.Request], httpx.Response]


def echo_lengths(requests: list[dict[str, object]]) -> Handler:
    """An /embeddings API whose vectors are [len(text), 1.0], in reverse order."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        data = [
            {"index": i, "embedding": [len(text), 1.0]}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    return handle


def embedder(
    handle: Handler,
    *,
    api_key: str | None = None,
    batch_size: int = 32,
    query_prefix: str = "",
    document_prefix: str = "",
) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        "model-x",
        "http://embed.test/v1/",
        api_key,
        batch_size=batch_size,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        transport=httpx.MockTransport(handle),
    )


def test_prepare_adds_prefix_and_truncates() -> None:
    assert prepare("hello", "q: ", 100) == "q: hello"
    assert prepare("x" * 50, "p: ", 10) == "p: " + "x" * 27


def test_embed_batches_prefixes_and_orders() -> None:
    requests: list[dict[str, object]] = []
    e = embedder(
        echo_lengths(requests), batch_size=2, query_prefix="q: ", document_prefix="d: "
    )
    assert e.embed(["a", "bb", "ccc"]) == [[4.0, 1.0], [5.0, 1.0], [6.0, 1.0]]
    assert [r["input"] for r in requests] == [["d: a", "d: bb"], ["d: ccc"]]
    assert requests[0]["model"] == "model-x"
    assert e.embed(["a"], "query") == [[4.0, 1.0]]
    assert e.embed([]) == []
    e.close()


def test_dimension_is_probed_once() -> None:
    requests: list[dict[str, object]] = []
    e = embedder(echo_lengths(requests))
    assert e.dimension == 2
    assert e.dimension == 2
    assert len(requests) == 1


def test_sends_api_key() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test"  # not-a-secret
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [{"embedding": [1]}]})

    assert embedder(handle, api_key="sk-test").embed(["a"]) == [[1.0]]  # not-a-secret


def test_http_error() -> None:
    with pytest.raises(EmbeddingError, match="request failed"):
        embedder(lambda request: httpx.Response(500)).embed(["a"])


def test_non_json() -> None:
    with pytest.raises(EmbeddingError, match="non-JSON"):
        embedder(lambda request: httpx.Response(200, text="<html>")).embed(["a"])


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"data": "x"},
        {"data": [{"embedding": []}]},
        {"data": [{"embedding": [True]}]},
        {"data": [{"index": "0", "embedding": [1]}]},
    ],
)
def test_parse_rejects_malformed(body: object) -> None:
    with pytest.raises(EmbeddingError, match="no list"):
        parse_embeddings(body, 1)


def test_parse_rejects_wrong_count() -> None:
    with pytest.raises(EmbeddingError, match="2 embeddings for 1"):
        parse_embeddings({"data": [{"embedding": [1]}, {"embedding": [2]}]}, 1)
