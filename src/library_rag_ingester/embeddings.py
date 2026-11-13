"""Embeddings from an OpenAI-compatible `/embeddings` API.

That is OpenAI itself or a local server that speaks its API
(text-embeddings-inference, Ollama's /v1, vLLM, LM Studio). Request and
response shapes: https://platform.openai.com/docs/api-reference/embeddings
"""

import logging
from collections.abc import Sequence
from typing import Literal, Protocol

import httpx

logger = logging.getLogger(__name__)

type Kind = Literal["query", "document"]

# Conservative, so truncated text stays inside the model's context.
_CHARS_PER_TOKEN = 3


class Embedder(Protocol):
    """Texts in, vectors out, embedded as documents or as search queries."""

    def embed(
        self, texts: Sequence[str], kind: Kind = "document"
    ) -> list[list[float]]: ...


class EmbeddingError(Exception):
    """The embeddings API failed or returned something unusable."""


def prepare(text: str, prefix: str, max_tokens: int) -> str:
    """`prefix` + `text`, truncated so the whole fits in about `max_tokens`."""
    max_chars = max(1, max_tokens * _CHARS_PER_TOKEN - len(prefix))
    if len(text) > max_chars:
        logger.warning(
            "Truncating text from %d to %d chars (~%d token limit)",
            len(text),
            max_chars,
            max_tokens,
        )
        text = text[:max_chars]
    return prefix + text


def _is_vector(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(x, int | float) and not isinstance(x, bool) for x in value)
    )


def parse_embeddings(data: object, count: int) -> list[list[float]]:
    """The vectors from an /embeddings response body, in input order."""
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("index", 0), int)
        and _is_vector(item.get("embedding"))
        for item in items
    ):
        raise EmbeddingError("embeddings API returned no list of embeddings")
    if len(items) != count:
        raise EmbeddingError(
            f"embeddings API returned {len(items)} embeddings for {count} inputs"
        )
    ordered = sorted(items, key=lambda item: item.get("index", 0))
    return [[float(x) for x in item["embedding"]] for item in ordered]


class OpenAIEmbedder:
    """Embeds documents and queries, each with its model prompt prefix."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout_seconds: int = 120,
        batch_size: int = 32,
        max_tokens: int = 512,
        query_prefix: str = "",
        document_prefix: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._prefixes: dict[Kind, str] = {
            "query": query_prefix,
            "document": document_prefix,
        }
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            timeout=timeout_seconds,
            transport=transport,
        )
        self._dimension: int | None = None

    def close(self) -> None:
        self._client.close()

    @property
    def dimension(self) -> int:
        """The model's vector size, asked of the API on first use."""
        if self._dimension is None:
            self._dimension = len(self._request(["dimension probe"])[0])
        return self._dimension

    def embed(self, texts: Sequence[str], kind: Kind = "document") -> list[list[float]]:
        prefix = self._prefixes[kind]
        prepared = [prepare(text, prefix, self._max_tokens) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self._batch_size):
            vectors.extend(self._request(prepared[start : start + self._batch_size]))
        return vectors

    def _request(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.post(
                "/embeddings", json={"model": self.model, "input": texts}
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embeddings API request failed: {exc}") from exc
        except ValueError as exc:  # the body isn't JSON
            raise EmbeddingError(f"embeddings API returned non-JSON: {exc}") from exc
        return parse_embeddings(body, len(texts))
