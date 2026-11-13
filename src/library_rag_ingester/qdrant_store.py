"""The Qdrant collection: setup, change detection, upserts, deletes, search."""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    ExtendedPointId,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from library_rag_ingester.documents import SOURCE, Document

logger = logging.getLogger(__name__)

# Any: point payloads are free-form JSON that this app writes.
type Payload = dict[str, Any]

_BATCH = 256

_DISTANCES: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
    "manhattan": Distance.MANHATTAN,
}

PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,
    "chunk_type": PayloadSchemaType.KEYWORD,
    "book_id": PayloadSchemaType.KEYWORD,
    "authors": PayloadSchemaType.KEYWORD,
    "subjects": PayloadSchemaType.KEYWORD,
    "isbns": PayloadSchemaType.KEYWORD,
    "location": PayloadSchemaType.KEYWORD,
    "language": PayloadSchemaType.KEYWORD,
    "published_year": PayloadSchemaType.KEYWORD,
    "added_at": PayloadSchemaType.DATETIME,
    "updated_at": PayloadSchemaType.DATETIME,
}


def _source_filter() -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=SOURCE))])


class QdrantStore:
    def __init__(
        self,
        *,
        collection: str,
        url: str | None = None,
        api_key: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection = collection
        self.client = client or QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, *, vector_size: int, distance: str) -> None:
        """Create the collection and payload indexes if they're missing.

        Raises ValueError if the collection exists with another vector size,
        i.e. it was made with a different embedding model.
        """
        if not self.client.collection_exists(self.collection):
            logger.info(
                "Creating Qdrant collection %r (size=%d, distance=%s)",
                self.collection,
                vector_size,
                distance,
            )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size, distance=_DISTANCES[distance]
                ),
            )
        else:
            existing = self._vector_size()
            if existing is not None and existing != vector_size:
                raise ValueError(
                    f"Qdrant collection {self.collection!r} has vector size "
                    f"{existing}, but the embedding model produces {vector_size}. "
                    "Use a new QDRANT_COLLECTION or delete the collection."
                )
        for field_name, schema in PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except UnexpectedResponse as exc:
                logger.warning("Payload index %r not created: %s", field_name, exc)

    def _vector_size(self) -> int | None:
        vectors = self.client.get_collection(self.collection).config.params.vectors
        if isinstance(vectors, VectorParams):
            return vectors.size
        return None  # named vectors: not made by this app; don't guess

    def content_hashes(self) -> dict[str, str]:
        """{point id: content_hash} for every book this app has stored."""
        hashes: dict[str, str] = {}
        offset: ExtendedPointId | None = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=_source_filter(),
                limit=_BATCH,
                offset=offset,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            for point in points:
                hashes[str(point.id)] = str((point.payload or {}).get("content_hash"))
            if offset is None:
                return hashes

    def upsert(self, docs: Sequence[Document], vectors: Sequence[list[float]]) -> int:
        if len(docs) != len(vectors):
            raise ValueError("docs and vectors must have equal lengths")
        now = datetime.now(tz=UTC).isoformat()
        points = [
            PointStruct(
                id=doc.point_id,
                vector=list(vector),
                payload={
                    **doc.payload,
                    "text": doc.text,
                    "content_hash": doc.content_hash,
                    "ingested_at": now,
                },
            )
            for doc, vector in zip(docs, vectors, strict=True)
        ]
        for start in range(0, len(points), _BATCH):
            self.client.upsert(
                collection_name=self.collection, points=points[start : start + _BATCH]
            )
        return len(points)

    def delete(self, point_ids: Sequence[str]) -> int:
        ids: list[ExtendedPointId] = sorted(point_ids)
        for start in range(0, len(ids), _BATCH):
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=ids[start : start + _BATCH]),
            )
        return len(ids)

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[tuple[float, Payload]]:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=_source_filter(),
            limit=limit,
            with_payload=True,
            score_threshold=min_score,
        ).points
        return [(hit.score, hit.payload or {}) for hit in hits]
