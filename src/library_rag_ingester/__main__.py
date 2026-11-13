"""Command line: `library-rag-ingester` or `python -m library_rag_ingester`.

With no command it syncs the library catalog into Qdrant every
POLL_INTERVAL_SECONDS (or once, with --once). `query` searches the
collection; `healthcheck` is for the production container's health check.
"""

import argparse
import logging
import signal
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException

from library_rag_ingester import __version__, heartbeat
from library_rag_ingester.catalog import CatalogClient
from library_rag_ingester.config import ConfigError, Settings
from library_rag_ingester.embeddings import Embedder, OpenAIEmbedder
from library_rag_ingester.qdrant_store import QdrantStore
from library_rag_ingester.query import format_hits, search
from library_rag_ingester.service import CYCLE_ERRORS, IngestionService

logger = logging.getLogger("library_rag_ingester")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="library-rag-ingester", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="sync once and exit (like POLL_INTERVAL_SECONDS=0)",
    )
    commands = parser.add_subparsers(dest="command")

    query = commands.add_parser("query", help="search the collection")
    query.add_argument("text", help="natural-language query")
    query.add_argument("--limit", type=int, default=5)
    query.add_argument("--min-score", type=float, default=None)
    query.add_argument("--show-text", action="store_true", help="print each text")
    query.add_argument("--width", type=int, default=100)

    commands.add_parser("healthcheck", help="exit 0 if the running sync is alive")
    return parser


def build_embedder(settings: Settings) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        model=settings.embedding_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout_seconds=max(settings.request_timeout_seconds, 120),
        batch_size=settings.embedding_batch_size,
        max_tokens=settings.embedding_context_length,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )


def connect_store(
    settings: Settings,
    vector_size: int,
    *,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> QdrantStore:
    """Open the collection, waiting for Qdrant to come up (it may start after us)."""
    for attempt in range(1, attempts + 1):
        try:
            store = QdrantStore(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection=settings.qdrant_collection,
            )
            store.ensure_collection(
                vector_size=vector_size, distance=settings.qdrant_distance
            )
        except (ResponseHandlingException, httpx.TransportError) as exc:
            if attempt == attempts:
                raise
            logger.warning(
                "Qdrant not ready (attempt %d/%d): %s; retrying in 5s",
                attempt,
                attempts,
                exc,
            )
            sleep(5)
        else:
            return store
    raise AssertionError("unreachable")


def run_sync(
    settings: Settings,
    *,
    once: bool,
    embedder: Embedder | None = None,
    vector_size: int | None = None,
    store: QdrantStore | None = None,
    catalog: CatalogClient | None = None,
    heartbeat_path: Path = heartbeat.DEFAULT_PATH,
) -> int:
    if embedder is None or vector_size is None:
        openai = build_embedder(settings)
        embedder, vector_size = openai, openai.dimension
    logger.info("Embedding dimension: %d", vector_size)
    store = store or connect_store(settings, vector_size)
    catalog = catalog or CatalogClient(
        settings.library_catalog_url,
        settings.library_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    with catalog:
        service = IngestionService(
            settings=settings,
            catalog=catalog,
            store=store,
            embedder=embedder,
            heartbeat_path=None if once else heartbeat_path,
        )

        # `docker stop` sends SIGTERM: finish the current batch, then exit.
        def stop(signum: int, frame: FrameType | None) -> None:
            logger.info("received signal %d, stopping", signum)
            service.stop()

        signal.signal(signal.SIGTERM, stop)

        if once or settings.poll_interval_seconds == 0:
            service.run_cycle()
            return 0
        logger.info("Syncing every %ss", settings.poll_interval_seconds)
        service.run_forever()
    return 0


def run_query(settings: Settings, args: argparse.Namespace) -> int:
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
    )
    hits = search(
        store,
        build_embedder(settings),
        args.text,
        limit=args.limit,
        min_score=args.min_score,
    )
    format_hits(hits, sys.stdout, show_text=args.show_text, width=args.width)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "healthcheck":
        return 0 if heartbeat.is_fresh(heartbeat.DEFAULT_PATH) else 1

    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"library-rag-ingester: {e}\n")

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # httpx logs every request at INFO, URLs included.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info(
        "library-rag-ingester %s (model=%s, collection=%s)",
        __version__,
        settings.embedding_model,
        settings.qdrant_collection,
    )

    try:
        if args.command == "query":
            return run_query(settings, args)
        return run_sync(settings, once=args.once)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    # ValueError: the collection's vector size doesn't match the model.
    except (*CYCLE_ERRORS, ValueError) as exc:
        logger.error("Stopping: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
