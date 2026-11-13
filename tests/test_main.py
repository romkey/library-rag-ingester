import json
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from library_rag_ingester import __main__ as cli
from library_rag_ingester import __version__, heartbeat
from library_rag_ingester.catalog import CatalogClient, CatalogError
from library_rag_ingester.config import Settings
from library_rag_ingester.documents import build_document
from library_rag_ingester.qdrant_store import QdrantStore
from library_rag_ingester.service import IngestionService
from tests.fakes import DIMENSION, FakeEmbedder, book, vector

ENV = {
    "LIBRARY_CATALOG_URL": "https://library.example.com/api/v1/catalog",
    "LIBRARY_API_KEY": "lib_test",  # not-a-secret
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def signals(monkeypatch: pytest.MonkeyPatch) -> dict[int, Callable[..., Any]]:
    handlers: dict[int, Callable[..., Any]] = {}
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)
    return handlers


def catalog_client(*books_json: dict[str, object]) -> CatalogClient:
    body = "".join(json.dumps(b) + "\n" for b in books_json)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    return CatalogClient(ENV["LIBRARY_CATALOG_URL"], "k", transport=transport)


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"library-rag-ingester {__version__}\n"


def test_bad_config_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LIBRARY_CATALOG_URL", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "LIBRARY_CATALOG_URL is required" in capsys.readouterr().err


@pytest.mark.parametrize("fresh", [True, False])
def test_healthcheck(monkeypatch: pytest.MonkeyPatch, fresh: bool) -> None:
    monkeypatch.setattr(heartbeat, "is_fresh", lambda path: fresh)
    assert cli.main(["healthcheck"]) == (0 if fresh else 1)


def test_build_embedder() -> None:
    settings = Settings.from_env({**ENV, "OPENAI_API_KEY": "sk"})  # not-a-secret
    embedder = cli.build_embedder(settings)
    assert embedder.model == "nomic-ai/modernbert-embed-base"


def test_run_sync_once(
    settings: Settings,
    store: QdrantStore,
    signals: dict[int, Callable[..., Any]],
    tmp_path: Path,
) -> None:
    code = cli.run_sync(
        settings,
        once=True,
        embedder=FakeEmbedder(),
        vector_size=DIMENSION,
        store=store,
        catalog=catalog_client({"id": 1, "title": "One"}),
        heartbeat_path=tmp_path / "beat",
    )
    assert code == 0
    assert len(store.content_hashes()) == 1
    assert signal.SIGTERM in signals
    assert not (tmp_path / "beat").exists()  # only the long-running mode beats


def test_run_sync_forever_until_sigterm(
    settings: Settings,
    store: QdrantStore,
    signals: dict[int, Callable[..., Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_forever(self: IngestionService) -> None:
        signals[signal.SIGTERM](signal.SIGTERM, None)
        assert self._stop.is_set()

    monkeypatch.setattr(IngestionService, "run_forever", run_forever)
    assert (
        cli.run_sync(
            settings,
            once=False,
            embedder=FakeEmbedder(),
            vector_size=DIMENSION,
            store=store,
            catalog=catalog_client(),
        )
        == 0
    )


def test_run_sync_builds_real_clients(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, store: QdrantStore
) -> None:
    class StubEmbedder(FakeEmbedder):
        def __init__(self, settings: Settings) -> None:
            super().__init__()

    built: list[int] = []

    def connect_store(settings: Settings, vector_size: int) -> QdrantStore:
        built.append(vector_size)
        return store

    def fetch(self: CatalogClient, etag: str | None = None) -> None:
        raise CatalogError("down")

    monkeypatch.setattr(cli, "build_embedder", StubEmbedder)
    monkeypatch.setattr(cli, "connect_store", connect_store)
    monkeypatch.setattr(CatalogClient, "fetch", fetch)
    with pytest.raises(CatalogError):
        cli.run_sync(settings, once=True)
    assert built == [DIMENSION]


def test_main_sync_reports_failures(env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(settings: Settings, *, once: bool) -> int:
        raise CatalogError("library down")

    monkeypatch.setattr(cli, "run_sync", fail)
    assert cli.main(["--once"]) == 1


def test_main_interrupted(env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(settings: Settings, *, once: bool) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_sync", interrupt)
    assert cli.main([]) == 130


@pytest.mark.usefixtures("no_qdrant_server")
def test_main_query(
    env: None,
    monkeypatch: pytest.MonkeyPatch,
    store: QdrantStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = build_document(book("5", title="Make: Electronics"), "sig")
    store.upsert([doc], [vector(doc.text)])
    monkeypatch.setattr(cli, "QdrantStore", lambda **kwargs: store)
    monkeypatch.setattr(cli, "build_embedder", lambda settings: FakeEmbedder())

    assert cli.main(["query", "--limit", "1", doc.text]) == 0
    assert "Make: Electronics" in capsys.readouterr().out


@pytest.fixture
def no_qdrant_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """QdrantStore(url=...) gets an in-memory client instead of a network one."""
    monkeypatch.setattr(
        "library_rag_ingester.qdrant_store.QdrantClient",
        lambda **kwargs: QdrantClient(":memory:"),
    )


@pytest.mark.usefixtures("no_qdrant_server")
def test_connect_store_retries_until_qdrant_is_up(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def ensure(self: QdrantStore, *, vector_size: int, distance: str) -> None:
        attempts.append(vector_size)
        if len(attempts) < 3:
            raise ResponseHandlingException(ConnectionError("refused"))

    monkeypatch.setattr(QdrantStore, "ensure_collection", ensure)
    sleeps: list[float] = []
    store = cli.connect_store(settings, 8, sleep=sleeps.append)
    assert isinstance(store, QdrantStore)
    assert sleeps == [5, 5]


@pytest.mark.usefixtures("no_qdrant_server")
def test_connect_store_gives_up(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ensure(self: QdrantStore, *, vector_size: int, distance: str) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(QdrantStore, "ensure_collection", ensure)
    with pytest.raises(httpx.ConnectError):
        cli.connect_store(settings, 8, attempts=2, sleep=lambda s: None)
