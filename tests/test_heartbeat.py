from pathlib import Path

from library_rag_ingester import heartbeat


def test_missing_file_is_stale(tmp_path: Path) -> None:
    assert not heartbeat.is_fresh(tmp_path / "nope")


def test_fresh_then_stale(tmp_path: Path) -> None:
    path = tmp_path / "beat"
    heartbeat.touch(path)
    mtime = path.stat().st_mtime
    assert heartbeat.is_fresh(path)
    assert heartbeat.is_fresh(path, max_age=10, now=mtime + 10)
    assert not heartbeat.is_fresh(path, max_age=10, now=mtime + 11)
