import os
import time

import pandas as pd
import pytest

from world_model.data import norgate
from world_model.data.norgate import read_parquet_settled, wait_for_settled_catalog


def test_read_retries_a_file_that_is_mid_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "price_file_catalog.parquet"
    good = tmp_path / "good.parquet"
    pd.DataFrame({"symbol": ["$SPX"]}).to_parquet(good)
    path.write_bytes(good.read_bytes()[:20])  # truncated: no parquet footer
    calls = []

    def finish_write(seconds):
        calls.append(seconds)
        path.write_bytes(good.read_bytes())

    monkeypatch.setattr(norgate.time, "sleep", finish_write)
    frame = read_parquet_settled(path, attempts=3, wait_seconds=5)
    assert list(frame["symbol"]) == ["$SPX"]
    assert calls == [5]


def test_read_gives_a_clear_error_when_the_file_never_settles(tmp_path, monkeypatch):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not parquet")
    monkeypatch.setattr(norgate.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="unreadable after 2 attempts"):
        read_parquet_settled(path, attempts=2)


def test_wait_returns_once_the_catalog_is_quiet_and_readable(tmp_path):
    path = tmp_path / "price_file_catalog.parquet"
    pd.DataFrame({"symbol": ["$SPX"]}).to_parquet(path)
    old = time.time() - 600
    os.utime(path, (old, old))
    assert wait_for_settled_catalog(tmp_path, quiet_seconds=180, timeout_seconds=1, log=None)


def test_wait_times_out_while_the_catalog_keeps_changing(tmp_path, monkeypatch):
    path = tmp_path / "price_file_catalog.parquet"
    pd.DataFrame({"symbol": ["$SPX"]}).to_parquet(path)
    monkeypatch.setattr(norgate.time, "sleep", lambda seconds: None)
    assert not wait_for_settled_catalog(tmp_path, quiet_seconds=180, timeout_seconds=0, log=None)
