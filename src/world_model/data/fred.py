from __future__ import annotations

from pathlib import Path

import pandas as pd


VINTAGE_COLUMNS = ["series_id", "observation_date", "value", "realtime_start", "realtime_end"]


def vintage_path(fred_root: str | Path, series_id: str) -> Path:
    return Path(fred_root).expanduser() / "vintage" / f"{series_id}.parquet"


def latest_path(fred_root: str | Path, series_id: str) -> Path:
    return Path(fred_root).expanduser() / "latest" / f"{series_id}.parquet"


def has_vintage_series(fred_root: str | Path, series_id: str) -> bool:
    return vintage_path(fred_root, series_id).exists()


def load_vintage_table(fred_root: str | Path, series_id: str) -> pd.DataFrame:
    path = vintage_path(fred_root, series_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing FRED vintage file for {series_id}: {path}")

    table = pd.read_parquet(path)
    missing_columns = set(VINTAGE_COLUMNS).difference(table.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    table = table[VINTAGE_COLUMNS].copy()
    table["observation_date"] = pd.to_datetime(table["observation_date"], errors="coerce").dt.normalize()
    table["realtime_start"] = pd.to_datetime(table["realtime_start"], errors="coerce").dt.normalize()
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.dropna(subset=["observation_date", "realtime_start"])
    return table.sort_values(["realtime_start", "observation_date"]).reset_index(drop=True)


def available_vintage_series(fred_root: str | Path) -> set[str]:
    root = Path(fred_root).expanduser() / "vintage"
    if not root.exists():
        return set()
    return {path.stem for path in root.glob("*.parquet")}
