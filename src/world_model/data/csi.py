from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date").sort_index()
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    raise ValueError("CSI parquet must have a Date/date column or a DatetimeIndex.")


def read_price_series(root: str | Path, filename: str, column: str = "Close") -> pd.Series:
    path = Path(root).expanduser() / filename
    if not path.exists():
        raise FileNotFoundError(path)
    df = _normalize_datetime_index(pd.read_parquet(path))
    if column not in df.columns:
        raise ValueError(f"{path} does not contain column {column!r}. Available: {list(df.columns)}")
    series = pd.to_numeric(df[column], errors="coerce").sort_index()
    series.index = pd.to_datetime(series.index).normalize()
    return series


def load_price_panel(root: str | Path, instruments: dict[str, dict[str, Any]]) -> pd.DataFrame:
    series_by_asset: dict[str, pd.Series] = {}
    for asset, settings in instruments.items():
        series = read_price_series(root, settings["file"])
        series_by_asset[asset] = series.rename(asset)
    if not series_by_asset:
        raise ValueError("No instruments configured.")
    panel = pd.concat(series_by_asset.values(), axis=1).sort_index()
    return panel[~panel.index.duplicated(keep="last")]


def read_cot_series(
    root: str | Path,
    filename: str,
    column: str = "lg_spec_pct_oi",
) -> pd.Series:
    path = Path(root).expanduser() / filename
    if not path.exists():
        raise FileNotFoundError(path)
    df = _normalize_datetime_index(pd.read_parquet(path))
    if column not in df.columns:
        raise ValueError(f"{path} does not contain column {column!r}. Available: {list(df.columns)}")
    series = pd.to_numeric(df[column], errors="coerce").sort_index()
    series.index = pd.to_datetime(series.index).normalize()
    return series


def load_cot_panel(root: str | Path, cot_config: dict[str, dict[str, Any]]) -> pd.DataFrame:
    series_by_asset: dict[str, pd.Series] = {}
    for asset, settings in cot_config.items():
        source_column = settings.get("source_column", "lg_spec_pct_oi")
        series = read_cot_series(root, settings["file"], source_column)
        series_by_asset[asset] = series.rename(asset)
    if not series_by_asset:
        return pd.DataFrame()
    panel = pd.concat(series_by_asset.values(), axis=1).sort_index()
    return panel[~panel.index.duplicated(keep="last")]
