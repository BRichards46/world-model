from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_source_table(root: str | Path, table: str, symbols: Iterable[str]) -> pd.DataFrame:
    """Read selected symbols from a partitioned Tanuki normalized source table."""
    table_root = Path(root).expanduser() / "normalized" / "source_tables" / table
    if not table_root.exists():
        raise FileNotFoundError(table_root)
    symbol_set = set(symbols)
    frames: list[pd.DataFrame] = []
    for path in sorted(table_root.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        if "symbol" not in frame:
            continue
        selected = frame[frame["symbol"].astype(str).isin(symbol_set)]
        if not selected.empty:
            frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_by_symbol(frame: pd.DataFrame, date_column: str = "snapshot_date") -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce", utc=True)
    return data.dropna(subset=[date_column]).sort_values(date_column).groupby("symbol", as_index=False).tail(1)


def select_delta_expiry(
    frame: pd.DataFrame,
    target_dte: int = 30,
    minimum_dte: int = 20,
    maximum_dte: int = 45,
) -> pd.DataFrame:
    """Select each symbol's latest snapshot and expiry nearest the target DTE."""
    if frame.empty:
        return frame
    data = frame.copy()
    data["snapshot_at"] = pd.to_datetime(data["snapshot_at"], errors="coerce", utc=True)
    data["dte"] = pd.to_numeric(data["dte"], errors="coerce")
    latest = data.groupby("symbol")["snapshot_at"].transform("max")
    data = data[data["snapshot_at"].eq(latest) & data["dte"].between(minimum_dte, maximum_dte)].copy()
    data["target_distance"] = (data["dte"] - target_dte).abs()
    data["monthly_rank"] = ~data["is_monthly"].fillna(False).astype(bool)
    return data.sort_values(["symbol", "target_distance", "monthly_rank", "dte"]).groupby("symbol", as_index=False).head(1)
