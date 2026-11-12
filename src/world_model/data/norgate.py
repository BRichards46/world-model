from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_COMMON_STOCK_EXCHANGES = ("NYSE", "Nasdaq", "NYSE American")
CATALOG_NAME = "price_file_catalog.parquet"


def read_parquet_settled(path: str | Path, columns: list[str] | None = None, attempts: int = 6, wait_seconds: float = 10.0):
    """Read a parquet file that the Norgate update may be rewriting in place.

    The Norgate job (Windows, via a Parallels shared folder) rewrites the catalog by truncating and copying
    into the same file, so a read mid-write sees a file with no parquet footer. Retry for about a minute
    before giving up.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return pd.read_parquet(path, columns=columns)
        except (pa.ArrowInvalid, OSError) as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(wait_seconds)
    raise RuntimeError(f"{path} is unreadable after {attempts} attempts; is the Norgate update still writing it?") from last_error


def wait_for_settled_catalog(
    metadata_root: str | Path,
    quiet_seconds: float = 180.0,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 30.0,
    log=print,
) -> bool:
    """Wait until the Norgate price catalog has not changed for ``quiet_seconds`` and parses.

    Returns False on timeout (the caller decides whether to proceed with the last good data).
    """
    path = Path(metadata_root).expanduser() / CATALOG_NAME
    deadline = time.monotonic() + timeout_seconds
    announced = False
    while True:
        try:
            age = time.time() - path.stat().st_mtime
            if age >= quiet_seconds:
                pq.read_metadata(path)
                return True
            reason = f"changed {age:.0f}s ago"
        except (FileNotFoundError, pa.ArrowInvalid, OSError) as error:
            reason = f"not readable ({type(error).__name__})"
        if time.monotonic() >= deadline:
            if log:
                log(f"Norgate catalog still unsettled after {timeout_seconds / 60:.0f} min: {reason}.")
            return False
        if log and not announced:
            log(f"Waiting for the Norgate update to finish writing its catalog ({reason})...")
            announced = True
        time.sleep(poll_seconds)


def read_catalog_price_series(
    price_root: str | Path,
    metadata_root: str | Path,
    symbol: str,
    adjustment: str = "TOTALRETURN",
    column: str = "close",
) -> pd.Series:
    """Read the authoritative Norgate price file selected by the price catalog."""
    prices = Path(price_root).expanduser()
    metadata = Path(metadata_root).expanduser()
    catalog_path = metadata / "price_file_catalog.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    catalog = read_parquet_settled(catalog_path)
    rows = catalog[
        catalog["symbol"].astype(str).eq(symbol)
        & catalog["dataset"].astype(str).eq("equities")
        & catalog["price_adjustment"].astype(str).str.upper().eq(adjustment.upper())
    ]
    if rows.empty:
        raise ValueError(f"No canonical Norgate {adjustment} file found for {symbol}.")
    row = rows.sort_values(["last_date", "row_count"]).iloc[-1]
    path = metadata.parent / str(row["relative_path"])
    if path.parent.resolve() != prices.resolve():
        raise ValueError(f"Norgate catalog path for {symbol} is outside the configured price root: {path}")
    data = read_parquet_settled(path, columns=["date", column])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().sort_values("date").drop_duplicates("date", keep="last")
    series = data.set_index("date")[column].astype(float)
    series.index = series.index.normalize()
    return series.rename(symbol)


def read_catalog_price_frame(
    price_root: str | Path,
    metadata_root: str | Path,
    symbol: str,
    adjustment: str = "TOTALRETURN",
    columns: Iterable[str] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """Read canonical Norgate OHLC data for path-dependent forward outcomes."""
    prices = Path(price_root).expanduser()
    metadata = Path(metadata_root).expanduser()
    catalog_path = metadata / "price_file_catalog.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    catalog = read_parquet_settled(catalog_path)
    rows = catalog[
        catalog["symbol"].astype(str).eq(symbol)
        & catalog["dataset"].astype(str).eq("equities")
        & catalog["price_adjustment"].astype(str).str.upper().eq(adjustment.upper())
    ]
    if rows.empty:
        raise ValueError(f"No canonical Norgate {adjustment} file found for {symbol}.")
    row = rows.sort_values(["last_date", "row_count"]).iloc[-1]
    path = metadata.parent / str(row["relative_path"])
    if path.parent.resolve() != prices.resolve():
        raise ValueError(f"Norgate catalog path for {symbol} is outside the configured price root: {path}")
    requested = ["date", *columns]
    data = read_parquet_settled(path, columns=requested)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    data = data.set_index("date")
    data.index = data.index.normalize()
    return data.astype(float)


def read_norgate_index_series(
    metadata_root: str | Path,
    symbol: str,
    adjustment: str = "TOTALRETURN",
    column: str = "close",
    dataset: str = "us_indices",
) -> pd.Series:
    """Read a Norgate index series (for example ``$SPX`` or ``$SPXTR``) via the price catalog."""
    metadata = Path(metadata_root).expanduser()
    catalog_path = metadata / "price_file_catalog.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(catalog_path)
    catalog = read_parquet_settled(catalog_path)
    rows = catalog[
        catalog["symbol"].astype(str).eq(symbol)
        & catalog["dataset"].astype(str).eq(dataset)
        & catalog["price_adjustment"].astype(str).str.upper().eq(adjustment.upper())
    ]
    if rows.empty:
        raise ValueError(f"No Norgate {dataset} {adjustment} file found for {symbol}.")
    row = rows.sort_values(["last_date", "row_count"]).iloc[-1]
    norgate_root = metadata.parent.resolve()
    path = (metadata.parent / str(row["relative_path"])).resolve()
    if norgate_root not in path.parents:
        raise ValueError(f"Norgate catalog path for {symbol} is outside the Norgate root: {path}")
    data = read_parquet_settled(path, columns=["date", column])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().sort_values("date").drop_duplicates("date", keep="last")
    series = data.set_index("date")[column].astype(float)
    series.index = series.index.normalize()
    return series.rename(symbol)


@dataclass(frozen=True)
class BreadthBuildStats:
    selected_assets: int
    processed_assets: int
    skipped_assets: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None


def load_norgate_metadata(metadata_root: str | Path) -> pd.DataFrame:
    path = Path(metadata_root).expanduser() / "symbol_metadata_exported.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    metadata = read_parquet_settled(path)
    metadata["assetid"] = pd.to_numeric(metadata["assetid"], errors="coerce").astype("Int64")
    return metadata


def select_common_stock_universe(
    metadata: pd.DataFrame,
    exchanges: Iterable[str] = DEFAULT_COMMON_STOCK_EXCHANGES,
    domicile: str = "United States Of America",
    currency: str = "USD",
) -> pd.DataFrame:
    exchange_set = set(exchanges)
    selected = metadata[
        metadata["base_type"].eq("Stock Market")
        & metadata["subtype1"].eq("Equity")
        & metadata["subtype2"].eq("Operating/Holding Company")
        & metadata["domicile"].eq(domicile)
        & metadata["currency"].eq(currency)
        & metadata["exchange_name"].isin(exchange_set)
    ].copy()
    selected = selected.dropna(subset=["assetid", "symbol"])
    selected["assetid"] = selected["assetid"].astype(int)
    return selected.sort_values(["assetid", "symbol"]).drop_duplicates(subset=["assetid"], keep="last")


def build_price_file_map(price_root: str | Path) -> dict[int, Path]:
    root = Path(price_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    mapping: dict[int, Path] = {}
    for path in root.glob("assetid=*_symbol=*.parquet"):
        stem = path.stem
        try:
            asset_text = stem.split("_", 1)[0].split("=", 1)[1]
            mapping[int(asset_text)] = path
        except Exception:
            continue
    return mapping


def build_market_breadth(
    price_root: str | Path,
    metadata_root: str | Path,
    start_date: str | pd.Timestamp | None = "1990-01-01",
    exchanges: Iterable[str] = DEFAULT_COMMON_STOCK_EXCHANGES,
    max_assets: int | None = None,
    batch_size: int = 250,
) -> tuple[pd.DataFrame, BreadthBuildStats]:
    metadata = load_norgate_metadata(metadata_root)
    universe = select_common_stock_universe(metadata, exchanges=exchanges)
    if max_assets is not None:
        universe = universe.head(max_assets)

    file_map = build_price_file_map(price_root)
    start_ts = pd.Timestamp(start_date) if start_date is not None else None

    accumulator = pd.DataFrame()
    batch: list[pd.DataFrame] = []
    processed = 0
    skipped = 0
    for row in universe.itertuples(index=False):
        path = file_map.get(int(row.assetid))
        if path is None:
            skipped += 1
            continue
        try:
            data = read_parquet_settled(path, columns=["date", "close", "major_exchange_listed"], attempts=1)
        except Exception:
            skipped += 1
            continue
        if data.empty:
            skipped += 1
            continue

        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data["major_exchange_listed"] = pd.to_numeric(data["major_exchange_listed"], errors="coerce").fillna(0).astype(bool)
        data = data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if data.empty:
            skipped += 1
            continue

        close = data.set_index("date")["close"].astype(float)
        eligible = data.set_index("date")["major_exchange_listed"] & close.notna()

        features = pd.DataFrame(index=close.index)
        features["universe_count"] = eligible
        features["above_50d_count"] = eligible & (close > close.rolling(50, min_periods=40).mean())
        features["above_100d_count"] = eligible & (close > close.rolling(100, min_periods=80).mean())
        features["above_200d_count"] = eligible & (close > close.rolling(200, min_periods=160).mean())
        features["new_52w_high_count"] = eligible & (close >= close.rolling(252, min_periods=200).max())
        features["new_52w_low_count"] = eligible & (close <= close.rolling(252, min_periods=200).min())
        daily_return = close.pct_change(fill_method=None)
        features["advancing_count"] = eligible & (daily_return > 0)
        features["declining_count"] = eligible & (daily_return < 0)

        if start_ts is not None:
            features = features.loc[features.index >= start_ts]
        if features.empty:
            skipped += 1
            continue

        batch.append(features.astype(float))
        if len(batch) >= batch_size:
            accumulator = _flush_breadth_batch(accumulator, batch)
            batch.clear()
        processed += 1

    if batch:
        accumulator = _flush_breadth_batch(accumulator, batch)

    breadth = _counts_to_breadth(accumulator)
    stats = BreadthBuildStats(
        selected_assets=int(len(universe)),
        processed_assets=processed,
        skipped_assets=skipped,
        first_date=breadth.index.min() if not breadth.empty else None,
        last_date=breadth.index.max() if not breadth.empty else None,
    )
    return breadth, stats


def write_market_breadth(breadth: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    breadth.to_parquet(path)


def _flush_breadth_batch(accumulator: pd.DataFrame, batch: list[pd.DataFrame]) -> pd.DataFrame:
    if not batch:
        return accumulator
    batch_sum = pd.concat(batch).groupby(level=0).sum()
    batch_sum.index = pd.to_datetime(batch_sum.index).normalize()
    batch_sum = batch_sum.groupby(level=0).sum()
    if accumulator.empty:
        return batch_sum.sort_index()
    return accumulator.add(batch_sum, fill_value=0.0).sort_index()


def _counts_to_breadth(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    frame = frame.copy().sort_index()
    frame.index.name = "date"
    denom = frame["universe_count"].replace(0, np.nan)
    frame["pct_above_50d"] = frame["above_50d_count"] / denom
    frame["pct_above_100d"] = frame["above_100d_count"] / denom
    frame["pct_above_200d"] = frame["above_200d_count"] / denom
    frame["pct_new_52w_high"] = frame["new_52w_high_count"] / denom
    frame["pct_new_52w_low"] = frame["new_52w_low_count"] / denom
    frame["net_new_52w_high_low"] = frame["pct_new_52w_high"] - frame["pct_new_52w_low"]
    frame["pct_advancing"] = frame["advancing_count"] / denom
    frame["pct_declining"] = frame["declining_count"] / denom
    frame["pct_advancing_10d"] = frame["pct_advancing"].rolling(10, min_periods=5).mean()
    return frame
