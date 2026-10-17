from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_COMMON_STOCK_EXCHANGES = ("NYSE", "Nasdaq", "NYSE American")


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
    metadata = pd.read_parquet(path)
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
            data = pd.read_parquet(path, columns=["date", "close", "major_exchange_listed"])
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
