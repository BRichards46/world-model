"""Kalshi economic-release markets: the price of each "above X" contract shortly before the release.

Kalshi lists, for each release, a ladder of contracts such as "CPI rises more than 0.3%". Their prices just
before trading closes (five minutes before the 8:30 ET release) are the market's probabilities for the print.
Market data is public; no key is needed.

Markets settled before Kalshi's archive cutoff are served from the ``/historical`` endpoints; newer ones from the
live endpoints. Both are read here and cached so later runs only fetch new releases.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
USER_AGENT = "world-model-research/1.0"
LADDER_COLUMNS = [
    "series",
    "event_ticker",
    "ticker",
    "reference_month",
    "strike",
    "close_time",
    "result",
    "volume",
    "price_at_close",
    "price_day_before",
    "source",
]
_MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _get(path: str, params: dict[str, Any] | None = None, retries: int = 4) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {}
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return {}


def _paged(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = _get(path, {**params, "limit": 1000, "cursor": cursor})
        markets.extend(page.get("markets") or [])
        cursor = page.get("cursor")
        if not cursor:
            return markets


def list_settled_markets(series: str) -> list[dict[str, Any]]:
    """Every settled market in a series, archived and recent, tagged with where it came from."""
    archived = [{**m, "_source": "historical"} for m in _paged("/historical/markets", {"series_ticker": series})]
    recent = [{**m, "_source": "live"} for m in _paged("/markets", {"series_ticker": series, "status": "settled"})]
    seen = {m["ticker"] for m in archived}
    return archived + [m for m in recent if m["ticker"] not in seen]


def list_open_markets(series: str) -> list[dict[str, Any]]:
    return [{**m, "_source": "live"} for m in _paged("/markets", {"series_ticker": series, "status": "open"})]


def strike_of(market: dict[str, Any]) -> float | None:
    """The "above" level. Older markets carry it only in the ticker (``T0.3``, ``TN0.1`` for -0.1)."""
    strike_type = market.get("strike_type")
    if strike_type not in (None, "greater"):
        return None
    if market.get("floor_strike") is not None:
        return float(market["floor_strike"])
    match = re.search(r"-T(N?)(-?\d+(?:\.\d+)?)$", str(market.get("ticker", "")))
    if not match:
        return None
    value = float(match.group(2))
    return -value if match.group(1) else value


def reference_month(event_ticker: str) -> pd.Timestamp | None:
    match = re.search(r"-(\d{2})([A-Z]{3})(?:\d{2})?$", event_ticker)
    if not match or match.group(2) not in _MONTHS:
        return None
    return pd.Timestamp(year=2000 + int(match.group(1)), month=_MONTHS[match.group(2)], day=1)


def candle_price(candle: dict[str, Any], max_spread: float = 0.15) -> float | None:
    """Mid of the closing bid and ask when the spread is tight, else the last trade."""
    bid = _field(candle.get("yes_bid"), "close")
    ask = _field(candle.get("yes_ask"), "close")
    if bid is not None and ask is not None and 0.0 <= ask - bid <= max_spread:
        return (bid + ask) / 2.0
    price = candle.get("price") or {}
    last = _field(price, "close")
    if last is None:
        last = _field(price, "previous")
    if last is not None:
        return last
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return None


def pre_release_prices(market: dict[str, Any], series: str) -> tuple[float | None, float | None]:
    """Price at the last hourly candle before trading closed, and about a day earlier."""
    close = pd.Timestamp(market["close_time"])
    close_ts = int(close.timestamp())
    params = {"start_ts": close_ts - 3 * 86400, "end_ts": close_ts + 3600, "period_interval": 60}
    if market.get("_source") == "historical":
        path = f"/historical/markets/{market['ticker']}/candlesticks"
    else:
        path = f"/series/{series}/markets/{market['ticker']}/candlesticks"
    candles = _get(path, params).get("candlesticks") or []
    at_close = day_before = None
    for candle in sorted(candles, key=lambda c: c.get("end_period_ts", 0)):
        end = int(candle.get("end_period_ts", 0))
        price = candle_price(candle)
        if price is None:
            continue
        if end <= close_ts + 3600:
            at_close = price
        if end <= close_ts - 86400:
            day_before = price
    return at_close, day_before


def update_ladders(series: str, cache_dir: str | Path, pause: float = 0.05, log: Any = print) -> pd.DataFrame:
    """Fetch pre-release prices for settled markets not yet cached; return the full ladder table."""
    path = Path(cache_dir).expanduser() / f"{series}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=LADDER_COLUMNS)
    if len(cached):
        # Recent markets with no price yet are fetched again; archived gaps are genuine and stay.
        retry = cached["price_at_close"].isna() & (cached["source"] == "live")
        cached = cached[~retry]
    have = set(cached["ticker"]) if len(cached) else set()
    rows = []
    markets = [m for m in list_settled_markets(series) if m["ticker"] not in have]
    for i, market in enumerate(markets, 1):
        strike = strike_of(market)
        month = reference_month(str(market.get("event_ticker", "")))
        if strike is None or month is None:
            continue
        at_close, day_before = pre_release_prices(market, series)
        rows.append(
            {
                "series": series,
                "event_ticker": market["event_ticker"],
                "ticker": market["ticker"],
                "reference_month": month,
                "strike": strike,
                "close_time": pd.Timestamp(market["close_time"]),
                "result": market.get("result"),
                "volume": _float(market.get("volume_fp")) or _float(market.get("volume")) or 0.0,
                "price_at_close": at_close,
                "price_day_before": day_before,
                "source": market.get("_source"),
            }
        )
        if log and i % 100 == 0:
            log(f"  {series}: {i}/{len(markets)} markets")
        time.sleep(pause)
    table = pd.concat([cached, pd.DataFrame.from_records(rows, columns=LADDER_COLUMNS)], ignore_index=True)
    table = table.drop_duplicates("ticker", keep="last").sort_values(["reference_month", "event_ticker", "strike"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def open_ladders(series_list: Iterable[str]) -> pd.DataFrame:
    """Current prices for releases still trading (last price, or mid of bid and ask)."""
    rows = []
    for series in series_list:
        for market in list_open_markets(series):
            strike = strike_of(market)
            month = reference_month(str(market.get("event_ticker", "")))
            if strike is None or month is None:
                continue
            bid, ask = _float(market.get("yes_bid_dollars")), _float(market.get("yes_ask_dollars"))
            price = (bid + ask) / 2.0 if bid is not None and ask is not None and ask >= bid else _float(market.get("last_price_dollars"))
            rows.append(
                {
                    "series": series,
                    "event_ticker": market["event_ticker"],
                    "ticker": market["ticker"],
                    "reference_month": month,
                    "strike": strike,
                    "close_time": pd.Timestamp(market["close_time"]),
                    "volume": _float(market.get("volume_fp")) or 0.0,
                    "price_at_close": price,
                }
            )
    return pd.DataFrame.from_records(rows)


def _field(block: Any, name: str) -> float | None:
    """Archived candles name prices ``close``; live candles name them ``close_dollars``."""
    if not isinstance(block, dict):
        return None
    value = _float(block.get(name))
    return value if value is not None else _float(block.get(f"{name}_dollars"))


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
