"""Cleveland Fed inflation nowcasts: the daily month-over-month nowcast for each target month.

The Cleveland Fed's chart data (one JSON file) holds, for every target month since July 2013, the nowcast as
shown on each business day plus the actual once released. The nowcast for a release is the last value shown
before the release date.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

MONTHLY_URL = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json?sc_lang=en"
SERIES = {
    "CPI Inflation": "headline_cpi_mom",
    "Core CPI Inflation": "core_cpi_mom",
    "PCE Inflation": "headline_pce_mom",
    "Core PCE Inflation": "core_pce_mom",
}
ACTUALS = {
    "Actual CPI Inflation": "headline_cpi_mom",
    "Actual Core CPI Inflation": "core_cpi_mom",
    "Actual PCE Inflation": "headline_pce_mom",
    "Actual Core PCE Inflation": "core_pce_mom",
}


def download(path: str | Path, url: str = MONTHLY_URL) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (world-model research)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def parse_daily(charts: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per target month, series and business day: the nowcast (or actual) shown that day."""
    rows = []
    for chart in charts:
        year, month = (int(part) for part in str(chart["chart"]["subcaption"]).split("-"))
        target = pd.Timestamp(year=year, month=month, day=1)
        dates = [_date(c["label"], target) for c in chart["categories"][0]["category"] if not c.get("vline")]
        for dataset in chart["dataset"]:
            name = dataset["seriesname"]
            kind = "nowcast" if name in SERIES else "actual" if name in ACTUALS else None
            if kind is None:
                continue
            target_id = SERIES.get(name) or ACTUALS[name]
            for date, point in zip(dates, dataset["data"]):
                value = point.get("value")
                if value in ("", None):
                    continue
                rows.append(
                    {"reference_month": target, "target_id": target_id, "kind": kind, "date": date, "value": float(value)}
                )
    return pd.DataFrame.from_records(rows, columns=["reference_month", "target_id", "kind", "date", "value"])


def release_nowcasts(daily: pd.DataFrame) -> pd.DataFrame:
    """Per target month and series: the last nowcast before the release, the release date and the actual."""
    rows = []
    for (month, target_id), group in daily.groupby(["reference_month", "target_id"]):
        actual = group[group["kind"] == "actual"].sort_values("date")
        nowcasts = group[group["kind"] == "nowcast"].sort_values("date")
        release = actual["date"].iloc[0] if len(actual) else None
        before = nowcasts[nowcasts["date"] < release] if release is not None else nowcasts
        if before.empty:
            continue
        rows.append(
            {
                "reference_month": month,
                "target_id": target_id,
                "nowcast": float(before["value"].iloc[-1]),
                "nowcast_date": before["date"].iloc[-1],
                "release_date": release,
                "cleveland_actual": float(actual["value"].iloc[0]) if len(actual) else None,
            }
        )
    return pd.DataFrame.from_records(rows)


def load_release_nowcasts(path: str | Path) -> pd.DataFrame:
    return release_nowcasts(parse_daily(json.loads(Path(path).expanduser().read_text(encoding="utf-8"))))


def _date(label: str, target: pd.Timestamp) -> pd.Timestamp:
    month, day = (int(part) for part in label.split("/"))
    year = target.year if month >= target.month else target.year + 1
    return pd.Timestamp(year=year, month=month, day=day)
