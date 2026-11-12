"""Score our release forecasts against the market (Kalshi) and the Cleveland Fed nowcast.

All three are compared with the first print. Kalshi's expectation is read from the ladder of "above X"
contracts just before trading closes: their prices give the probability of each range, and the mean of
that distribution is the market's expected print. Kalshi settles on the rounded print, so "above 0.3%" means
the unrounded change was at least 0.35%; strikes are shifted by half a rounding step before taking the mean.

The key question is not only who has the smallest error but whether our forecast knows something the market
does not: when we differ from the market, does the print tend to land on our side?
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

KALSHI_SERIES = {
    "KXCPI": {"target_id": "headline_cpi_mom", "rounding": 0.1, "scale": 1.0},
    "KXCPICORE": {"target_id": "core_cpi_mom", "rounding": 0.1, "scale": 1.0},
    "KXPCECORE": {"target_id": "core_pce_mom", "rounding": 0.1, "scale": 1.0},
    "KXPAYROLLS": {"target_id": "payroll_change", "rounding": 1.0, "scale": 0.001},
}
TARGET_LABELS = {
    "headline_cpi_mom": "Headline CPI MoM (%)",
    "core_cpi_mom": "Core CPI MoM (%)",
    "core_pce_mom": "Core PCE MoM (%)",
    "payroll_change": "Payroll change (thousands)",
}
OUR_VARIANTS = {"headline_cpi_mom": "pipeline", "core_cpi_mom": "pipeline", "core_pce_mom": "pipeline", "payroll_change": "core"}
SOURCES = ("ours", "kalshi", "cleveland", "naive")


def ladder_distribution(strikes: Iterable[float], prices: Iterable[float], rounding: float) -> dict[str, float] | None:
    """Mean, median and spread implied by "above strike" prices.

    The probability of finishing above each strike is forced to fall as the strike rises (stale quotes can
    break that). Mass between two strikes sits at their midpoint; each tail sits half a strike gap outside.
    """
    frame = pd.DataFrame({"strike": list(strikes), "price": list(prices)}).dropna()
    frame = frame.groupby("strike", as_index=False)["price"].mean().sort_values("strike")
    if len(frame) < 2:
        return None
    thresholds = frame["strike"].to_numpy(float) + rounding / 2.0
    above = np.minimum.accumulate(np.clip(frame["price"].to_numpy(float), 0.0, 1.0))
    gaps = np.diff(thresholds)
    points = np.concatenate([[thresholds[0] - gaps[0] / 2.0], (thresholds[:-1] + thresholds[1:]) / 2.0, [thresholds[-1] + gaps[-1] / 2.0]])
    mass = np.concatenate([[1.0 - above[0]], above[:-1] - above[1:], [above[-1]]])
    if mass.sum() <= 0:
        return None
    mass = mass / mass.sum()
    mean = float((points * mass).sum())
    spread = float(np.sqrt(max((mass * (points - mean) ** 2).sum(), 0.0)))
    return {
        "mean": mean,
        "median": _quantile(thresholds, above, 0.5),
        "low": _quantile(thresholds, above, 0.1),
        "high": _quantile(thresholds, above, 0.9),
        # True when the 10th/90th percentile lies beyond the outermost strike, so the range end is a bound.
        "low_open": bool(above[0] < 0.9),
        "high_open": bool(above[-1] >= 0.1),
        "std": spread,
        "tail_mass": float(mass[0] + mass[-1]),
    }


def _quantile(thresholds: np.ndarray, above: np.ndarray, q: float) -> float:
    """Level with probability ``q`` below it, interpolated between strikes; capped at the outermost strikes."""
    level = 1.0 - q
    if above[0] < level:
        return float(thresholds[0])
    if above[-1] >= level:
        return float(thresholds[-1])
    i = int(np.argmax(above < level)) - 1
    drop = above[i] - above[i + 1]
    weight = (above[i] - level) / drop if drop > 0 else 0.5
    return float(thresholds[i] + weight * (thresholds[i + 1] - thresholds[i]))


def kalshi_expectations(ladders: pd.DataFrame, price_column: str = "price_at_close") -> pd.DataFrame:
    """One row per series and reference month: the market's implied expectation before the release.

    Where Kalshi listed the same month under an old and a new ticker prefix, the busier event is kept.
    """
    rows = []
    for (series, event), group in ladders.groupby(["series", "event_ticker"]):
        settings = KALSHI_SERIES.get(series)
        if settings is None:
            continue
        implied = ladder_distribution(group["strike"], group[price_column], settings["rounding"])
        if implied is None:
            continue
        scale = settings["scale"]
        rows.append(
            {
                "target_id": settings["target_id"],
                "reference_month": pd.Timestamp(group["reference_month"].iloc[0]),
                "event_ticker": event,
                # The median is the market estimate: it does not depend on where mass beyond the last
                # strike is placed, and it scored as well as or better than the mean on every series.
                "kalshi": implied["median"] * scale,
                "kalshi_mean": implied["mean"] * scale,
                "kalshi_low": implied["low"] * scale,
                "kalshi_high": implied["high"] * scale,
                "kalshi_range_open": [implied["low_open"], implied["high_open"]],
                "kalshi_std": implied["std"] * scale,
                "kalshi_tail_mass": implied["tail_mass"],
                "kalshi_volume": float(group["volume"].sum()),
                "kalshi_close": pd.Timestamp(group["close_time"].max()),
                "kalshi_strikes": int(len(group)),
            }
        )
    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table
    table = table.sort_values("kalshi_volume").drop_duplicates(["target_id", "reference_month"], keep="last")
    return table.sort_values(["target_id", "reference_month"]).reset_index(drop=True)


def our_forecasts(
    forecasts: pd.DataFrame,
    payroll_levels: pd.Series | None = None,
    headline_bottom_up: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Our out-of-sample forecasts in display units (payrolls converted from basis points to thousands).

    With ``headline_bottom_up`` (the energy CPI backtest), headline CPI is scored on the bottom-up nowcast,
    which is what the dashboard shows; the autoregressive benchmark stays the naive comparison.
    """
    rows = []
    for target_id, variant in OUR_VARIANTS.items():
        part = forecasts[(forecasts["target_id"] == target_id) & (forecasts["variant_id"] == variant)].copy()
        if part.empty:
            continue
        part["reference_month"] = pd.to_datetime(part["observation_date"]).dt.to_period("M").dt.to_timestamp()
        if target_id == "payroll_change":
            if payroll_levels is None:
                continue
            level = part["reference_month"].map(payroll_levels)
            for column in ("actual", "forecast", "baseline_forecast"):
                part[column] = part[column] / 10000.0 * level
        if target_id == "headline_cpi_mom" and headline_bottom_up is not None and not headline_bottom_up.empty:
            bottom_up = headline_bottom_up.set_index("month")
            part = part.set_index("reference_month", drop=False)
            part["forecast"] = bottom_up["headline_forecast"].reindex(part.index).to_numpy()
            part = part.dropna(subset=["forecast"]).reset_index(drop=True)
        rows.append(
            part.rename(columns={"forecast": "ours", "baseline_forecast": "naive"})[
                ["target_id", "reference_month", "release_date", "actual", "ours", "naive"]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prior_payroll_levels(vintage: pd.DataFrame, releases: pd.DataFrame) -> pd.Series:
    """For each payroll reference month: the prior month's employment level (thousands) as published on release day."""
    table = vintage.copy()
    table["observation_date"] = pd.to_datetime(table["observation_date"])
    table["realtime_start"] = pd.to_datetime(table["realtime_start"])
    table["realtime_end"] = pd.to_datetime(table["realtime_end"], errors="coerce").fillna(pd.Timestamp.max.normalize())
    levels = {}
    for row in releases.itertuples(index=False):
        released, prior = pd.Timestamp(row.release_date), pd.Timestamp(row.reference_month) - pd.DateOffset(months=1)
        known = table[
            (table["observation_date"] == prior) & (table["realtime_start"] <= released) & (table["realtime_end"] >= released)
        ]
        if len(known):
            levels[pd.Timestamp(row.reference_month)] = float(known.sort_values("realtime_start")["value"].iloc[-1])
    return pd.Series(levels, dtype=float)


def build_comparison(ours: pd.DataFrame, kalshi: pd.DataFrame, cleveland: pd.DataFrame) -> pd.DataFrame:
    """Every release where we have a forecast, joined with the market and the Cleveland Fed where available."""
    table = ours.copy()
    if not kalshi.empty:
        table = table.merge(kalshi, on=["target_id", "reference_month"], how="left")
    if not cleveland.empty:
        table = table.merge(
            cleveland[["target_id", "reference_month", "nowcast", "nowcast_date"]].rename(columns={"nowcast": "cleveland"}),
            on=["target_id", "reference_month"],
            how="left",
        )
    for column in ("kalshi", "cleveland"):
        if column not in table:
            table[column] = np.nan
    return table.sort_values(["target_id", "reference_month"]).reset_index(drop=True)


def accuracy(table: pd.DataFrame, sources: Iterable[str]) -> dict[str, Any]:
    """Errors for each source on the releases where every listed source has a value."""
    sources = list(sources)
    sample = table.dropna(subset=["actual", *sources])
    result: dict[str, Any] = {"releases": int(len(sample))}
    if sample.empty:
        return result
    result["start"] = f"{sample['reference_month'].min():%Y-%m}"
    result["end"] = f"{sample['reference_month'].max():%Y-%m}"
    for source in sources:
        error = sample["actual"] - sample[source]
        result[source] = {"mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean())), "bias": float(error.mean())}
    if len(sources) >= 2:
        first, second = sources[0], sources[1]
        closer = (sample["actual"] - sample[first]).abs() < (sample["actual"] - sample[second]).abs()
        result[f"{first}_closer_share"] = float(closer.mean())
    return result


def edge_over(table: pd.DataFrame, benchmark: str, source: str = "ours", min_gap: float = 0.0) -> dict[str, Any]:
    """Does our gap from the benchmark predict the surprise against that benchmark?

    Regresses (actual - benchmark) on (ours - benchmark). A slope near 1 means our differences are right;
    near 0 means they are noise. The hit rate counts how often the print landed on our side of the benchmark
    when we differed by more than ``min_gap``.
    """
    sample = table.dropna(subset=["actual", benchmark, source])
    if len(sample) < 8:
        return {"releases": int(len(sample))}
    surprise = (sample["actual"] - sample[benchmark]).to_numpy(float)
    signal = (sample[source] - sample[benchmark]).to_numpy(float)
    slope, intercept, t_stat = _ols(signal, surprise)
    differ = np.abs(signal) > min_gap
    agree = np.sign(signal[differ]) == np.sign(surprise[differ])
    return {
        "releases": int(len(sample)),
        "slope": slope,
        "intercept": intercept,
        "t_stat": t_stat,
        "correlation": float(np.corrcoef(signal, surprise)[0, 1]) if signal.std() > 0 and surprise.std() > 0 else None,
        "differed": int(differ.sum()),
        "same_side_rate": float(agree.mean()) if differ.any() else None,
        "min_gap": min_gap,
    }


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = max(len(y) - 2, 1)
    variance = (residual @ residual) / dof * np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(max(variance[1, 1], 0.0)))
    return float(coef[1]), float(coef[0]), float(coef[1] / se) if se > 0 else float("nan")


MIN_GAP = {"headline_cpi_mom": 0.05, "core_cpi_mom": 0.05, "core_pce_mom": 0.05, "payroll_change": 25.0}


PANDEMIC_MONTHS = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-01"))


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    """Per target: three-way accuracy on the common sample, pairwise accuracy, and edge tests.

    March to December 2020 is left out: payroll swings of millions would swamp every average.
    """
    summary = {}
    months = pd.to_datetime(table["reference_month"])
    table = table[~months.between(*PANDEMIC_MONTHS)]
    for target_id, group in table.groupby("target_id"):
        gap = MIN_GAP.get(target_id, 0.0)
        summary[target_id] = {
            "label": TARGET_LABELS.get(target_id, target_id),
            "all_sources": accuracy(group, [s for s in ("ours", "kalshi", "cleveland", "naive") if group[s].notna().any()]),
            "ours_vs_kalshi": accuracy(group, ["ours", "kalshi"]),
            "ours_vs_cleveland": accuracy(group, ["ours", "cleveland"]),
            "cleveland_vs_kalshi": accuracy(group, ["cleveland", "kalshi"]),
            "edge_ours_over_kalshi": edge_over(group, "kalshi", "ours", gap),
            "edge_cleveland_over_kalshi": edge_over(group, "kalshi", "cleveland", gap),
            "edge_ours_over_cleveland": edge_over(group, "cleveland", "ours", gap),
        }
    return summary


def current_expectations(
    open_kalshi: pd.DataFrame,
    cleveland: pd.DataFrame,
    current_ours: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Upcoming releases: our forecast beside the market's and the Cleveland Fed's."""
    rows = []
    months = {}
    if not open_kalshi.empty:
        for row in open_kalshi.itertuples(index=False):
            months.setdefault((row.target_id, pd.Timestamp(row.reference_month)), {})["kalshi"] = row
    if not cleveland.empty:
        # Only months after the latest release count as upcoming (October 2025 CPI was never published).
        last_released = cleveland.dropna(subset=["release_date"]).groupby("target_id")["reference_month"].max()
        pending = cleveland[
            cleveland["release_date"].isna()
            & (cleveland["reference_month"] > cleveland["target_id"].map(last_released))
        ]
        for row in pending.itertuples(index=False):
            months.setdefault((row.target_id, pd.Timestamp(row.reference_month)), {})["cleveland"] = row
    for (target_id, month), parts in sorted(months.items()):
        market = parts.get("kalshi")
        nowcast = parts.get("cleveland")
        rows.append(
            {
                "target_id": target_id,
                "reference_month": f"{month:%Y-%m}",
                "ours": current_ours.get(f"{target_id}:{month:%Y-%m}"),
                "kalshi": None if market is None else float(market.kalshi),
                "kalshi_std": None if market is None else float(market.kalshi_std),
                "kalshi_range_80": None if market is None else [float(market.kalshi_low), float(market.kalshi_high)],
                "kalshi_range_open": None if market is None else [bool(v) for v in market.kalshi_range_open],
                "kalshi_volume": None if market is None else float(market.kalshi_volume),
                "kalshi_close": None if market is None else f"{pd.Timestamp(market.kalshi_close):%Y-%m-%d}",
                "cleveland": None if nowcast is None else float(nowcast.nowcast),
            }
        )
    return rows


def build_site_payload(summary: Mapping[str, Any], upcoming: list[dict[str, Any]], as_of: str) -> dict[str, Any]:
    """What the dashboard needs: upcoming expectations and each source's track record."""
    track = {}
    for target_id, record in summary.items():
        common = record.get("all_sources", {})
        edge = record.get("edge_ours_over_kalshi", {})
        track[target_id] = {
            "label": record.get("label"),
            "releases": common.get("releases"),
            "start": common.get("start"),
            "end": common.get("end"),
            "mae": {source: common[source]["mae"] for source in SOURCES if isinstance(common.get(source), dict)},
            "ours_edge_over_market": {
                "slope": edge.get("slope"),
                "t_stat": edge.get("t_stat"),
                "same_side_rate": edge.get("same_side_rate"),
                "differed": edge.get("differed"),
            },
        }
    return {
        "as_of_date": as_of,
        "upcoming": upcoming,
        "track_record": track,
        "notes": [
            "Market: the median of Kalshi's pre-release prices for each range of the print.",
            "Cleveland Fed: its inflation nowcast on the latest business day.",
            "Track record: average miss against the first print on releases every source covers, March-December "
            "2020 excluded.",
        ],
    }
