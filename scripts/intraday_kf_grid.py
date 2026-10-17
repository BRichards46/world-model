from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".codex_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pandas as pd


COST_BPS_GRID = [0.0, 0.25, 1.0]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    bar_minutes: int
    source: str


def pine_ema(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values).ewm(span=period, adjust=False).mean().to_numpy()


def adaptive_kf_smooth(
    df: pd.DataFrame,
    order: int = 1,
    p0: float = 100.0,
    proc_base: float = 0.02,
    meas_base: float = 3.0,
    p_sha: int = 10,
    atr_short: int = 14,
    atr_long: int = 63,
    warmup: int = 10,
) -> pd.Series:
    open_ = df["open"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float, index=df.index, name="kf")

    ema_o = pine_ema(open_, p_sha)
    ema_h = pine_ema(high, p_sha)
    ema_l = pine_ema(low, p_sha)
    ema_c = pine_ema(close, p_sha)

    ha_close = (ema_o + ema_h + ema_l + ema_c) / 4.0
    ha_open = np.empty(n, dtype=float)
    ha_open[0] = ema_o[0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) * 0.5
    ha_high = np.maximum(ema_h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(ema_l, np.minimum(ha_open, ha_close))

    sha_open = pine_ema(ha_open, p_sha)
    sha_high = pine_ema(ha_high, p_sha)
    sha_low = pine_ema(ha_low, p_sha)
    sha_close = pine_ema(ha_close, p_sha)

    oc_max = np.maximum(sha_open, sha_close)
    oc_min = np.minimum(sha_open, sha_close)
    sha_body = sha_close - sha_open
    sha_upper_wick = np.maximum(sha_high - oc_max, 0.0)
    sha_lower_wick = np.maximum(oc_min - sha_low, 0.0)

    rng = np.maximum(high - low, 1e-9)
    body_abs = np.abs(sha_body)
    chop = np.clip((sha_upper_wick + sha_lower_wick) / (body_abs + 1e-9), 0.0, 5.0)
    trend = np.clip(body_abs / rng, 0.0, 2.0)
    atr14 = pine_ema(rng, atr_short)
    atr63 = pine_ema(rng, atr_long)
    atr_ratio = np.maximum(atr14 / np.maximum(atr63, 1e-9), 0.0)

    q_t = proc_base * (1.0 + 16.0 * trend)
    r_t = np.clip(meas_base * (1.0 + 2.0 * chop) * atr_ratio, 0.5, 6.0)
    log_price = np.log(np.maximum(close, 1e-12))

    x = np.full(order, log_price[0], dtype=float)
    p = np.full(order, p0, dtype=float)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if i <= warmup or not np.isfinite(q_t[i]) or not np.isfinite(r_t[i]):
            continue
        z = log_price[i]
        if not np.isfinite(z):
            continue
        for k in range(order):
            pk = p[k] + q_t[i]
            kk = pk / (pk + r_t[i])
            x[k] = x[k] + kk * (z - x[k])
            p[k] = (1.0 - kk) * pk
        out[i] = math.exp(x[0])
    return pd.Series(out, index=df.index, name="kf")


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    if spec.source == "spy_parquet":
        df = pd.read_parquet(spec.path)
        df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert("America/New_York")
        keep = ["datetime", "open", "high", "low", "close", "volume"]
        df = df[keep]
    elif spec.source == "tradingview_csv":
        df = pd.read_csv(spec.path)
        df["datetime"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
        df = df.rename(columns={"KF_adapt_smooth": "tv_kf"})
        if "volume" not in df.columns:
            df["volume"] = 1.0
        keep = ["datetime", "open", "high", "low", "close", "volume", "tv_kf"]
        keep = [col for col in keep if col in df.columns]
        df = df[keep]
    else:
        raise ValueError(f"Unsupported source: {spec.source}")

    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    local_time = df["datetime"].dt.time
    rth = (local_time >= pd.Timestamp("09:30").time()) & (local_time <= pd.Timestamp("16:00").time())
    df = df.loc[rth].copy()
    df["session"] = df["datetime"].dt.date.astype(str)
    df = df.groupby("session", group_keys=False).filter(lambda x: len(x) >= max(20, int(180 / spec.bar_minutes)))
    df = df.reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    out = df.copy()
    out["kf"] = adaptive_kf_smooth(out)
    out["typical_price"] = (out["high"] + out["low"] + out["close"]) / 3.0
    volume = out["volume"].fillna(0.0).clip(lower=0.0)
    pv = out["typical_price"] * volume
    cum_pv = pv.groupby(out["session"]).cumsum()
    cum_volume = volume.groupby(out["session"]).cumsum()
    out["vwap"] = (cum_pv / cum_volume.replace(0.0, np.nan)).fillna(out["typical_price"])
    out["session_open"] = out.groupby("session")["open"].transform("first")
    out["session_high"] = out.groupby("session")["high"].cummax()
    out["session_low"] = out.groupby("session")["low"].cummin()
    out["session_return"] = out["close"] / out["session_open"] - 1.0
    out["drawdown_from_session_high"] = out["close"] / out["session_high"] - 1.0
    session_range = (out["session_high"] - out["session_low"]).replace(0.0, np.nan)
    out["session_range_pos"] = ((out["close"] - out["session_low"]) / session_range).fillna(0.5)

    rng = (out["high"] - out["low"]).clip(lower=1e-9)
    short_span = max(2, int(round(20 / bar_minutes)))
    long_span = max(short_span + 1, int(round(120 / bar_minutes)))
    out["range_ema_short"] = rng.ewm(span=short_span, adjust=False).mean()
    out["range_ema_long"] = rng.ewm(span=long_span, adjust=False).mean()
    out["range_ratio"] = out["range_ema_short"] / out["range_ema_long"].replace(0.0, np.nan)
    return out


def bars_from_minutes(minutes: int, bar_minutes: int) -> int:
    return max(2, int(round(minutes / bar_minutes)))


def strategy_return(
    df: pd.DataFrame,
    desired_position: pd.Series,
    one_way_cost_bps: float = 1.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    desired = desired_position.astype(float).reindex(df.index).fillna(0.0).clip(-1.0, 1.0)
    same_next_session = df["session"].eq(df["session"].shift(-1))
    desired = desired.where(same_next_session, 0.0)

    same_prev_session = df["session"].eq(df["session"].shift(1))
    base_ret = df["close"].pct_change().where(same_prev_session, 0.0).fillna(0.0)
    executed_position = desired.shift(1).fillna(0.0)
    turnover = desired.diff().abs().fillna(desired.abs())
    net_ret = executed_position * base_ret - turnover * (one_way_cost_bps / 10_000.0)
    return net_ret.rename("return"), executed_position.rename("position"), turnover.rename("turnover")


def annualized_return(total_return: float, sessions: int) -> float:
    if sessions <= 0 or total_return <= -1.0:
        return np.nan
    return (1.0 + total_return) ** (252.0 / sessions) - 1.0


def period_cagr(daily_returns: pd.Series, start: str, end: str) -> float:
    subset = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= end)]
    if subset.empty:
        return np.nan
    total = float((1.0 + subset).prod() - 1.0)
    return annualized_return(total, len(subset))


def summarize_returns(
    dataset: str,
    family: str,
    params: dict[str, object],
    returns: pd.Series,
    position: pd.Series,
    turnover: pd.Series,
    sessions: pd.Series,
) -> dict[str, object]:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    drawdown = equity / equity.cummax() - 1.0
    daily = (1.0 + returns).groupby(sessions).prod() - 1.0
    daily.index = pd.to_datetime(daily.index)
    sessions_count = int(daily.size)
    daily_std = float(daily.std(ddof=0))
    sharpe = np.nan if daily_std == 0.0 else float(np.sqrt(252.0) * daily.mean() / daily_std)
    cagr = annualized_return(total_return, sessions_count)
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    calmar = np.nan if max_dd == 0.0 or not np.isfinite(max_dd) else cagr / abs(max_dd)
    position_changes = int(turnover.gt(0.0).sum())

    row: dict[str, object] = {
        "dataset": dataset,
        "family": family,
        **params,
        "sessions": sessions_count,
        "total_return": total_return,
        "cagr": cagr,
        "daily_sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "daily_win_rate": float(daily.gt(0.0).mean()) if sessions_count else np.nan,
        "avg_abs_exposure": float(position.abs().mean()),
        "long_fraction": float(position.gt(0.0).mean()),
        "short_fraction": float(position.lt(0.0).mean()),
        "turnover_per_day": float(turnover.sum() / sessions_count) if sessions_count else np.nan,
        "position_changes_per_day": float(position_changes / sessions_count) if sessions_count else np.nan,
        "cagr_2013_2018": period_cagr(daily, "2013-01-01", "2018-12-31"),
        "cagr_2019_2022": period_cagr(daily, "2019-01-01", "2022-12-31"),
        "cagr_2023_2026": period_cagr(daily, "2023-01-01", "2026-12-31"),
    }
    return row


def trigger_masks(df: pd.DataFrame, close_ma_span: int, trigger: str) -> tuple[pd.Series, pd.Series]:
    close_ma = df["close"].ewm(span=close_ma_span, adjust=False).mean()
    if trigger == "close_ma_gt_kf":
        return close_ma.gt(df["kf"]), close_ma.lt(df["kf"])
    if trigger == "kf_gt_close_ma":
        return df["kf"].gt(close_ma), df["kf"].lt(close_ma)
    raise ValueError(f"Unsupported trigger: {trigger}")


def threshold_trigger_masks(
    df: pd.DataFrame,
    close_ma_span: int,
    trigger: str,
    deadband_bps: float,
) -> tuple[pd.Series, pd.Series]:
    close_ma = df["close"].ewm(span=close_ma_span, adjust=False).mean()
    if trigger == "close_ma_gt_kf":
        spread_bps = (close_ma / df["kf"] - 1.0) * 10_000.0
    elif trigger == "kf_gt_close_ma":
        spread_bps = (df["kf"] / close_ma - 1.0) * 10_000.0
    else:
        raise ValueError(f"Unsupported trigger: {trigger}")
    return spread_bps.gt(deadband_bps), spread_bps.lt(-deadband_bps)


def state_machine_position(
    candidate: pd.Series,
    sessions: pd.Series,
    confirm_bars: int,
    min_hold_bars: int,
) -> pd.Series:
    values = candidate.astype(float).to_numpy()
    session_values = sessions.to_numpy()
    out = np.zeros(len(values), dtype=float)
    pos = 0.0
    pending = np.nan
    pending_count = 0
    bars_since_change = min_hold_bars
    prev_session = session_values[0] if len(session_values) else None

    for i, target in enumerate(values):
        if i == 0 or session_values[i] != prev_session:
            pos = 0.0
            pending = np.nan
            pending_count = 0
            bars_since_change = min_hold_bars
            prev_session = session_values[i]

        if np.isnan(target) or target == pos:
            pending = np.nan
            pending_count = 0
        else:
            if np.isfinite(pending) and target == pending:
                pending_count += 1
            else:
                pending = target
                pending_count = 1
            if pending_count >= confirm_bars and bars_since_change >= min_hold_bars:
                pos = float(target)
                pending = np.nan
                pending_count = 0
                bars_since_change = 0

        out[i] = pos
        bars_since_change += 1

    return pd.Series(out, index=candidate.index)


def desired_ma_vs_kf(df: pd.DataFrame, close_ma_span: int, use_vwap: bool, trigger: str) -> pd.Series:
    long_ok, short_ok = trigger_masks(df, close_ma_span, trigger)
    if use_vwap:
        long_ok &= df["close"].gt(df["vwap"])
        short_ok &= df["close"].lt(df["vwap"])
    return pd.Series(np.select([long_ok, short_ok], [1.0, -1.0], default=0.0), index=df.index)


def desired_kf_gate_ma_vs_kf(
    df: pd.DataFrame,
    close_ma_span: int,
    gate_span: int,
    use_vwap: bool,
    trigger: str,
) -> pd.Series:
    kf_gate = df["kf"].ewm(span=gate_span, adjust=False).mean()
    base_long, base_short = trigger_masks(df, close_ma_span, trigger)
    gate_long = df["kf"].gt(kf_gate)
    gate_short = df["kf"].lt(kf_gate)
    long_ok = base_long & gate_long
    short_ok = base_short & gate_short
    if use_vwap:
        long_ok &= df["close"].gt(df["vwap"])
        short_ok &= df["close"].lt(df["vwap"])
    return pd.Series(np.select([long_ok, short_ok], [1.0, -1.0], default=0.0), index=df.index)


def desired_tape_state(
    df: pd.DataFrame,
    close_ma_span: int,
    gate_span: int,
    loss_bps: int,
    drawdown_bps: int,
    trigger: str,
) -> pd.Series:
    kf_gate = df["kf"].ewm(span=gate_span, adjust=False).mean()
    base_long, base_short = trigger_masks(df, close_ma_span, trigger)
    gate_long = df["kf"].gt(kf_gate)
    gate_short = df["kf"].lt(kf_gate)

    constructive = (
        df["close"].gt(df["vwap"])
        & gate_long
        & df["session_range_pos"].ge(0.45)
        & df["session_return"].gt(-loss_bps / 10_000.0)
        & df["range_ratio"].lt(1.75)
    )
    liquidation = (
        df["close"].lt(df["vwap"])
        & gate_short
        & df["session_return"].le(-loss_bps / 10_000.0)
        & df["drawdown_from_session_high"].le(-drawdown_bps / 10_000.0)
        & df["session_range_pos"].le(0.35)
    )

    neutral_long = base_long & gate_long & df["close"].gt(df["vwap"])
    neutral_short = base_short & gate_short & df["close"].lt(df["vwap"])
    desired = pd.Series(0.0, index=df.index)
    desired.loc[neutral_long] = 1.0
    desired.loc[neutral_short] = -1.0
    desired.loc[constructive] = 1.0
    desired.loc[liquidation & base_short] = -1.0
    desired.loc[liquidation & ~base_short] = 0.0
    return desired


def desired_kf_state_machine(
    df: pd.DataFrame,
    close_ma_span: int,
    gate_span: int,
    trigger: str,
    deadband_bps: float,
    confirm_bars: int,
    min_hold_bars: int,
    use_vwap: bool,
) -> pd.Series:
    kf_gate = df["kf"].ewm(span=gate_span, adjust=False).mean()
    gate_spread_bps = (df["kf"] / kf_gate - 1.0) * 10_000.0
    gate_long = gate_spread_bps.gt(deadband_bps)
    gate_short = gate_spread_bps.lt(-deadband_bps)
    base_long, base_short = threshold_trigger_masks(df, close_ma_span, trigger, deadband_bps)

    long_ok = base_long & gate_long
    short_ok = base_short & gate_short
    if use_vwap:
        long_ok &= df["close"].gt(df["vwap"])
        short_ok &= df["close"].lt(df["vwap"])

    candidate = pd.Series(np.nan, index=df.index, dtype=float)
    candidate.loc[long_ok] = 1.0
    candidate.loc[short_ok] = -1.0
    return state_machine_position(candidate, df["session"], confirm_bars, min_hold_bars)


def desired_tape_state_machine(
    df: pd.DataFrame,
    close_ma_span: int,
    gate_span: int,
    trigger: str,
    deadband_bps: float,
    confirm_bars: int,
    min_hold_bars: int,
    loss_bps: int,
    drawdown_bps: int,
) -> pd.Series:
    kf_gate = df["kf"].ewm(span=gate_span, adjust=False).mean()
    gate_spread_bps = (df["kf"] / kf_gate - 1.0) * 10_000.0
    gate_long = gate_spread_bps.gt(deadband_bps)
    gate_short = gate_spread_bps.lt(-deadband_bps)
    base_long, base_short = threshold_trigger_masks(df, close_ma_span, trigger, deadband_bps)

    constructive = (
        df["close"].gt(df["vwap"])
        & gate_long
        & df["session_range_pos"].ge(0.45)
        & df["session_return"].gt(-loss_bps / 10_000.0)
        & df["range_ratio"].lt(1.75)
    )
    liquidation = (
        df["close"].lt(df["vwap"])
        & gate_short
        & df["session_return"].le(-loss_bps / 10_000.0)
        & df["drawdown_from_session_high"].le(-drawdown_bps / 10_000.0)
        & df["session_range_pos"].le(0.35)
    )
    neutral_long = base_long & gate_long & df["close"].gt(df["vwap"])
    neutral_short = base_short & gate_short & df["close"].lt(df["vwap"])

    candidate = pd.Series(np.nan, index=df.index, dtype=float)
    candidate.loc[neutral_long | constructive] = 1.0
    candidate.loc[neutral_short | liquidation] = -1.0
    return state_machine_position(candidate, df["session"], confirm_bars, min_hold_bars)


def iter_grid(df: pd.DataFrame, bar_minutes: int) -> Iterable[tuple[str, dict[str, object], pd.Series]]:
    close_ma_minutes = [5, 10, 20, 30, 60, 120]
    gate_minutes = [60, 120, 240, 390, 780]
    triggers = ["close_ma_gt_kf", "kf_gt_close_ma"]
    for trigger in triggers:
        for close_minutes in close_ma_minutes:
            close_span = bars_from_minutes(close_minutes, bar_minutes)
            for use_vwap in [False, True]:
                yield (
                    "ma_vs_kf",
                    {
                        "trigger": trigger,
                        "close_ma_minutes": close_minutes,
                        "gate_minutes": np.nan,
                        "use_vwap": use_vwap,
                        "loss_bps": np.nan,
                        "drawdown_bps": np.nan,
                        "deadband_bps": np.nan,
                        "confirm_minutes": np.nan,
                        "min_hold_minutes": np.nan,
                    },
                    desired_ma_vs_kf(df, close_span, use_vwap, trigger),
                )
            for gate_minutes_ in gate_minutes:
                gate_span = bars_from_minutes(gate_minutes_, bar_minutes)
                for use_vwap in [False, True]:
                    yield (
                        "kf_gate_ma_vs_kf",
                        {
                            "trigger": trigger,
                            "close_ma_minutes": close_minutes,
                            "gate_minutes": gate_minutes_,
                            "use_vwap": use_vwap,
                            "loss_bps": np.nan,
                            "drawdown_bps": np.nan,
                            "deadband_bps": np.nan,
                            "confirm_minutes": np.nan,
                            "min_hold_minutes": np.nan,
                        },
                        desired_kf_gate_ma_vs_kf(df, close_span, gate_span, use_vwap, trigger),
                    )
                for loss_bps in [20, 35, 50]:
                    yield (
                        "tape_state",
                        {
                            "trigger": trigger,
                            "close_ma_minutes": close_minutes,
                            "gate_minutes": gate_minutes_,
                            "use_vwap": True,
                            "loss_bps": loss_bps,
                            "drawdown_bps": 50,
                            "deadband_bps": np.nan,
                            "confirm_minutes": np.nan,
                            "min_hold_minutes": np.nan,
                        },
                        desired_tape_state(
                            df,
                            close_span,
                            gate_span,
                            loss_bps=loss_bps,
                            drawdown_bps=50,
                            trigger=trigger,
                        ),
                    )

    anti_whipsaw_close_minutes = [60, 120]
    anti_whipsaw_gate_minutes = [390, 780]
    anti_whipsaw_deadbands = [5, 10]
    anti_whipsaw_min_holds = [60, 120]
    trigger = "kf_gt_close_ma"
    for close_minutes in anti_whipsaw_close_minutes:
        close_span = bars_from_minutes(close_minutes, bar_minutes)
        for gate_minutes_ in anti_whipsaw_gate_minutes:
            gate_span = bars_from_minutes(gate_minutes_, bar_minutes)
            for deadband_bps in anti_whipsaw_deadbands:
                for min_hold_minutes in anti_whipsaw_min_holds:
                    confirm_minutes = 10
                    confirm_bars = bars_from_minutes(confirm_minutes, bar_minutes)
                    min_hold_bars = bars_from_minutes(min_hold_minutes, bar_minutes)
                    for use_vwap in [False, True]:
                        yield (
                            "kf_state_machine",
                            {
                                "trigger": trigger,
                                "close_ma_minutes": close_minutes,
                                "gate_minutes": gate_minutes_,
                                "use_vwap": use_vwap,
                                "loss_bps": np.nan,
                                "drawdown_bps": np.nan,
                                "deadband_bps": deadband_bps,
                                "confirm_minutes": confirm_minutes,
                                "min_hold_minutes": min_hold_minutes,
                            },
                            desired_kf_state_machine(
                                df,
                                close_span,
                                gate_span,
                                trigger=trigger,
                                deadband_bps=deadband_bps,
                                confirm_bars=confirm_bars,
                                min_hold_bars=min_hold_bars,
                                use_vwap=use_vwap,
                            ),
                        )
                    yield (
                        "tape_state_machine",
                        {
                            "trigger": trigger,
                            "close_ma_minutes": close_minutes,
                            "gate_minutes": gate_minutes_,
                            "use_vwap": True,
                            "loss_bps": 35,
                            "drawdown_bps": 50,
                            "deadband_bps": deadband_bps,
                            "confirm_minutes": confirm_minutes,
                            "min_hold_minutes": min_hold_minutes,
                        },
                        desired_tape_state_machine(
                            df,
                            close_span,
                            gate_span,
                            trigger=trigger,
                            deadband_bps=deadband_bps,
                            confirm_bars=confirm_bars,
                            min_hold_bars=min_hold_bars,
                            loss_bps=35,
                            drawdown_bps=50,
                        ),
                    )


def buy_hold_summary(dataset: str, df: pd.DataFrame, cost_bps: float) -> dict[str, object]:
    desired = pd.Series(1.0, index=df.index)
    returns, position, turnover = strategy_return(df, desired, one_way_cost_bps=cost_bps)
    return summarize_returns(
        dataset=dataset,
        family="rth_buy_hold",
        params={
            "cost_bps": cost_bps,
            "trigger": np.nan,
            "close_ma_minutes": np.nan,
            "gate_minutes": np.nan,
            "use_vwap": False,
            "loss_bps": np.nan,
            "drawdown_bps": np.nan,
            "deadband_bps": np.nan,
            "confirm_minutes": np.nan,
            "min_hold_minutes": np.nan,
        },
        returns=returns,
        position=position,
        turnover=turnover,
        sessions=df["session"],
    )


def run_dataset(spec: DatasetSpec, out_dir: Path) -> pd.DataFrame:
    raw = load_dataset(spec)
    featured = add_features(raw, spec.bar_minutes)
    featured = featured.dropna(subset=["kf"]).reset_index(drop=True)

    rows = [buy_hold_summary(spec.name, featured, cost_bps) for cost_bps in COST_BPS_GRID]
    for family, params, desired in iter_grid(featured, spec.bar_minutes):
        for cost_bps in COST_BPS_GRID:
            returns, position, turnover = strategy_return(featured, desired, one_way_cost_bps=cost_bps)
            rows.append(
                summarize_returns(
                    dataset=spec.name,
                    family=family,
                    params={"cost_bps": cost_bps, **params},
                    returns=returns,
                    position=position,
                    turnover=turnover,
                    sessions=featured["session"],
                )
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["daily_sharpe", "cagr"], ascending=False)
    summary.to_csv(out_dir / f"{spec.name.lower()}_summary.csv", index=False)

    if "tv_kf" in raw.columns:
        valid = featured[["kf", "tv_kf"]].dropna()
        if not valid.empty:
            diagnostics = pd.DataFrame(
                [
                    {
                        "dataset": spec.name,
                        "rows_compared": len(valid),
                        "corr_computed_vs_tv_kf": valid["kf"].corr(valid["tv_kf"]),
                        "median_abs_diff": float((valid["kf"] - valid["tv_kf"]).abs().median()),
                        "max_abs_diff": float((valid["kf"] - valid["tv_kf"]).abs().max()),
                    }
                ]
            )
            diagnostics.to_csv(out_dir / f"{spec.name.lower()}_kf_diagnostics.csv", index=False)

    return summary


def write_report(combined: pd.DataFrame, out_dir: Path) -> None:
    report_path = out_dir / "report.md"

    def format_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        return str(value)

    def markdown_table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            rows.append("| " + " | ".join(format_cell(value) for value in row) + " |")
        return "\n".join(rows)

    lines = [
        "# Intraday KF Grid",
        "",
        "Assumptions:",
        "",
        "- Recomputed the Pine-style adaptive KF from OHLC using the default parameters in the prompt.",
        "- Regular trading hours only: 09:30-16:00 America/New_York.",
        "- Positions are flattened at the session boundary; no overnight return is taken.",
        "- Signals are computed on the current bar and applied to the next bar's close-to-close return.",
        "- Results are run at 0, 0.25, and 1 bp one-way transaction cost on absolute exposure changes.",
        "- `ma_vs_kf`: tests both trigger orientations: close EMA above/below KF and KF above/below close EMA.",
        "- `kf_gate_ma_vs_kf`: same trigger, but longs require KF > EMA(KF), shorts require KF < EMA(KF).",
        "- `tape_state`: VWAP/session-state version; constructive tape holds long, liquidation tape allows shorts, neutral tape uses gated MA-vs-KF.",
        "- `kf_state_machine` and `tape_state_machine`: anti-whipsaw variants using deadbands, 10-minute confirmation, and 60/120-minute minimum holds.",
        "",
    ]
    for dataset, group in combined.groupby("dataset", sort=False):
        lines.extend([f"## {dataset}", ""])
        cols = [
            "family",
            "cost_bps",
            "trigger",
            "close_ma_minutes",
            "gate_minutes",
            "use_vwap",
            "loss_bps",
            "deadband_bps",
            "confirm_minutes",
            "min_hold_minutes",
            "cagr",
            "daily_sharpe",
            "max_drawdown",
            "turnover_per_day",
            "avg_abs_exposure",
            "cagr_2019_2022",
            "cagr_2023_2026",
        ]
        for cost_bps in [1.0, 0.25, 0.0]:
            top = (
                group[(group["family"] != "rth_buy_hold") & group["cost_bps"].eq(cost_bps)]
                .sort_values(["daily_sharpe", "cagr"], ascending=False)
                .head(8)
            )
            lines.append(f"Top strategies at {cost_bps:g} bp one-way cost:")
            lines.append("")
            lines.append(markdown_table(top[cols]))
            lines.append("")
            low_turnover = (
                group[
                    (group["family"] != "rth_buy_hold")
                    & group["cost_bps"].eq(cost_bps)
                    & group["turnover_per_day"].le(5.0)
                ]
                .sort_values(["daily_sharpe", "cagr"], ascending=False)
                .head(8)
            )
            lines.append(f"Top low-turnover strategies at {cost_bps:g} bp one-way cost, <=5 turns/day:")
            lines.append("")
            lines.append(markdown_table(low_turnover[cols]))
            lines.append("")
        bh = group[group["family"] == "rth_buy_hold"]
        if not bh.empty:
            lines.append("Buy-hold RTH baseline:")
            lines.append("")
            lines.append(
                markdown_table(
                    bh[
                        [
                            "cost_bps",
                            "cagr",
                            "daily_sharpe",
                            "max_drawdown",
                            "turnover_per_day",
                            "avg_abs_exposure",
                            "cagr_2019_2022",
                            "cagr_2023_2026",
                        ]
                    ]
                )
            )
            lines.append("")
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the intraday adaptive-KF parameter grid.")
    parser.add_argument("--spy-path", type=Path, required=True, help="SPY one-minute parquet input")
    parser.add_argument("--es-path", type=Path, required=True, help="ES five-minute TradingView CSV input")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "intraday_kf_grid",
        help="Directory for summaries and the report",
    )
    args = parser.parse_args()

    datasets = [
        DatasetSpec("SPY_1m_RTH", args.spy_path.expanduser(), 1, "spy_parquet"),
        DatasetSpec("ES_5m_RTH", args.es_path.expanduser(), 5, "tradingview_csv"),
    ]
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for spec in datasets:
        summaries.append(run_dataset(spec, out_dir))
    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.sort_values(["dataset", "daily_sharpe", "cagr"], ascending=[True, False, False])
    combined.to_csv(out_dir / "combined_summary.csv", index=False)
    write_report(combined, out_dir)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
