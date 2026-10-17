from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from world_model.data.csi import load_price_panel
from world_model.states.economic import EconomicStateEngine


@dataclass(frozen=True)
class GrowthTrendResult:
    panel: pd.DataFrame
    summary: pd.DataFrame
    annual_returns: pd.DataFrame
    trades: pd.DataFrame
    current_snapshot: pd.DataFrame


class GrowthTrendTimingEngine:
    def __init__(
        self,
        data_roots: dict[str, str],
        config: dict[str, Any],
        market_config: dict[str, Any],
        economic_config: dict[str, Any],
    ):
        self.data_roots = data_roots
        self.config = config
        self.market_config = market_config
        self.economic_config = economic_config

    def load_prices(self) -> pd.DataFrame:
        return load_price_panel(self.data_roots["csi_processed"], self.market_config["instruments"])

    def build_economic_state(self) -> pd.DataFrame:
        return EconomicStateEngine(self.data_roots, self.economic_config).build_state_history()

    def run(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> GrowthTrendResult:
        prices = self.load_prices().ffill()
        economic_state = self.build_economic_state()
        panel = build_growth_trend_panel(
            prices=prices,
            economic_state=economic_state,
            config=self.config,
            market_config=self.market_config,
        )
        if start is not None:
            panel = panel.loc[panel.index >= pd.Timestamp(start).normalize()]
        if end is not None:
            panel = panel.loc[panel.index <= pd.Timestamp(end).normalize()]
        strategies = self.config.get("strategies", [])
        strategy_ids = [strategy["id"] for strategy in strategies]
        summary = summarize_timing_strategies(
            panel=panel,
            strategies=strategies,
            annualization_days=int(self.config.get("annualization_days", 252)),
            whipsaw_days=int(self.config.get("whipsaw_days", 63)),
        )
        annual_returns = summarize_annual_returns(panel=panel, strategies=strategies)
        trades = build_trade_log(panel=panel, strategies=strategies, whipsaw_days=int(self.config.get("whipsaw_days", 63)))
        return GrowthTrendResult(
            panel=panel,
            summary=summary,
            annual_returns=annual_returns,
            trades=trades,
            current_snapshot=current_signal_snapshot(panel, strategies=strategies, strategy_ids=strategy_ids),
        )


def build_growth_trend_panel(
    prices: pd.DataFrame,
    economic_state: pd.DataFrame,
    config: dict[str, Any],
    market_config: dict[str, Any],
) -> pd.DataFrame:
    target_asset = config.get("target_asset", market_config.get("target_asset", "ES"))
    if target_asset not in prices:
        raise KeyError(f"Price panel is missing target asset: {target_asset}")

    price = prices[target_asset].dropna().rename("target_price")
    panel = pd.DataFrame(index=price.index)
    panel["target_price"] = price
    panel["target_return"] = price.pct_change()

    for rule_id, rule in config.get("trend_rules", {}).items():
        panel[f"trend_{rule_id}"] = build_trend_signal(price=price, rule=rule).reindex(panel.index).ffill()

    aligned_economic = economic_state.reindex(panel.index).ffill()
    for filter_id, growth_filter in config.get("growth_filters", {}).items():
        panel[f"growth_{filter_id}"] = build_growth_filter(aligned_economic, growth_filter)

    one_way_cost = float(config.get("one_way_transaction_cost", 0.0))
    defensive_return = build_defensive_return(panel.index, config)
    for strategy in config.get("strategies", []):
        exposure = build_strategy_exposure(panel=panel, strategy=strategy)
        strategy_id = strategy["id"]
        panel[f"exposure_{strategy_id}"] = exposure
        returns = strategy_returns(
            target_return=panel["target_return"],
            defensive_return=defensive_return,
            exposure=exposure,
            one_way_transaction_cost=one_way_cost,
        )
        panel[f"return_{strategy_id}"] = returns
        panel[f"equity_{strategy_id}"] = (1.0 + returns.fillna(0.0)).cumprod()
    return panel


def build_trend_signal(price: pd.Series, rule: dict[str, Any]) -> pd.Series:
    frequency = str(rule.get("frequency", "B")).upper()
    window = int(rule["window"])
    min_periods = int(rule.get("min_periods", window))
    include_current = bool(rule.get("include_current_in_average", True))
    if frequency in {"B", "D", "DAILY"}:
        average = price.rolling(window=window, min_periods=min_periods).mean()
        if not include_current:
            average = average.shift(1)
        signal = (price > average).astype(float).where(average.notna())
        return signal.rename("trend_signal")

    sampled = price.resample(frequency).last().dropna()
    average = sampled.rolling(window=window, min_periods=min_periods).mean()
    if not include_current:
        average = average.shift(1)
    signal = (sampled > average).astype(float).where(average.notna())
    return signal.reindex(price.index).ffill().rename("trend_signal")


def build_growth_filter(economic_state: pd.DataFrame, growth_filter: dict[str, Any]) -> pd.Series:
    rule_health = []
    for rule in growth_filter.get("rules", []):
        column = rule["column"]
        if column not in economic_state:
            raise KeyError(f"Economic state is missing growth filter column: {column}")
        value = economic_state[column].astype(float)
        comparator = _rule_comparator(value, rule)
        rule_health.append(_apply_operator(value, comparator, str(rule.get("operator", ">"))))

    if not rule_health:
        raise ValueError("Growth filter must define at least one rule.")
    frame = pd.concat(rule_health, axis=1)
    healthy_when = str(growth_filter.get("healthy_when", "all")).lower()
    if healthy_when == "all":
        healthy = frame.all(axis=1)
    elif healthy_when == "any":
        healthy = frame.any(axis=1)
    else:
        raise ValueError(f"Unsupported growth filter combiner: {healthy_when}")
    available = frame.notna().all(axis=1) if healthy_when == "all" else frame.notna().any(axis=1)
    return healthy.astype(float).where(available)


def build_strategy_exposure(panel: pd.DataFrame, strategy: dict[str, Any]) -> pd.Series:
    kind = strategy.get("kind")
    if kind == "buy_hold":
        return pd.Series(1.0, index=panel.index, name=strategy["id"])
    if kind == "trend":
        return panel[f"trend_{strategy['trend_rule']}"].astype(float).rename(strategy["id"])
    if kind == "economic":
        return panel[f"growth_{strategy['growth_filter']}"].astype(float).rename(strategy["id"])
    if kind == "growth_trend":
        growth = panel[f"growth_{strategy['growth_filter']}"].astype(float)
        trend = panel[f"trend_{strategy['trend_rule']}"].astype(float)
        exposure = pd.Series(np.nan, index=panel.index, dtype=float)
        available = growth.notna() & trend.notna()
        exposure.loc[available] = np.where(growth.loc[available].eq(1.0), 1.0, trend.loc[available])
        return exposure.rename(strategy["id"])
    raise ValueError(f"Unsupported timing strategy kind: {kind}")


def build_defensive_return(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.Series:
    defensive = config.get("defensive_return", "cash")
    if defensive != "cash":
        raise ValueError("Only cash defensive_return is currently supported.")
    return pd.Series(0.0, index=index, name="defensive_return")


def strategy_returns(
    target_return: pd.Series,
    defensive_return: pd.Series,
    exposure: pd.Series,
    one_way_transaction_cost: float,
) -> pd.Series:
    clean_exposure = exposure.astype(float)
    effective_exposure = clean_exposure.shift(1)
    blended = effective_exposure * target_return + (1.0 - effective_exposure) * defensive_return
    turnover = clean_exposure.diff().abs().where(clean_exposure.notna())
    return (blended - turnover.fillna(0.0) * one_way_transaction_cost).where(effective_exposure.notna())


def summarize_timing_strategies(
    panel: pd.DataFrame,
    strategies: list[dict[str, Any]],
    annualization_days: int = 252,
    whipsaw_days: int = 63,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    buy_hold_cagr = np.nan
    buy_hold_max_drawdown = np.nan
    common_index = common_strategy_return_index(panel, [strategy["id"] for strategy in strategies])

    for strategy in strategies:
        strategy_id = strategy["id"]
        returns = panel[f"return_{strategy_id}"].reindex(common_index).dropna()
        exposure = panel[f"exposure_{strategy_id}"].reindex(returns.index)
        if returns.empty:
            continue
        equity = (1.0 + returns.fillna(0.0)).cumprod()
        metrics = performance_metrics(
            returns=returns,
            equity=equity,
            exposure=exposure,
            annualization_days=annualization_days,
        )
        trades = trade_round_trips(
            exposure=panel[f"exposure_{strategy_id}"].reindex(common_index),
            price=panel["target_price"].reindex(common_index),
            whipsaw_days=whipsaw_days,
        )
        row = {
            "strategy": strategy_id,
            "strategy_label": strategy.get("label", strategy_id),
            "kind": strategy.get("kind", "unknown"),
            **metrics,
            **trades,
        }
        if strategy_id == "buy_hold":
            buy_hold_cagr = row["cagr"]
            buy_hold_max_drawdown = row["max_drawdown"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["excess_cagr_vs_buy_hold"] = summary["cagr"] - buy_hold_cagr
    summary["max_drawdown_improvement_vs_buy_hold"] = summary["max_drawdown"] - buy_hold_max_drawdown
    return summary.sort_values(["cagr"], ascending=False).reset_index(drop=True)


def common_strategy_return_index(panel: pd.DataFrame, strategy_ids: list[str]) -> pd.DatetimeIndex:
    common_index: pd.Index | None = None
    for strategy_id in strategy_ids:
        column = f"return_{strategy_id}"
        if column not in panel:
            continue
        index = panel[column].dropna().index
        common_index = index if common_index is None else common_index.intersection(index)
    if common_index is None or common_index.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(common_index).sort_values()


def summarize_annual_returns(panel: pd.DataFrame, strategies: list[dict[str, Any]]) -> pd.DataFrame:
    strategy_ids = [strategy["id"] for strategy in strategies]
    common_index = common_strategy_return_index(panel, strategy_ids)
    if common_index.empty:
        return pd.DataFrame()
    full_year_start = pd.Timestamp(common_index.min())
    full_year_end = pd.Timestamp(common_index.max())
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        strategy_id = strategy["id"]
        returns = panel[f"return_{strategy_id}"].reindex(common_index).dropna()
        for year, group in returns.groupby(returns.index.year):
            if group.empty:
                continue
            start_date = pd.Timestamp(group.index.min())
            end_date = pd.Timestamp(group.index.max())
            rows.append(
                {
                    "year": int(year),
                    "strategy": strategy_id,
                    "strategy_label": strategy.get("label", strategy_id),
                    "return": float((1.0 + group).prod() - 1.0),
                    "observations": int(len(group)),
                    "start_date": str(start_date.date()),
                    "end_date": str(end_date.date()),
                    "is_partial_year": bool(start_date == full_year_start or end_date == full_year_end),
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "strategy"]).reset_index(drop=True)


def performance_metrics(
    returns: pd.Series,
    equity: pd.Series,
    exposure: pd.Series,
    annualization_days: int,
) -> dict[str, float | int | str]:
    start_date = pd.Timestamp(returns.index[0])
    end_date = pd.Timestamp(returns.index[-1])
    years = max((end_date - start_date).days / 365.25, 1.0 / annualization_days)
    final_value = float(equity.iloc[-1])
    cagr = final_value ** (1.0 / years) - 1.0 if final_value > 0 else -1.0
    annual_vol = float(returns.std(ddof=0) * np.sqrt(annualization_days))
    sharpe = cagr / annual_vol if annual_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    daily_turnover = exposure.diff().abs()
    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "observations": int(len(returns)),
        "cagr": float(cagr),
        "annual_volatility": annual_vol,
        "sharpe_like": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else np.nan,
        "final_value": final_value,
        "average_exposure": float(exposure.mean()),
        "trade_count": int(daily_turnover.fillna(0.0).gt(0.0).sum()),
    }


def trade_round_trips(exposure: pd.Series, price: pd.Series, whipsaw_days: int) -> dict[str, float | int]:
    changes = exposure.dropna().diff()
    exit_dates = list(changes[changes < 0].index)
    entry_dates = list(changes[changes > 0].index)
    round_trips = 0
    whipsaws = 0
    days_out: list[int] = []
    missed_returns: list[float] = []
    for exit_date in exit_dates:
        later_entries = [date for date in entry_dates if date > exit_date]
        if not later_entries:
            continue
        entry_date = later_entries[0]
        round_trips += 1
        days = len(price.loc[(price.index > exit_date) & (price.index <= entry_date)])
        days_out.append(days)
        exit_price = price.get(exit_date, np.nan)
        entry_price = price.get(entry_date, np.nan)
        if pd.notna(exit_price) and pd.notna(entry_price) and exit_price != 0:
            missed = float(entry_price / exit_price - 1.0)
            missed_returns.append(missed)
            if days <= whipsaw_days and missed > 0.0:
                whipsaws += 1
    return {
        "round_trips": int(round_trips),
        "whipsaw_count": int(whipsaws),
        "avg_days_out": float(np.mean(days_out)) if days_out else np.nan,
        "avg_missed_return_while_out": float(np.mean(missed_returns)) if missed_returns else np.nan,
    }


def build_trade_log(panel: pd.DataFrame, strategies: list[dict[str, Any]], whipsaw_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        strategy_id = strategy["id"]
        exposure = panel[f"exposure_{strategy_id}"].dropna()
        changes = exposure.diff()
        for date, change in changes[changes.abs() > 0].items():
            prior = float(exposure.loc[:date].iloc[-2]) if len(exposure.loc[:date]) >= 2 else np.nan
            current = float(exposure.loc[date])
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "strategy": strategy_id,
                    "strategy_label": strategy.get("label", strategy_id),
                    "from_exposure": prior,
                    "to_exposure": current,
                    "change": float(change),
                    "target_price": float(panel.loc[date, "target_price"]),
                    "whipsaw_window_days": int(whipsaw_days),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "strategy",
                "strategy_label",
                "from_exposure",
                "to_exposure",
                "change",
                "target_price",
                "whipsaw_window_days",
            ]
        )
    return pd.DataFrame(rows).sort_values(["strategy", "date"]).reset_index(drop=True)


def current_signal_snapshot(
    panel: pd.DataFrame,
    strategies: list[dict[str, Any]],
    strategy_ids: list[str],
) -> pd.DataFrame:
    usable = panel.dropna(subset=["target_price"])
    if usable.empty:
        return pd.DataFrame()
    row = usable.iloc[-1]
    records: list[dict[str, Any]] = []
    for strategy in strategies:
        strategy_id = strategy["id"]
        exposure_column = f"exposure_{strategy_id}"
        records.append(
            {
                "as_of_date": pd.Timestamp(row.name),
                "strategy": strategy_id,
                "strategy_label": strategy.get("label", strategy_id),
                "kind": strategy.get("kind", "unknown"),
                "exposure": _clean_float(row.get(exposure_column)),
                "target_price": _clean_float(row.get("target_price")),
            }
        )
    return pd.DataFrame(records)


def write_growth_trend_outputs(result: GrowthTrendResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": root / "growth_trend_timing_panel.parquet",
        "summary": root / "growth_trend_timing_summary.csv",
        "annual_returns": root / "growth_trend_timing_annual_returns.csv",
        "trades": root / "growth_trend_timing_trades.csv",
        "current_snapshot": root / "growth_trend_timing_current_snapshot.csv",
    }
    result.panel.to_parquet(paths["panel"])
    result.summary.to_csv(paths["summary"], index=False)
    result.annual_returns.to_csv(paths["annual_returns"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    result.current_snapshot.to_csv(paths["current_snapshot"], index=False)
    return paths


def _rule_comparator(value: pd.Series, rule: dict[str, Any]) -> pd.Series | float:
    if "moving_average_window" in rule:
        window = int(rule["moving_average_window"])
        min_periods = int(rule.get("min_periods", window))
        return value.rolling(window=window, min_periods=min_periods).mean()
    return float(rule.get("threshold", 0.0))


def _apply_operator(value: pd.Series, comparator: pd.Series | float, operator: str) -> pd.Series:
    if operator == ">":
        result = value > comparator
    elif operator == ">=":
        result = value >= comparator
    elif operator == "<":
        result = value < comparator
    elif operator == "<=":
        result = value <= comparator
    else:
        raise ValueError(f"Unsupported rule operator: {operator}")
    available = value.notna()
    if isinstance(comparator, pd.Series):
        available &= comparator.notna()
    return result.astype(float).where(available)


def _clean_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
