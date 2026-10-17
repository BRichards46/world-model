from __future__ import annotations

from pathlib import Path

import pandas as pd

from world_model.states.growth_trend import GrowthTrendResult


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def format_num(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def write_growth_trend_report(result: GrowthTrendResult, output_path: str | Path) -> None:
    lines: list[str] = []
    as_of = _latest_date(result.panel)
    lines.append(f"# Growth-Trend Timing - {as_of}")
    lines.append("")
    _add_current_snapshot(lines, result.current_snapshot)
    _add_summary_table(lines, result.summary)
    _add_annual_return_table(lines, result.annual_returns)
    _add_recent_trade_table(lines, result.trades)
    lines.append("## Notes")
    lines.append("")
    lines.append("- GTT exposure is long when the growth filter is healthy; otherwise it defers to the trend rule.")
    lines.append("- Trend and growth signals are observed at the close and applied to the next return.")
    lines.append("- The first pass uses cash as the defensive asset and subtracts the configured one-way transaction cost on exposure changes.")
    lines.append("- FRED inputs come from the point-in-time economic state engine, so historical growth filters use data available at the time.")
    lines.append("")
    _write_lines(lines, output_path)


def _add_current_snapshot(lines: list[str], snapshot: pd.DataFrame) -> None:
    lines.append("## Current Exposure")
    lines.append("")
    if snapshot.empty:
        lines.append("No current snapshot available.")
        lines.append("")
        return
    lines.append("| Strategy | Kind | Exposure | Target price |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in snapshot.to_dict("records"):
        exposure = row.get("exposure")
        price = row.get("target_price")
        lines.append(
            "| {label} | {kind} | {exposure} | {price} |".format(
                label=row["strategy_label"],
                kind=row["kind"],
                exposure=format_pct(exposure),
                price=format_num(price),
            )
        )
    lines.append("")


def _add_summary_table(lines: list[str], summary: pd.DataFrame) -> None:
    lines.append("## Strategy Summary")
    lines.append("")
    if summary.empty:
        lines.append("No strategy summary available.")
        lines.append("")
        return
    lines.append(
        "| Strategy | CAGR | Vol | Max DD | Excess CAGR | DD improvement | Avg exposure | Trades | Whipsaws |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    display = summary.sort_values("cagr", ascending=False)
    for row in display.to_dict("records"):
        lines.append(
            "| {label} | {cagr} | {vol} | {dd} | {excess} | {dd_imp} | {exposure} | {trades} | {whipsaws} |".format(
                label=row["strategy_label"],
                cagr=format_pct(row["cagr"]),
                vol=format_pct(row["annual_volatility"]),
                dd=format_pct(row["max_drawdown"]),
                excess=format_pct(row["excess_cagr_vs_buy_hold"]),
                dd_imp=format_pct(row["max_drawdown_improvement_vs_buy_hold"]),
                exposure=format_pct(row["average_exposure"]),
                trades=int(row["trade_count"]),
                whipsaws=int(row["whipsaw_count"]),
            )
        )
    lines.append("")


def _add_annual_return_table(lines: list[str], annual_returns: pd.DataFrame) -> None:
    lines.append("## Calendar-Year Returns")
    lines.append("")
    if annual_returns.empty:
        lines.append("No annual returns available.")
        lines.append("")
        return
    pivot = annual_returns.pivot(index="year", columns="strategy_label", values="return")
    partial = annual_returns.groupby("year")["is_partial_year"].any()
    columns = list(pivot.columns)
    lines.append("| Year | Partial | " + " | ".join(columns) + " |")
    lines.append("| ---: | --- | " + " | ".join(["---:"] * len(columns)) + " |")
    for year, row in pivot.sort_index().iterrows():
        label = "yes" if bool(partial.get(year, False)) else ""
        values = [format_pct(row.get(column)) for column in columns]
        lines.append(f"| {int(year)} | {label} | " + " | ".join(values) + " |")
    lines.append("")


def _add_recent_trade_table(lines: list[str], trades: pd.DataFrame) -> None:
    lines.append("## Recent Exposure Changes")
    lines.append("")
    if trades.empty:
        lines.append("No exposure changes available.")
        lines.append("")
        return
    recent = trades.sort_values("date").tail(20)
    lines.append("| Date | Strategy | From | To | Target price |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in recent.to_dict("records"):
        lines.append(
            "| {date} | {label} | {from_exp} | {to_exp} | {price} |".format(
                date=str(pd.Timestamp(row["date"]).date()),
                label=row["strategy_label"],
                from_exp=format_pct(row["from_exposure"]),
                to_exp=format_pct(row["to_exposure"]),
                price=format_num(row["target_price"]),
            )
        )
    lines.append("")


def _latest_date(panel: pd.DataFrame) -> str:
    if panel.empty:
        return "n/a"
    return str(pd.Timestamp(panel.index[-1]).date())


def _write_lines(lines: list[str], output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
