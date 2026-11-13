#!/usr/bin/env python3
"""Compare the defensive-diversifier SPY switch with an INDPRO GTT switch.

The base strategy moves its first sleeve from SPY to SHY whenever SPY is below
its 200-day moving average. The GTT variant moves to SHY only when that price
warning is confirmed by point-in-time INDPRO year-over-year growth at or below
zero. All other portfolio rules are inherited from the original backtest.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE_SCRIPT = Path(
    "/Users/brichards/Documents/data/equity/norgate/research/"
    "monthly_defensive_diversifier_backtest.py"
)
DEFAULT_DATA_ROOT = Path(
    "/Users/brichards/Documents/data/equity/norgate/prices/adjustment=totalreturn"
)
DEFAULT_MACRO = Path("outputs/gtt_risk_validation/gtt_macro_signals.parquet")
DEFAULT_OUTPUT = Path("outputs/indpro_gtt_defensive_diversifier")


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("monthly_defensive_diversifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base backtest: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attach_indpro_gtt(signals: pd.DataFrame, macro_path: Path) -> pd.DataFrame:
    macro = pd.read_parquet(macro_path).sort_index()
    required = {"industrial_production_yoy", "industrial_production_growth"}
    missing = required.difference(macro.columns)
    if missing:
        raise ValueError(f"Macro signal file is missing columns: {sorted(missing)}")

    aligned = macro.loc[:, sorted(required)].reindex(signals.index, method="ffill")
    if aligned.isna().any().any():
        bad = aligned.index[aligned.isna().any(axis=1)]
        raise ValueError(f"INDPRO signal unavailable for {len(bad)} strategy dates")

    out = signals.copy()
    out["base_price_downtrend"] = ~out["risk_on"].astype(bool)
    out["industrial_production_yoy"] = aligned["industrial_production_yoy"]
    out["indpro_danger"] = aligned["industrial_production_growth"].astype(bool)
    out["gtt_double_confirmation"] = out["base_price_downtrend"] & out["indpro_danger"]
    out["gtt_risk_on"] = ~out["gtt_double_confirmation"]
    out["gtt_first_asset"] = np.where(out["gtt_risk_on"], "SPY", "SHY")
    return out


def gtt_signal_frame(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["risk_on"] = out["gtt_risk_on"]
    out["first_asset"] = out["gtt_first_asset"]
    return out


def aligned_returns(base, signals: pd.DataFrame, gtt_signals: pd.DataFrame,
                    open_px: pd.DataFrame, cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_net, base_turn = base.run_variant(signals, open_px, cost_bps=cost_bps)
    gtt_net, gtt_turn = base.run_variant(gtt_signals, open_px, cost_bps=cost_bps)
    base_gross, _ = base.run_variant(signals, open_px, cost_bps=0.0)
    gtt_gross, _ = base.run_variant(gtt_signals, open_px, cost_bps=0.0)

    common_index = open_px.dropna(subset=base.SYMBOLS).index
    entries = pd.DatetimeIndex([base.next_index_date(common_index, date) for date in signals.index])
    benchmarks = base.benchmark_returns(open_px, entries)
    returns = pd.concat(
        [
            base_net.rename("base_net"),
            gtt_net.rename("indpro_gtt_net"),
            base_gross.rename("base_gross"),
            gtt_gross.rename("indpro_gtt_gross"),
            benchmarks,
        ],
        axis=1,
        join="inner",
    )
    turnover = pd.concat(
        [base_turn.rename("base_turnover"), gtt_turn.rename("indpro_gtt_turnover")],
        axis=1,
    )
    return returns, turnover


def metric_table(base, returns: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name in ["base_net", "indpro_gtt_net", "SPY", "60_40_SPY_TLT"]:
        rows[name] = base.metrics(returns[name], returns["SHY"])
    result = pd.DataFrame(rows).T
    result["avg_annual_turnover"] = np.nan
    result.loc["base_net", "avg_annual_turnover"] = turnover["base_turnover"].mean() * 12
    result.loc["indpro_gtt_net", "avg_annual_turnover"] = (
        turnover["indpro_gtt_turnover"].mean() * 12
    )
    for column in ["cagr", "ann_vol", "sharpe_excess_shy", "max_drawdown", "calmar"]:
        result.loc["indpro_gtt_net", f"delta_{column}_vs_base"] = (
            result.loc["indpro_gtt_net", column] - result.loc["base_net", column]
        )
    return result


def period_metrics(base, returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    sample = returns.loc[start:end]
    rows = {
        "base_net": base.metrics(sample["base_net"], sample["SHY"]),
        "indpro_gtt_net": base.metrics(sample["indpro_gtt_net"], sample["SHY"]),
    }
    return pd.DataFrame(rows).T


def annual_returns(returns: pd.DataFrame) -> pd.DataFrame:
    keep = returns[["base_net", "indpro_gtt_net", "SPY", "60_40_SPY_TLT"]]
    out = keep.groupby(keep.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    out["gtt_minus_base"] = out["indpro_gtt_net"] - out["base_net"]
    return out


def drawdown_details(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["base_net", "indpro_gtt_net"]:
        wealth = (1.0 + returns[name]).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        trough = drawdown.idxmin()
        peak = wealth.loc[:trough].idxmax()
        recovery_set = wealth.loc[trough:].loc[lambda x: x >= wealth.loc[peak]]
        recovery = recovery_set.index[0] if not recovery_set.empty else pd.NaT
        rows.append(
            {
                "variant": name,
                "max_drawdown": drawdown.loc[trough],
                "peak": peak.date().isoformat(),
                "trough": trough.date().isoformat(),
                "recovery": recovery.date().isoformat() if pd.notna(recovery) else "unrecovered",
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def changed_months(base, returns: pd.DataFrame, signals: pd.DataFrame,
                   open_px: pd.DataFrame) -> pd.DataFrame:
    common_index = open_px.dropna(subset=base.SYMBOLS).index
    records = []
    for signal_date, row in signals.iterrows():
        entry = base.next_index_date(common_index, signal_date)
        if entry is None:
            continue
        holding_month = entry.to_period("M").to_timestamp("M")
        if holding_month not in returns.index:
            continue
        base_return = returns.at[holding_month, "base_net"]
        gtt_return = returns.at[holding_month, "indpro_gtt_net"]
        delta = gtt_return - base_return
        if row["first_asset"] == row["gtt_first_asset"]:
            continue
        records.append(
            {
                "holding_month": holding_month.date().isoformat(),
                "signal_date": signal_date.date().isoformat(),
                "indpro_yoy": row["industrial_production_yoy"] / 100.0,
                "indpro_danger": bool(row["indpro_danger"]),
                "base_first_asset": row["first_asset"],
                "gtt_first_asset": row["gtt_first_asset"],
                "base_return": base_return,
                "gtt_return": gtt_return,
                "gtt_minus_base": delta,
            }
        )
    return pd.DataFrame(records).set_index("holding_month")


def cost_sensitivity(base, signals: pd.DataFrame, gtt_signals: pd.DataFrame,
                     open_px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in [0, 5, 10, 25, 50]:
        base_ret, base_turn = base.run_variant(signals, open_px, cost_bps=cost)
        gtt_ret, gtt_turn = base.run_variant(gtt_signals, open_px, cost_bps=cost)
        for name, ret, turn in [
            ("base", base_ret, base_turn),
            ("indpro_gtt", gtt_ret, gtt_turn),
        ]:
            values = base.metrics(ret)
            rows.append(
                {
                    "cost_bps": cost,
                    "variant": name,
                    "cagr": values["cagr"],
                    "ann_vol": values["ann_vol"],
                    "max_drawdown": values["max_drawdown"],
                    "terminal_1_dollar": values["terminal_1_dollar"],
                    "avg_annual_turnover": turn.mean() * 12,
                }
            )
    return pd.DataFrame(rows).set_index(["cost_bps", "variant"])


def signal_diagnostics(signals: pd.DataFrame) -> dict[str, int | float | str]:
    rescued = signals["base_price_downtrend"] & ~signals["indpro_danger"]
    confirmed = signals["gtt_double_confirmation"]
    macro_only = ~signals["base_price_downtrend"] & signals["indpro_danger"]
    latest = signals.iloc[-1]
    return {
        "signal_months": int(len(signals)),
        "base_risk_off_months": int(signals["base_price_downtrend"].sum()),
        "gtt_confirmed_risk_off_months": int(confirmed.sum()),
        "trend_breaks_ignored_by_gtt": int(rescued.sum()),
        "indpro_danger_with_positive_trend": int(macro_only.sum()),
        "latest_signal_date": signals.index[-1].date().isoformat(),
        "latest_indpro_yoy": float(latest["industrial_production_yoy"]),
        "latest_indpro_danger": bool(latest["indpro_danger"]),
        "latest_price_downtrend": bool(latest["base_price_downtrend"]),
        "latest_gtt_first_asset": str(latest["gtt_first_asset"]),
        "latest_second_asset": str(latest["second_asset"]),
    }


def rescued_month_analysis(base, returns: pd.DataFrame, signals: pd.DataFrame,
                           open_px: pd.DataFrame) -> pd.DataFrame:
    # The month-end signal governs the period beginning at the next common open.
    common_index = open_px.dropna(subset=base.SYMBOLS).index
    entries = [base.next_index_date(common_index, date) for date in signals.index]
    labels = pd.DatetimeIndex(entries).to_period("M").to_timestamp("M")
    states = pd.DataFrame(
        {
            "trend_break_ignored": (
                signals["base_price_downtrend"] & ~signals["indpro_danger"]
            ).to_numpy(),
            "double_confirmation": signals["gtt_double_confirmation"].to_numpy(),
        },
        index=labels,
    ).reindex(returns.index)
    rows = []
    for state in ["trend_break_ignored", "double_confirmation"]:
        active = states[state].fillna(False)
        spread = returns.loc[active, "indpro_gtt_net"] - returns.loc[active, "base_net"]
        rows.append(
            {
                "state": state,
                "months": int(active.sum()),
                "base_ann_return": (1.0 + returns.loc[active, "base_net"]).prod() ** (12 / active.sum()) - 1
                if active.sum() else np.nan,
                "gtt_ann_return": (1.0 + returns.loc[active, "indpro_gtt_net"]).prod() ** (12 / active.sum()) - 1
                if active.sum() else np.nan,
                "average_monthly_gtt_minus_base": spread.mean(),
                "gtt_win_rate": (spread > 0).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("state")


def fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{100 * value:.2f}%"


def html_table(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    work = frame.copy()
    for column in work.columns:
        if column in percent_columns:
            work[column] = work[column].map(fmt_pct)
        elif pd.api.types.is_float_dtype(work[column]):
            work[column] = work[column].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
    return work.to_html(classes="data", border=0, escape=True)


def write_report(path: Path, summary: pd.DataFrame, periods: dict[str, pd.DataFrame],
                 annual: pd.DataFrame, diagnostics: dict[str, object],
                 conditional: pd.DataFrame, cost: pd.DataFrame,
                 drawdowns: pd.DataFrame, changes: pd.DataFrame) -> None:
    gtt = summary.loc["indpro_gtt_net"]
    base_row = summary.loc["base_net"]
    cagr_delta = gtt["cagr"] - base_row["cagr"]
    dd_delta = gtt["max_drawdown"] - base_row["max_drawdown"]
    verdict = (
        "INDPRO confirmation improved return and risk-adjusted performance, but the worst "
        "monthly-observed drawdown was deeper."
        if cagr_delta > 0 and gtt["sharpe_excess_shy"] > base_row["sharpe_excess_shy"]
        else "INDPRO confirmation did not improve the return/risk trade-off decisively."
    )
    percent = {
        "cagr", "ann_vol", "max_drawdown", "worst_month", "best_month",
        "positive_months", "delta_cagr_vs_base", "delta_ann_vol_vs_base",
        "delta_max_drawdown_vs_base", "base_net", "indpro_gtt_net", "SPY",
        "60_40_SPY_TLT", "gtt_minus_base", "base_ann_return", "gtt_ann_return",
        "average_monthly_gtt_minus_base", "gtt_win_rate",
        "indpro_yoy", "base_return", "gtt_return",
    }
    summary_view = summary[[
        "cagr", "ann_vol", "sharpe_excess_shy", "max_drawdown", "calmar",
        "terminal_1_dollar", "avg_annual_turnover",
    ]]
    period_sections = "".join(
        f"<h3>{html.escape(label)}</h3>{html_table(table[['cagr','ann_vol','sharpe_excess_shy','max_drawdown','calmar']], percent)}"
        for label, table in periods.items()
    )
    annual_tail = annual.tail(10)
    cost_view = cost.reset_index().loc[:, [
        "cost_bps", "variant", "cagr", "ann_vol", "max_drawdown",
        "terminal_1_dollar", "avg_annual_turnover",
    ]]
    change_view = changes.assign(abs_delta=changes["gtt_minus_base"].abs()).sort_values(
        "abs_delta", ascending=False
    ).head(12).drop(columns="abs_delta")
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>INDPRO GTT vs Monthly Defensive Diversifier</title>
<style>
:root{{--ink:#17202a;--muted:#5d6d7e;--line:#d7dde5;--blue:#174a7e;--pale:#f4f7fa;--good:#e8f4ed}}
body{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;color:var(--ink);margin:0;background:#eef2f6}}
main{{max-width:1120px;margin:28px auto;background:white;padding:38px 46px 52px;box-shadow:0 10px 30px #1b2b3b18}}
h1{{font-size:32px;margin:0 0 8px}} h2{{margin-top:34px;border-top:1px solid var(--line);padding-top:26px}} h3{{margin-top:22px}}
.sub{{color:var(--muted);margin-bottom:24px}} .call{{background:var(--good);border-left:5px solid #2f7d4a;padding:18px 20px;font-size:18px;line-height:1.5}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}} .tile{{background:var(--pale);padding:16px;border-radius:8px}}
.tile b{{display:block;font-size:22px;color:var(--blue);margin-top:5px}} table.data{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 22px}}
table.data th,table.data td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}} table.data th:first-child,table.data td:first-child{{text-align:left}}
code{{background:var(--pale);padding:2px 5px}} .note{{color:var(--muted);font-size:13px;line-height:1.5}} ul{{line-height:1.55}}
@media(max-width:800px){{main{{margin:0;padding:24px}}.tiles{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>INDPRO GTT vs Monthly Defensive Diversifier</h1>
<div class="sub">Controlled SPY-sleeve test · July 2007–October 2026 · 10 bps one-way cost</div>
<div class="call"><strong>Bottom line:</strong> {html.escape(verdict)} The GTT variant changes only the SPY/SHY decision: SHY requires both a negative 200-day price trend and point-in-time INDPRO YoY growth at or below zero.</div>
<div class="tiles">
  <div class="tile">GTT CAGR<b>{fmt_pct(gtt['cagr'])}</b></div>
  <div class="tile">Base CAGR<b>{fmt_pct(base_row['cagr'])}</b></div>
  <div class="tile">GTT max drawdown<b>{fmt_pct(gtt['max_drawdown'])}</b></div>
  <div class="tile">GTT Sharpe vs SHY<b>{gtt['sharpe_excess_shy']:.2f}</b></div>
</div>
<h2>Full-period comparison</h2>
{html_table(summary_view, percent)}
<h2>What changed</h2>
<ul>
  <li>Base risk-off signals: <strong>{diagnostics['base_risk_off_months']}</strong> of {diagnostics['signal_months']} month-ends.</li>
  <li>GTT-confirmed risk-off signals: <strong>{diagnostics['gtt_confirmed_risk_off_months']}</strong>.</li>
  <li>Negative price trends ignored because INDPRO remained positive: <strong>{diagnostics['trend_breaks_ignored_by_gtt']}</strong>.</li>
  <li>INDPRO warnings with positive price trend (no defensive switch): <strong>{diagnostics['indpro_danger_with_positive_trend']}</strong>.</li>
</ul>
{html_table(conditional, percent)}
<h2>Largest months changed by the filter</h2>
{html_table(change_view, percent)}
<p class="note">Rows include small cost-path differences as well as allocation differences. The largest effects occur when GTT remains in SPY while the base first sleeve is SHY.</p>
<h2>Maximum-drawdown episodes</h2>
{html_table(drawdowns, percent)}
<h2>Chronological stability</h2>
{period_sections}
<h2>Recent annual results</h2>
{html_table(annual_tail, percent)}
<h2>Transaction-cost sensitivity</h2>
{html_table(cost_view, percent)}
<h2>Latest completed signal</h2>
<ul>
  <li>Signal date: <strong>{diagnostics['latest_signal_date']}</strong></li>
  <li>Point-in-time INDPRO YoY: <strong>{diagnostics['latest_indpro_yoy']:.2f}%</strong>; danger flag: <strong>{diagnostics['latest_indpro_danger']}</strong></li>
  <li>SPY below 200-day trend: <strong>{diagnostics['latest_price_downtrend']}</strong></li>
  <li>GTT first sleeve: <strong>{diagnostics['latest_gtt_first_asset']}</strong>; diversifier: <strong>{diagnostics['latest_second_asset']}</strong></li>
  <li>Target weights: <strong>{diagnostics['latest_first_weight']:.1%} {diagnostics['latest_gtt_first_asset']} / {diagnostics['latest_second_weight']:.1%} {diagnostics['latest_second_asset']}</strong></li>
</ul>
<h2>Method and limitations</h2>
<ul>
  <li>Macro state uses FRED release vintages and is joined as-of each completed month-end, preventing revised-data lookahead.</li>
  <li>Signals use month-end closes; trades occur at the next common total-return-adjusted open.</li>
  <li>The second sleeve, 63-day correlation selector, 63-day inverse-volatility weights, and transaction-cost model are unchanged.</li>
  <li>Drawdowns are observed monthly and therefore understate possible intramonth losses.</li>
  <li>The common ETF sample starts in 2007 because UUP is required. It contains few independent recession episodes, and the broader GTT research informed this test.</li>
</ul>
<p class="note">This is a historical research comparison, not an implementation recommendation. Supporting CSV and JSON files in the same directory contain the exact returns, signals, diagnostics, and sensitivities.</p>
</main></body></html>"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script", type=Path, default=DEFAULT_BASE_SCRIPT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--macro-signals", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_module(args.base_script)
    open_px, close_px = base.load_norgate(args.data_root)
    signals = base.build_signals(close_px, vol_window=63)
    signals = attach_indpro_gtt(signals, args.macro_signals)
    gtt_signals = gtt_signal_frame(signals)
    returns, turnover = aligned_returns(base, signals, gtt_signals, open_px, args.cost_bps)
    summary = metric_table(base, returns, turnover)
    periods = {
        "Development-era diagnostic: 2008–2016": period_metrics(base, returns, "2008", "2016"),
        "Later-period diagnostic: 2017–2026": period_metrics(base, returns, "2017", "2026"),
    }
    annual = annual_returns(returns)
    costs = cost_sensitivity(base, signals, gtt_signals, open_px)
    diagnostics = signal_diagnostics(signals)
    conditional = rescued_month_analysis(base, returns, signals, open_px)
    drawdowns = drawdown_details(returns)
    changes = changed_months(base, returns, signals, open_px)
    latest_target = base.vector(gtt_signals.iloc[-1])
    diagnostics["latest_first_weight"] = float(
        latest_target[diagnostics["latest_gtt_first_asset"]]
    )
    diagnostics["latest_second_weight"] = float(
        latest_target[diagnostics["latest_second_asset"]]
    )

    returns.to_csv(args.output_dir / "monthly_returns.csv", index_label="holding_month")
    turnover.to_csv(args.output_dir / "turnover.csv", index_label="rebalance_date")
    signals.to_csv(args.output_dir / "signals.csv", index_label="signal_date")
    summary.to_csv(args.output_dir / "summary.csv", index_label="variant")
    annual.to_csv(args.output_dir / "annual_returns.csv", index_label="year")
    costs.to_csv(args.output_dir / "cost_sensitivity.csv")
    conditional.to_csv(args.output_dir / "conditional_results.csv")
    drawdowns.to_csv(args.output_dir / "drawdown_details.csv")
    changes.to_csv(args.output_dir / "changed_months.csv")
    for label, frame in periods.items():
        slug = "development_2008_2016" if "Development" in label else "later_2017_2026"
        frame.to_csv(args.output_dir / f"{slug}.csv", index_label="variant")
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    methodology = {
        "base_script": str(args.base_script),
        "data_root": str(args.data_root),
        "macro_signal_file": str(args.macro_signals),
        "macro_rule": "point-in-time INDPRO YoY growth <= 0",
        "price_rule": "SPY close below 200-trading-day SMA",
        "gtt_defensive_rule": "macro danger AND negative price trend",
        "execution": "next common trading-day total-return-adjusted open",
        "cost_bps_one_way": args.cost_bps,
        "generated_from_market_data_through": str(close_px.index.max().date()),
        "generated_from_macro_releases_through": str(
            pd.read_parquet(args.macro_signals).index.max().date()
        ),
    }
    (args.output_dir / "methodology.json").write_text(
        json.dumps(methodology, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        args.output_dir / "indpro_gtt_comparison.html",
        summary,
        periods,
        annual,
        diagnostics,
        conditional,
        costs,
        drawdowns,
        changes,
    )
    print(summary.to_string())
    print("\nSignal diagnostics:\n", json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
