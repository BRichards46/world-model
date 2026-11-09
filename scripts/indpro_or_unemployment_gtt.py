#!/usr/bin/env python3
"""Test INDPRO and unemployment-trend GTT gates on the SPY/SHY sleeve."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESEARCH_SCRIPT = Path("scripts/indpro_gtt_defensive_diversifier.py")
DEFAULT_MACRO = Path("outputs/gtt_risk_validation/gtt_macro_signals.parquet")
DEFAULT_OUTPUT = Path("outputs/indpro_or_unemployment_gtt")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attach_macro_states(signals: pd.DataFrame, macro_path: Path) -> pd.DataFrame:
    columns = [
        "industrial_production_yoy",
        "industrial_production_growth",
        "unemployment_rate",
        "unemployment_12m_average",
        "unemployment_trend",
    ]
    macro = pd.read_parquet(macro_path, columns=columns).sort_index()
    aligned = macro.reindex(signals.index, method="ffill")
    if aligned.isna().any().any():
        raise ValueError("A required point-in-time macro state is missing on a signal date")
    out = signals.copy()
    out[columns] = aligned[columns]
    out["price_downtrend"] = ~out["risk_on"].astype(bool)
    out["indpro_danger"] = out["industrial_production_growth"].astype(bool)
    out["unemployment_danger"] = out["unemployment_trend"].astype(bool)
    out["indpro_or_unemployment_danger"] = (
        out["indpro_danger"] | out["unemployment_danger"]
    )
    return out


def variant_signals(signals: pd.DataFrame, macro_danger: str | None) -> pd.DataFrame:
    out = signals.copy()
    defensive = out["price_downtrend"]
    if macro_danger is not None:
        defensive = defensive & out[macro_danger].astype(bool)
    out["risk_on"] = ~defensive
    out["first_asset"] = np.where(out["risk_on"], "SPY", "SHY")
    return out


def run_variants(base, signals: pd.DataFrame, open_px: pd.DataFrame,
                 cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    definitions = {
        "base": None,
        "indpro_gtt": "indpro_danger",
        "unemployment_gtt": "unemployment_danger",
        "indpro_or_unemployment_gtt": "indpro_or_unemployment_danger",
    }
    frames = {name: variant_signals(signals, danger) for name, danger in definitions.items()}
    returns = {}
    turnover = {}
    for name, frame in frames.items():
        returns[name], turnover[name] = base.run_variant(frame, open_px, cost_bps=cost_bps)
    common_index = open_px.dropna(subset=base.SYMBOLS).index
    entries = pd.DatetimeIndex([base.next_index_date(common_index, date) for date in signals.index])
    benchmarks = base.benchmark_returns(open_px, entries)
    return (
        pd.concat([*returns.values(), benchmarks["SHY"]], axis=1, keys=[*returns, "SHY"], join="inner"),
        pd.concat(turnover, axis=1),
        frames,
    )


def summarize(base, returns: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name in ["base", "indpro_gtt", "unemployment_gtt", "indpro_or_unemployment_gtt"]:
        rows[name] = base.metrics(returns[name], returns["SHY"])
        rows[name]["avg_annual_turnover"] = turnover[name].mean() * 12
    return pd.DataFrame(rows).T


def subperiods(base, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, start, end in [
        ("2008-2016", "2008", "2016"),
        ("2017-2026", "2017", "2026"),
    ]:
        sample = returns.loc[start:end]
        for name in ["base", "indpro_gtt", "unemployment_gtt", "indpro_or_unemployment_gtt"]:
            row = base.metrics(sample[name], sample["SHY"])
            rows.append({"period": period, "variant": name, **row})
    return pd.DataFrame(rows).set_index(["period", "variant"])


def signal_counts(signals: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        defensive = ~frame["risk_on"].astype(bool)
        rows.append(
            {
                "variant": name,
                "risk_off_signals": int(defensive.sum()),
                "risk_on_share": float(frame["risk_on"].mean()),
            }
        )
    result = pd.DataFrame(rows).set_index("variant")
    result["indpro_danger_month_ends"] = int(signals["indpro_danger"].sum())
    result["unemployment_danger_month_ends"] = int(signals["unemployment_danger"].sum())
    result["either_danger_month_ends"] = int(signals["indpro_or_unemployment_danger"].sum())
    return result


def drawdown_details(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["base", "indpro_gtt", "unemployment_gtt", "indpro_or_unemployment_gtt"]:
        wealth = (1.0 + returns[name]).cumprod()
        dd = wealth / wealth.cummax() - 1.0
        trough = dd.idxmin()
        peak = wealth.loc[:trough].idxmax()
        recovered = wealth.loc[trough:][wealth.loc[trough:] >= wealth.loc[peak]]
        recovery = recovered.index[0] if not recovered.empty else pd.NaT
        rows.append(
            {
                "variant": name,
                "max_drawdown": dd.loc[trough],
                "peak": peak.date().isoformat(),
                "trough": trough.date().isoformat(),
                "recovery": recovery.date().isoformat() if pd.notna(recovery) else "unrecovered",
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def annual_returns(returns: pd.DataFrame) -> pd.DataFrame:
    keep = returns.drop(columns="SHY")
    return keep.groupby(keep.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)


def fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{100.0 * value:.2f}%"


def table(frame: pd.DataFrame, pct: set[str]) -> str:
    out = frame.copy()
    for col in out.columns:
        if col in pct:
            out[col] = out[col].map(fmt_pct)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
    return out.to_html(classes="data", border=0, escape=True)


def write_report(path: Path, summary: pd.DataFrame, periods: pd.DataFrame,
                 counts: pd.DataFrame, drawdowns: pd.DataFrame,
                 diagnostics: dict[str, object]) -> None:
    display_names = {
        "base": "Base trend rule",
        "indpro_gtt": "INDPRO GTT",
        "unemployment_gtt": "Unemployment-trend GTT",
        "indpro_or_unemployment_gtt": "INDPRO or unemployment GTT",
    }
    cols = [
        "cagr", "ann_vol", "sharpe_excess_shy", "max_drawdown", "calmar",
        "terminal_1_dollar", "avg_annual_turnover",
    ]
    pct = {"cagr", "ann_vol", "max_drawdown", "risk_on_share"}
    summary_view = summary[cols].rename(index=display_names)
    period_view = periods.reset_index()[
        ["period", "variant", "cagr", "ann_vol", "sharpe_excess_shy", "max_drawdown", "calmar"]
    ]
    period_view["variant"] = period_view["variant"].map(display_names)
    counts_view = counts[["risk_off_signals", "risk_on_share"]].rename(index=display_names)
    drawdown_view = drawdowns.rename(index=display_names)
    or_row = summary.loc["indpro_or_unemployment_gtt"]
    ip_row = summary.loc["indpro_gtt"]
    if or_row["max_drawdown"] > ip_row["max_drawdown"] and or_row["cagr"] > ip_row["cagr"]:
        verdict = "The OR gate improved both return and downside protection versus INDPRO alone in this sample."
    elif or_row["max_drawdown"] > ip_row["max_drawdown"] and or_row["cagr"] >= ip_row["cagr"] - 0.002:
        verdict = "The OR gate improved downside protection without materially sacrificing the INDPRO-only return gain."
    elif or_row["max_drawdown"] > ip_row["max_drawdown"]:
        verdict = "The OR gate improved downside protection, but gave back part of the INDPRO-only return gain."
    else:
        verdict = "The OR gate did not improve the INDPRO-only return/drawdown trade-off."
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>INDPRO or Unemployment GTT Gate</title><style>
body{{font-family:Inter,ui-sans-serif,system-ui;color:#18222d;margin:0;background:#edf2f6}}main{{max-width:1120px;margin:28px auto;background:white;padding:40px 48px 54px;box-shadow:0 10px 30px #1b2b3b18}}h1{{font-size:32px;margin:0 0 8px}}h2{{margin-top:34px;border-top:1px solid #d7dde5;padding-top:25px}}.sub{{color:#607083;margin-bottom:24px}}.call{{background:#e8f4ed;border-left:5px solid #2f7d4a;padding:18px 20px;font-size:18px;line-height:1.5}}table.data{{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 24px}}table.data th,table.data td{{padding:8px 9px;border-bottom:1px solid #d7dde5;text-align:right}}table.data th:first-child,table.data td:first-child{{text-align:left}}.note{{color:#607083;font-size:13px;line-height:1.5}}li{{line-height:1.55}}</style></head><body><main>
<h1>INDPRO or Unemployment GTT Gate</h1><div class="sub">Controlled SPY-sleeve comparison · July 2007–October 2026 · 10 bps one-way costs</div>
<div class="call"><strong>Bottom line:</strong> {html.escape(verdict)}</div>
<h2>Employment metric selection</h2><p>The prior SPY overlay research identified <strong>unemployment versus its trailing 12-month average</strong> as the strongest employment filter, not PAYEMS. Macro danger is true when the latest unemployment rate is above that point-in-time trailing average.</p>
<h2>Full-period results</h2>{table(summary_view, pct)}
<h2>Signal frequency</h2>{table(counts_view, pct)}
<p class="note">Across 233 signal dates, INDPRO danger was active at 60 month-ends, unemployment danger at 78, and either danger at 101. A defensive allocation still requires a simultaneous negative SPY price trend.</p>
<h2>Chronological comparison</h2>{table(period_view, pct)}
<h2>Maximum-drawdown episodes</h2>{table(drawdown_view, pct)}
<h2>Latest signal</h2><ul><li>Signal date: <strong>{diagnostics['latest_signal_date']}</strong></li><li>INDPRO YoY: <strong>{diagnostics['latest_indpro_yoy']:.2f}%</strong>; danger: <strong>{diagnostics['latest_indpro_danger']}</strong></li><li>Unemployment: <strong>{diagnostics['latest_unemployment_rate']:.2f}%</strong> vs 12-month average <strong>{diagnostics['latest_unemployment_average']:.2f}%</strong>; danger: <strong>{diagnostics['latest_unemployment_danger']}</strong></li><li>OR-gate first sleeve: <strong>{diagnostics['latest_first_asset']}</strong>; diversifier: <strong>{diagnostics['latest_second_asset']}</strong></li></ul>
<h2>Research caveat</h2><p><strong>This is not an independent holdout test.</strong> Both INDPRO and the unemployment-trend signal were selected after examining related historical GTT evidence, and the portfolio sample overlaps that research. The OR rule is an additional post-research specification. Treat the result as candidate generation; freeze the exact rule before prospective or independent-market validation.</p>
<p class="note">Point-in-time FRED release vintages are joined as-of each signal date. The price rule, diversifier selector, inverse-volatility weights, next-open execution, and costs are unchanged.</p>
</main></body></html>"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-script", type=Path, default=DEFAULT_RESEARCH_SCRIPT)
    parser.add_argument("--macro-signals", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    research = load_module("indpro_research", args.research_script)
    base = research.load_base_module(research.DEFAULT_BASE_SCRIPT)
    open_px, close_px = base.load_norgate(research.DEFAULT_DATA_ROOT)
    signals = attach_macro_states(base.build_signals(close_px, vol_window=63), args.macro_signals)
    returns, turnover, frames = run_variants(base, signals, open_px, args.cost_bps)
    summary = summarize(base, returns, turnover)
    periods = subperiods(base, returns)
    counts = signal_counts(signals, frames)
    drawdowns = drawdown_details(returns)
    annual = annual_returns(returns)
    latest = signals.iloc[-1]
    diagnostics = {
        "latest_signal_date": signals.index[-1].date().isoformat(),
        "latest_indpro_yoy": float(latest["industrial_production_yoy"]),
        "latest_indpro_danger": bool(latest["indpro_danger"]),
        "latest_unemployment_rate": float(latest["unemployment_rate"]),
        "latest_unemployment_average": float(latest["unemployment_12m_average"]),
        "latest_unemployment_danger": bool(latest["unemployment_danger"]),
        "latest_first_asset": str(frames["indpro_or_unemployment_gtt"].iloc[-1]["first_asset"]),
        "latest_second_asset": str(latest["second_asset"]),
    }

    returns.to_csv(args.output_dir / "monthly_returns.csv", index_label="holding_month")
    summary.to_csv(args.output_dir / "summary.csv", index_label="variant")
    periods.to_csv(args.output_dir / "subperiods.csv")
    counts.to_csv(args.output_dir / "signal_counts.csv")
    drawdowns.to_csv(args.output_dir / "drawdown_details.csv")
    annual.to_csv(args.output_dir / "annual_returns.csv", index_label="year")
    signals.to_csv(args.output_dir / "signals.csv", index_label="signal_date")
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        args.output_dir / "indpro_or_unemployment_gtt.html",
        summary,
        periods,
        counts,
        drawdowns,
        diagnostics,
    )
    print(summary.to_string())
    print("\nSignal counts:\n", counts.to_string())
    print("\nLatest:\n", json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
