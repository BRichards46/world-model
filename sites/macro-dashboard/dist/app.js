const COMPONENTS = {
  all: { label: "All series", color: "var(--accent)" },
  activity_labor: { label: "Activity & labor", color: "var(--activity)" },
  inflation: { label: "Inflation", color: "var(--inflation)" },
  policy_financial_conditions: { label: "Policy & conditions", color: "var(--conditions)" },
  housing: { label: "Housing", color: "var(--housing)" },
};

const RECIPES = {
  three_month_growth: "3m/3m annualized log growth; velocity is its three-observation change.",
  three_month_average_change: "Latest three-observation average; velocity compares it with the preceding three observations.",
  quarterly_growth: "Quarter-over-quarter annualized log growth; velocity is the change from the prior quarter.",
  inflation_yoy_mom_change: "Year-over-year inflation is the level; velocity is the change in the latest seasonally adjusted month-over-month rate versus the prior month.",
  latest_value_change: "Latest published monthly rate; velocity is its change from the prior release.",
  real_policy_rate: "Three-month average effective fed funds rate less 12-month core PCE inflation; velocity is the three-month change in basis points.",
  daily_average_change: "20-business-day average; velocity is the change versus 63 observed business days earlier in basis points.",
  weekly_average_change: "Four-week average; velocity is the change versus 13 observations earlier.",
  weekly_average_change_bps: "Four-week average; velocity is the 13-observation change in basis points.",
  permits_yoy_growth: "Year-over-year growth of the three-month average; velocity is its three-observation change.",
  housing_starts_intensity: "Three-month average housing starts per 1,000 labor-force participants; velocity is its year-over-year percentage change.",
};

const MOMENTUM_LANGUAGE = {
  payroll_growth: { rising: "Accelerating", falling: "Decelerating" },
  unemployment_rate: { rising: "Rising", falling: "Falling" },
  real_consumer_spending: { rising: "Accelerating", falling: "Decelerating" },
  industrial_production: { rising: "Accelerating", falling: "Decelerating" },
  real_gdp_growth: { rising: "Accelerating", falling: "Decelerating" },
  core_pce_inflation: { rising: "Heating", falling: "Cooling" },
  core_cpi_inflation: { rising: "Heating", falling: "Cooling" },
  median_cpi: { rising: "Heating", falling: "Cooling" },
  five_year_inflation_expectations: { rising: "Heating", falling: "Cooling" },
  real_policy_rate: { rising: "Tightening", falling: "Easing" },
  treasury_curve: { rising: "Steepening", falling: "Flattening" },
  high_yield_spread: { rising: "Widening", falling: "Narrowing" },
  adjusted_financial_conditions: { rising: "Tightening", falling: "Easing" },
  building_permits: { rising: "Accelerating", falling: "Decelerating" },
  housing_starts_intensity: { rising: "Rising", falling: "Falling" },
  mortgage_rate: { rising: "Rising", falling: "Falling" },
};

const READING_LANGUAGE = {
  payroll_growth: { label: "Current growth pace", note: "3m/3m annualized" },
  unemployment_rate: { label: "Current unemployment rate", note: "3-month average" },
  real_consumer_spending: { label: "Current growth pace", note: "3m/3m annualized" },
  industrial_production: { label: "Current growth pace", note: "3m/3m annualized" },
  real_gdp_growth: { label: "Current growth rate", note: "QoQ annualized" },
  core_pce_inflation: { label: "Current inflation level", note: "year over year" },
  core_cpi_inflation: { label: "Current inflation level", note: "year over year" },
  median_cpi: { label: "Current monthly rate", note: "published annualized rate" },
  five_year_inflation_expectations: { label: "Current expected inflation", note: "3-month average" },
  real_policy_rate: { label: "Current real policy rate", note: "3-month average less core PCE YoY" },
  treasury_curve: { label: "Current curve spread", note: "20-business-day average" },
  high_yield_spread: { label: "Current credit spread", note: "20-business-day average" },
  adjusted_financial_conditions: { label: "Current index reading", note: "4-week average" },
  building_permits: { label: "Current growth pace", note: "3-month-average YoY" },
  housing_starts_intensity: { label: "Current starts intensity", note: "3-month average per 1,000 labor-force participants" },
  mortgage_rate: { label: "Current mortgage rate", note: "4-week average" },
};

const state = { rows: [], history: new Map(), riskOverlay: null, quad: null, quadAssets: null, releases: null, directionalMomentum: null, premiumSelling: null, component: "all", search: "", attentionOnly: false, view: "monitor", historyRange: "3y" };

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindStaticControls();
  try {
    const [snapshotResponse, historyResponse, overlayResponse, momentumResponse, premiumResponse] = await Promise.all([
      fetch("./data/snapshot.json", { cache: "no-store" }),
      fetch("./data/history.json", { cache: "no-store" }),
      fetch("./data/risk_overlay.json", { cache: "no-store" }),
      fetch("./data/directional_momentum.json", { cache: "no-store" }),
      fetch("./data/premium_selling.json", { cache: "no-store" }),
    ]);
    const optional = async (path) => {
      const response = await fetch(path, { cache: "no-store" }).catch(() => null);
      return response?.ok ? response.json() : null;
    };
    [state.quad, state.quadAssets, state.releases] = await Promise.all([
      optional("./data/growth_inflation_quad.json"),
      optional("./data/quad_asset_performance.json"),
      optional("./data/release_forecasts.json"),
    ]);
    if (!snapshotResponse.ok || !historyResponse.ok || !overlayResponse.ok || !momentumResponse.ok || !premiumResponse.ok) throw new Error("Dashboard data files are unavailable.");
    state.rows = await snapshotResponse.json();
    state.riskOverlay = await overlayResponse.json();
    state.directionalMomentum = await momentumResponse.json();
    state.premiumSelling = await premiumResponse.json();
    const history = await historyResponse.json();
    history.forEach((item) => {
      const collection = state.history.get(item.indicator_id) || [];
      collection.push(item);
      state.history.set(item.indicator_id, collection);
    });
    render();
  } catch (error) {
    document.querySelector(".monitor-panel").innerHTML = `<div class="error-state"><strong>Unable to load dashboard data.</strong><br>${escapeHtml(error.message)}</div>`;
  }
}

function bindStaticControls() {
  document.getElementById("series-search").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    renderDataViews();
  });
  document.getElementById("attention-only").addEventListener("change", (event) => {
    state.attentionOnly = event.target.checked;
    renderDataViews();
  });
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    updateViewChrome();
  }));
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("macro-theme", next);
  });
  const savedTheme = localStorage.getItem("macro-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  document.querySelectorAll("[data-close-drawer]").forEach((element) => element.addEventListener("click", closeDrawer));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });
}

function render() {
  renderDecision();
  renderRiskOverlay();
  renderQuadStrip();
  renderSummary();
  renderFilters();
  renderDataViews();
  updateViewChrome();
}

function updateViewChrome() {
  document.querySelectorAll("[data-view]").forEach((item) => item.classList.toggle("active", item.dataset.view === state.view));
  document.getElementById("monitor-view").hidden = state.view !== "monitor";
  document.getElementById("quads-view").hidden = state.view !== "quads";
  document.getElementById("momentum-view").hidden = state.view !== "momentum";
  document.getElementById("premium-view").hidden = state.view !== "premium";
  document.body.classList.toggle("momentum-active", state.view === "momentum");
  document.body.classList.toggle("premium-active", state.view === "premium");
  document.body.classList.toggle("quads-active", state.view === "quads");
  const copy = state.view === "quads"
    ? { eyebrow: "BASE EFFECTS", title: "Growth × inflation quads", description: "Realized rate-of-change regimes and the quarters ahead, with the hurdles each quarter must clear to accelerate." }
    : state.view === "premium"
    ? { eyebrow: "OPTIONS", title: "Option pricing", description: "Observed implied volatility against expected realized volatility, IV rank, and today's 16-delta strike distances. The strategy leans built from momentum analogs are untested on option P&L." }
    : state.view === "momentum"
    ? { eyebrow: "TREND", title: "Multi-horizon trend", description: "Volatility-adjusted direction over 1 to 12 months. Trend following is validated at 6–12 months; the 1- and 3-month columns are context." }
    : { eyebrow: "CONTEXT · CORE SERIES", title: "Level × velocity monitor", description: "Each series against its own history: level and momentum percentiles, recent path, and source freshness. Descriptive, not a signal." };
  setText("view-eyebrow", copy.eyebrow);
  setText("monitor-title", copy.title);
  setText("view-description", copy.description);
  document.querySelector(".legend").hidden = state.view !== "monitor";
}

function renderRiskOverlay() {
  const container = document.getElementById("risk-overlay");
  const overlay = state.riskOverlay;
  if (!overlay?.state_id) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  const inputs = Array.isArray(overlay.inputs) ? overlay.inputs : [];
  container.className = `risk-overlay state-${overlay.state_id}`;
  container.innerHTML = `<div class="risk-state">
    <div class="eyebrow">EQUITY RISK OVERLAY</div>
    <div class="risk-state-line"><span class="risk-state-dot"></span><h2>${escapeHtml(overlay.state_label)}</h2></div>
    <p>${escapeHtml(overlay.message)}</p>
    <small>${escapeHtml(overlay.method_note)}</small>
    ${(overlay.data_warnings || []).map((warning) => `<p class="risk-data-warning">${escapeHtml(warning)}</p>`).join("")}
  </div>
  <div class="risk-inputs">
    ${inputs.map(riskInputMarkup).join("")}
  </div>`;
}

function riskInputMarkup(input) {
  const value = numberOrNull(input.value);
  const display = value == null ? "n/a" : `${signed(value, 2)} ${input.unit || ""}`.trim();
  const role = input.role && input.role !== "primary" ? `<em>${escapeHtml(input.role)}</em>` : "";
  return `<article class="risk-input status-${escapeHtml(input.status)}">
    <div><span>${escapeHtml(input.label)}</span>${role}</div>
    <strong>${escapeHtml(display)}</strong>
    <small><i></i>${escapeHtml(input.status_label)} · ${escapeHtml(formatDate(input.observation_date))}</small>
  </article>`;
}

const QUAD_META = {
  1: { name: "Quad 1", short: "Growth ↑ · Inflation ↓" },
  2: { name: "Quad 2", short: "Growth ↑ · Inflation ↑" },
  3: { name: "Quad 3", short: "Growth ↓ · Inflation ↑" },
  4: { name: "Quad 4", short: "Growth ↓ · Inflation ↓" },
};

function quadRows() {
  return Array.isArray(state.quad?.quarters) ? state.quad.quarters : [];
}

function quadName(quad) { return QUAD_META[quad]?.name || "Unclassified"; }
function quadBadge(quad, size = "") { return `<span class="quad-badge q${escapeHtml(quad ?? "x")} ${size}">${quad ? `Q${escapeHtml(quad)}` : "—"}</span>`; }
function quadPct(value, decimals = 2) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : `${parsed.toFixed(decimals)}%`; }
function quadPp(value) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : `${signed(parsed, 2)} pp`; }
function quadArrow(value) { const parsed = numberOrNull(value); return parsed == null ? "·" : parsed > 0 ? "↑" : "↓"; }
function quadAgreement(row) {
  const probabilities = row.probabilities;
  if (probabilities && row.quad != null && probabilities[row.quad] != null) {
    return `${Math.round(probabilities[row.quad] * 100)}% probability`;
  }
  if (row.agreement == null || !row.scenario_total) return "";
  return `${Math.round(row.agreement * row.scenario_total)}/${row.scenario_total} scenarios`;
}
function quadQuarterShort(label) { const match = /^(\d{4})Q(\d)$/.exec(label || ""); return match ? `Q${match[2]}’${match[1].slice(2)}` : label; }

function renderQuadStrip() {
  const container = document.getElementById("quad-strip");
  const rows = quadRows();
  if (!rows.length) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  const realized = rows.filter((row) => row.status === "realized");
  const ahead = rows.filter((row) => row.status !== "realized");
  const lastRealized = realized[realized.length - 1];
  const current = ahead.find((row) => row.is_current_quarter) || ahead[0];
  const cells = [...realized.slice(-2), ...ahead];
  container.innerHTML = `<div class="quad-state">
      <div class="eyebrow">GROWTH × INFLATION QUAD</div>
      <div class="quad-state-line">${quadBadge(current?.quad, "large")}<h2>${current ? `${escapeHtml(current.quarter)} ${escapeHtml(current.status === "partial" ? "partly reported" : "projected")}` : "No projection"}</h2></div>
      <p>${current ? `${escapeHtml(QUAD_META[current.quad]?.short || "")} · ${escapeHtml(quadAgreement(current) || "base case")}.` : ""} Last realized: ${lastRealized ? `${escapeHtml(lastRealized.quarter)} ${escapeHtml(quadName(lastRealized.quad))}` : "n/a"}.</p>
      <button class="quad-link" type="button" data-open-quads>Open base-effect outlook →</button>
    </div>
    <div class="quad-path">${cells.map(quadCellMarkup).join("")}</div>`;
  container.querySelector("[data-open-quads]").addEventListener("click", () => {
    state.view = "quads";
    updateViewChrome();
    document.querySelector(".monitor-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function quadCellMarkup(row) {
  const status = row.is_current_quarter ? `current · ${row.status}` : row.status;
  const agreement = row.status !== "realized" ? `<small class="quad-agreement">${escapeHtml(quadAgreement(row))}</small>` : "";
  return `<article class="quad-cell ${escapeHtml(row.status)} ${row.is_current_quarter ? "current" : ""}">
    <div><span>${escapeHtml(row.quarter)}</span><em>${escapeHtml(status)}</em></div>
    ${quadBadge(row.quad)}
    <small>G ${quadPct(row.growth_yoy, 1)} <b class="dir">${quadArrow(row.growth_change)}</b> · I ${quadPct(row.inflation_yoy, 1)} <b class="dir">${quadArrow(row.inflation_change)}</b></small>
    ${agreement}
  </article>`;
}

function renderQuads() {
  const container = document.getElementById("quads-view");
  const payload = state.quad;
  const rows = quadRows();
  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">Quad data is unavailable. Run <code>python -m world_model.cli.growth_inflation_quad</code>.</div>`;
    return;
  }
  const assumptions = payload.assumptions || {};
  const ahead = rows.filter((row) => row.status !== "realized");
  const nowcasts = Object.entries(assumptions.nowcast_quarters || {}).map(([quarter, value]) => `${quarter} GDPNow ${quadPct(value, 1)}`).join(" · ");
  const interpolated = (payload.interpolated_cpi_months || []).join(", ");
  const method = payload.method || {};
  container.innerHTML = `<div class="quad-intro"><div><span class="eyebrow">RATE OF CHANGE, NOT LEVEL</span><h3>Where the year-over-year rates are heading</h3></div>
      <p>Growth is real GDP YoY; inflation is the quarterly average of headline CPI YoY. A quarter's quad depends on whether each rate accelerates or decelerates versus the prior quarter, and that depends heavily on the comparison period a year earlier.</p></div>
    <div class="quad-layout">
      <section class="quad-panel quad-map-panel"><header><span>Quad path</span><em>Δ growth YoY × Δ inflation YoY, pp per quarter</em></header>${quadMapMarkup(rows)}</section>
      <section class="quad-panel"><header><span>What each quarter has to clear</span><em>Base case vs hurdle</em></header>
        <div class="quad-hurdles">${ahead.map(quadHurdleMarkup).join("")}</div>
        <div class="quad-assumptions">
          <span><b>Growth base</b> ${escapeHtml(nowcasts || "no nowcast")} · trend ${quadPct(assumptions.growth_trend_saar, 1)} annualized (last ${escapeHtml(assumptions.growth_trend_quarters)} quarters) · range ±${escapeHtml(assumptions.growth_offset_saar)} pp</span>
          <span><b>Inflation base</b> ${quadPct(assumptions.inflation_base_mom)} per month (${quadPct(assumptions.inflation_base_annualized, 1)} annualized), the ${escapeHtml(assumptions.inflation_trend_months)}-month core CPI run-rate, plus gasoline pass-through for months with gasoline prices</span>
          <span><b>Probabilities</b> from this model's own past forecast errors at the same horizon (walk-forward), not from the scenario range</span>
          <span><b>Data through</b> GDP ${escapeHtml(payload.last_gdp_quarter)} · CPI ${escapeHtml(payload.last_cpi_month)}${interpolated ? ` · interpolated CPI month: ${escapeHtml(interpolated)} (not published)` : ""}</span>
        </div>
      </section>
    </div>
    ${quadTradeabilityMarkup(state.quadAssets)}
    <section class="quad-panel"><header><span>Quarter by quarter</span><em>Realized quarters use reported data; later quarters show base case and the low–high range</em></header>
      <div class="quad-table-wrap"><div class="quad-table-head"><span>Quarter</span><span>Quad</span><span>Real GDP YoY</span><span>Headline CPI YoY</span><span>Quad probabilities</span></div>
      ${rows.slice(Math.max(0, rows.findIndex((row) => row.status !== "realized") - 8)).reverse().map(quadTableRowMarkup).join("")}</div>
    </section>
    <section class="quad-panel"><header><span>CPI comparison months ahead</span><em>Month-over-month CPI from a year earlier that drops out of the YoY rate</em></header>${quadCompsMarkup(payload.cpi_comps || [], assumptions.inflation_base_mom)}</section>
    <section class="quad-panel"><header><span>Walk-forward check</span><em>${escapeHtml(payload.backtest?.start ? `${payload.backtest.start} → ${payload.backtest.end}, month-end projections` : "Not run")}</em></header>${quadBacktestMarkup(payload.backtest)}</section>
    <div class="quad-method"><strong>Method and limits</strong><span>${escapeHtml(method.growth || "")}</span><span>${escapeHtml(method.inflation || "")}</span><span>${escapeHtml(method.scenarios || "")}</span><span>${escapeHtml(method.hurdles || "")}</span><em>${escapeHtml(method.warning || "")}</em></div>`;
}

function quadHurdleMarkup(row) {
  const growthMargin = row.growth_hurdle_saar != null && row.growth_assumption_saar != null ? row.growth_assumption_saar - row.growth_hurdle_saar : null;
  const inflationBase = row.inflation_assumption_mom ?? state.quad?.assumptions?.inflation_base_mom;
  const inflationMargin = row.inflation_hurdle_mom != null && inflationBase != null ? inflationBase - row.inflation_hurdle_mom : null;
  const growthOdds = row.p_growth_up != null ? ` · P(↑) ${Math.round(row.p_growth_up * 100)}%` : "";
  const inflationOdds = row.p_inflation_up != null ? ` · P(↑) ${Math.round(row.p_inflation_up * 100)}%` : "";
  const growthLine = row.growth_hurdle_saar == null
    ? `<span class="hurdle-line"><b>Growth</b><em>reported</em><strong>${quadArrow(row.growth_change)} ${quadPp(row.growth_change)}</strong></span>`
    : `<span class="hurdle-line"><b>Growth</b><em>needs &gt; ${quadPct(row.growth_hurdle_saar, 1)} annualized; base ${quadPct(row.growth_assumption_saar, 1)} (${escapeHtml(row.growth_assumption_source || "")})</em><strong class="${growthMargin > 0 ? "up" : "down"}">${growthMargin > 0 ? "↑ clears" : "↓ misses"} by ${Math.abs(growthMargin).toFixed(1)} pp${growthOdds}</strong></span>`;
  const inflationLine = row.inflation_hurdle_mom == null
    ? `<span class="hurdle-line"><b>Inflation</b><em>reported</em><strong>${quadArrow(row.inflation_change)} ${quadPp(row.inflation_change)}</strong></span>`
    : `<span class="hurdle-line"><b>Inflation</b><em>needs &gt; ${quadPct(row.inflation_hurdle_mom)} per month (${quadPct(row.inflation_hurdle_annualized, 1)} annualized); base ${quadPct(inflationBase)}</em><strong class="${inflationMargin > 0 ? "up" : "down"}">${inflationMargin > 0 ? "↑ clears" : "↓ misses"} by ${Math.abs(inflationMargin).toFixed(2)} pp${inflationOdds}</strong></span>`;
  return `<article class="quad-hurdle"><header>${quadBadge(row.quad)}<strong>${escapeHtml(row.quarter)}</strong><em>${escapeHtml(quadAgreement(row))}</em></header>${growthLine}${inflationLine}</article>`;
}

function quadTableRowMarkup(row) {
  const probabilities = row.probabilities || null;
  const counts = row.scenario_counts || null;
  const grid = probabilities
    ? `<span class="quad-grid">${[1, 2, 3, 4].map((quad) => `<i class="q${quad}" style="flex:${probabilities[quad] || 0}" title="${quadName(quad)}: ${Math.round((probabilities[quad] || 0) * 100)}%"></i>`).join("")}</span><small>${[1, 2, 3, 4].filter((quad) => (probabilities[quad] || 0) >= 0.05).map((quad) => `Q${quad} ${Math.round(probabilities[quad] * 100)}%`).join(" · ")}</small>`
    : counts
    ? `<span class="quad-grid">${[1, 2, 3, 4].map((quad) => `<i class="q${quad}" style="flex:${counts[quad] || 0}" title="${quadName(quad)}: ${counts[quad] || 0} of 9"></i>`).join("")}</span><small>${[1, 2, 3, 4].filter((quad) => counts[quad]).map((quad) => `Q${quad} ${counts[quad]}`).join(" · ")}</small>`
    : `<small>realized</small>`;
  const growthRange = row.growth_yoy_range ? `<small>${quadPct(row.growth_yoy_range[0], 1)} – ${quadPct(row.growth_yoy_range[1], 1)}</small>` : "";
  const inflationRange = row.inflation_yoy_range ? `<small>${quadPct(row.inflation_yoy_range[0], 1)} – ${quadPct(row.inflation_yoy_range[1], 1)}</small>` : "";
  return `<div class="quad-table-row ${escapeHtml(row.status)} ${row.is_current_quarter ? "current" : ""}">
    <span class="quad-quarter"><strong>${escapeHtml(row.quarter)}</strong><small>${escapeHtml(row.is_current_quarter ? `current · ${row.status}` : row.status)}</small></span>
    <span>${quadBadge(row.quad)}</span>
    <span class="quad-metric"><strong>${quadPct(row.growth_yoy)}</strong><small>${quadArrow(row.growth_change)} ${quadPp(row.growth_change)}</small>${growthRange}</span>
    <span class="quad-metric"><strong>${quadPct(row.inflation_yoy)}</strong><small>${quadArrow(row.inflation_change)} ${quadPp(row.inflation_change)}</small>${inflationRange}</span>
    <span class="quad-scenarios">${grid}</span>
  </div>`;
}

function quadMapMarkup(rows) {
  const firstAhead = rows.findIndex((row) => row.status !== "realized");
  const start = Math.max(0, (firstAhead < 0 ? rows.length : firstAhead) - 4);
  const points = rows.slice(start).filter((row) => numberOrNull(row.growth_change) != null && numberOrNull(row.inflation_change) != null);
  if (!points.length) return `<div class="empty-state">No quarters to plot.</div>`;
  const extent = Math.max(0.5, ...points.flatMap((row) => [Math.abs(row.growth_change), Math.abs(row.inflation_change)])) * 1.15;
  const size = 320;
  const pad = 26;
  const scale = (value) => pad + ((value + extent) / (2 * extent)) * (size - 2 * pad);
  const x = (row) => scale(row.growth_change);
  const y = (row) => size - scale(row.inflation_change);
  const mid = size / 2;
  const realized = points.filter((row) => row.status === "realized");
  const lastRealizedIndex = points.lastIndexOf(realized[realized.length - 1]);
  const solid = points.slice(0, lastRealizedIndex + 1);
  const dashed = points.slice(Math.max(0, lastRealizedIndex));
  const line = (list) => list.map((row, index) => `${index ? "L" : "M"}${x(row).toFixed(1)},${y(row).toFixed(1)}`).join(" ");
  const labels = [
    { text: "QUAD 3", sub: "growth ↓ inflation ↑", x: pad + 6, y: pad + 12, anchor: "start" },
    { text: "QUAD 2", sub: "growth ↑ inflation ↑", x: size - pad - 6, y: pad + 12, anchor: "end" },
    { text: "QUAD 4", sub: "growth ↓ inflation ↓", x: pad + 6, y: size - pad - 16, anchor: "start" },
    { text: "QUAD 1", sub: "growth ↑ inflation ↓", x: size - pad - 6, y: size - pad - 16, anchor: "end" },
  ];
  return `<svg class="quad-map" viewBox="0 0 ${size} ${size}" role="img" aria-label="Quarterly path through the four quads">
    <rect class="qbg q3" x="${pad}" y="${pad}" width="${mid - pad}" height="${mid - pad}"></rect>
    <rect class="qbg q2" x="${mid}" y="${pad}" width="${mid - pad}" height="${mid - pad}"></rect>
    <rect class="qbg q4" x="${pad}" y="${mid}" width="${mid - pad}" height="${mid - pad}"></rect>
    <rect class="qbg q1" x="${mid}" y="${mid}" width="${mid - pad}" height="${mid - pad}"></rect>
    <line class="axis" x1="${pad}" x2="${size - pad}" y1="${mid}" y2="${mid}"></line>
    <line class="axis" y1="${pad}" y2="${size - pad}" x1="${mid}" x2="${mid}"></line>
    ${labels.map((label) => `<text class="quad-label" x="${label.x}" y="${label.y}" text-anchor="${label.anchor}">${label.text}</text><text class="quad-sublabel" x="${label.x}" y="${label.y + 10}" text-anchor="${label.anchor}">${label.sub}</text>`).join("")}
    <text class="axis-label" x="${size - pad}" y="${mid + 11}" text-anchor="end">growth accelerating →</text>
    <text class="axis-label" x="${mid + 5}" y="${pad - 8}">↑ inflation accelerating</text>
    <path class="trail" d="${line(solid)}"></path>
    <path class="trail projected" d="${line(dashed)}"></path>
    ${quadPointLabels(points, x, y).map(({ row, lx, ly }) => `<g class="quad-point ${row.status === "realized" ? "realized" : "projected"} q${row.quad}"><circle cx="${x(row).toFixed(1)}" cy="${y(row).toFixed(1)}" r="${row.is_current_quarter ? 6 : 4.5}"><title>${escapeHtml(row.quarter)} · ${escapeHtml(quadName(row.quad))} · growth ${quadPp(row.growth_change)}, inflation ${quadPp(row.inflation_change)}</title></circle><text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}">${escapeHtml(quadQuarterShort(row.quarter))}</text></g>`).join("")}
  </svg>
  <div class="quad-map-key"><span><i class="solid"></i>Realized</span><span><i class="dashed"></i>Projected base case</span></div>`;
}

function quadPointLabels(points, x, y) {
  const placed = [];
  return points.map((row) => {
    let lx = x(row) + 7;
    let ly = y(row) - 6;
    for (let attempt = 0; attempt < 6 && placed.some((label) => Math.abs(label.lx - lx) < 30 && Math.abs(label.ly - ly) < 10); attempt += 1) {
      ly += attempt % 2 === 0 ? 18 : -30;
      if (attempt >= 3) lx = x(row) - 38;
    }
    placed.push({ lx, ly });
    return { row, lx, ly };
  });
}

function quadCompsMarkup(comps, baseMom) {
  const values = comps.filter((row) => numberOrNull(row.comp_mom) != null);
  if (!values.length) return `<div class="empty-state">No comparison months available.</div>`;
  const base = numberOrNull(baseMom) ?? 0;
  const max = Math.max(0.3, base, ...values.map((row) => Math.abs(row.comp_mom))) * 1.15;
  const width = 640;
  const height = 170;
  const top = 18;
  const bottom = 34;
  const zero = top + (height - top - bottom) * (max / (2 * max));
  const scaleY = (value) => zero - (value / max) * ((height - top - bottom) / 2);
  const step = (width - 40) / comps.length;
  const bars = comps.map((row, index) => {
    const value = numberOrNull(row.comp_mom);
    const cx = 30 + index * step + step / 2;
    if (value == null) return `<text class="comp-month" x="${cx}" y="${height - 16}" text-anchor="middle">${escapeHtml(row.month.slice(2))}</text>`;
    const y1 = Math.min(scaleY(value), zero);
    const barHeight = Math.max(1, Math.abs(scaleY(value) - zero));
    const tone = `${value > base ? "drag" : "lift"}${row.comp_interpolated ? " interpolated" : ""}`;
    return `<rect class="comp-bar ${tone}" x="${(cx - step * 0.32).toFixed(1)}" y="${y1.toFixed(1)}" width="${(step * 0.64).toFixed(1)}" height="${barHeight.toFixed(1)}"><title>${escapeHtml(row.month)} compares with ${escapeHtml(row.comp_month)}: ${value.toFixed(2)}% MoM (${quadPct(row.comp_annualized, 1)} annualized)${row.comp_interpolated ? " · uses the interpolated, unpublished October 2025 CPI" : ""}</title></rect>
      <text class="comp-value" x="${cx}" y="${(value >= 0 ? y1 - 4 : y1 + barHeight + 10).toFixed(1)}" text-anchor="middle">${value.toFixed(2)}</text>
      <text class="comp-month" x="${cx}" y="${height - 16}" text-anchor="middle">${escapeHtml(row.month.slice(2))}</text>`;
  }).join("");
  return `<svg class="quad-comps" viewBox="0 0 ${width} ${height}" role="img" aria-label="CPI comparison months">
      <line class="axis" x1="24" x2="${width - 10}" y1="${zero}" y2="${zero}"></line>
      ${bars}
      <line class="base-line" x1="24" x2="${width - 10}" y1="${scaleY(base).toFixed(1)}" y2="${scaleY(base).toFixed(1)}"></line>
      <text class="base-label" x="26" y="${(scaleY(base) - 5).toFixed(1)}" text-anchor="start">base ${base.toFixed(2)}%/mo</text>
      <text class="comp-month" x="${width / 2}" y="${height - 2}" text-anchor="middle">month being reported (YY-MM) · bar = CPI change in the same month a year earlier, % MoM</text>
    </svg>
    <div class="quad-map-key"><span><i class="drag"></i>Above base: YoY inflation falls as it rolls off</span><span><i class="lift"></i>Below base: YoY inflation rises as it rolls off</span>${comps.some((row) => row.comp_interpolated) ? `<span><i class="interp"></i>Hatched: depends on an interpolated (unpublished) month</span>` : ""}</div>`;
}

function quadTradeabilityMarkup(payload) {
  const rows = (payload?.playbook || []).filter((row) => row.direction === "follow" && row.playbook === "textbook");
  if (!rows.length) return "";
  const hindsight = rows.find((row) => row.signal === "hindsight_current");
  const tradeable = rows.filter((row) => row.signal !== "hindsight_current");
  const anyEdge = (payload.playbook || []).some((row) => row.signal !== "hindsight_current" && row.significant);
  const range = tradeable.map((row) => numberOrNull(row.annualized_return)).filter((value) => value != null);
  const span = range.length ? `${quadSigned(Math.min(...range))} to ${quadSigned(Math.max(...range))}%/yr` : "n/a";
  return `<section class="quad-panel quad-verdict ${anyEdge ? "edge" : "none"}"><header><span>Is the quad tradeable?</span><em>Risk-balanced quad playbook, 3-month hold, ${escapeHtml(payload.sample_start)} → ${escapeHtml(payload.sample_end)}</em></header>
    <p><strong>${anyEdge ? "A tradeable quad signal shows an edge." : "No tradeable quad signal has an edge."}</strong> Following the textbook playbook with the projected or last-realized quad earned ${escapeHtml(span)}, none significant.${hindsight ? ` With hindsight quads it earned ${quadSigned(hindsight.annualized_return)}%/yr (q ${quadNum(hindsight.q_value, 2)}): the quad matters, but markets move before the data confirms it.` : ""} Treat the quad as context for HedgeEye-style commentary, not a position.</p>
  </section>`;
}

function quadSigned(value) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : signed(parsed, 1); }
function quadNum(value, decimals) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : parsed.toFixed(decimals); }

function quadBacktestMarkup(backtest) {
  const rows = backtest?.horizons || [];
  if (!rows.length) return `<div class="empty-state">Walk-forward check not run.</div>`;
  const rate = (value) => (value == null ? "n/a" : `${Math.round(value * 100)}%`);
  return `<div class="quad-table-wrap"><div class="quad-bt-head"><span>Horizon</span><span>Projections</span><span>Quad correct</span><span>Last quad repeats</span><span>Most common quad</span><span>Growth direction</span><span>Inflation direction</span><span>Brier (uniform 0.75)</span></div>
    ${rows.map((row) => `<div class="quad-bt-row"><span><strong>${escapeHtml(row.label)}</strong></span><span>${escapeHtml(row.observations)}</span><span class="emph">${rate(row.quad_hit_rate)}</span><span>${rate(row.persistence_hit_rate)}</span><span>${rate(row.unconditional_hit_rate)}</span><span>${rate(row.growth_direction_hit_rate)}</span><span>${rate(row.inflation_direction_hit_rate)}</span><span>${row.brier_calibrated == null ? "n/a" : row.brier_calibrated.toFixed(2)} <small>grid ${row.brier_scenario_grid == null ? "n/a" : row.brier_scenario_grid.toFixed(2)}</small></span></div>`).join("")}</div>
    <p class="quad-note">${escapeHtml(backtest.note || "")} Random choice among four quads is 25%. Brier scores the calibrated probabilities (lower is better); "grid" is the older 3×3 scenario share, which proved overconfident.</p>`;
}

function primaryRows() {
  return state.rows.filter((row) => !row.is_context);
}

function renderSummary() {
  const rows = primaryRows();
  const available = rows.filter((row) => row.history_status === "available").length;
  const turns = rows.filter((row) => row.persistence_label === "new turn").length;
  const stale = rows.filter((row) => row.freshness_status === "stale").length;
  const due = rows.filter((row) => row.freshness_status === "due").length;
  const asOf = rows.map((row) => new Date(row.as_of_date)).sort((a, b) => b - a)[0];
  setText("asof-date", formatDate(asOf));
  const missing = rows.filter((row) => row.history_status !== "available").map((row) => row.label);
  const parts = [
    `Data status · ${available} of ${rows.length} series with percentile history${missing.length ? ` (missing: ${missing.join(", ")})` : ""}`,
    `${turns} turned this release`,
    stale ? `${stale} stale, ${due} due for a new release` : due ? `${due} due for a new release, none stale` : "all sources fresh",
  ];
  document.getElementById("status-line").textContent = parts.join(" · ");
}

// --- Decision panel: validated signals only -------------------------------------------------

const OVERLAY_ACTION = {
  normal: { tone: "positive", action: "No action: stay invested." },
  macro_watch: { tone: "warning", action: "Watch: a macro warning without price confirmation." },
  market_warning: { tone: "warning", action: "Watch: price is below trend, but the economy has not confirmed. These breaks often whipsaw." },
  defensive: { tone: "negative", action: "Reduce risk: price and the economy both confirm. Future volatility and drawdowns have been reliably higher." },
  high_conviction: { tone: "negative", action: "Reduce risk: price, the economy, and durable orders all confirm." },
};

const TREND_ASSETS = [
  ["ES", "S&P 500"], ["NQ", "Nasdaq 100"], ["RTY", "Russell 2000"], ["ZN", "10y Treasury"],
  ["DX", "Dollar"], ["GC", "Gold"], ["CL", "Crude oil"],
];

function longTrendScore(asset) {
  const keys = new Set(["six_months", "ten_months", "twelve_months"]);
  const values = (asset?.components || []).filter((component) => keys.has(component.id)).map((component) => numberOrNull(component.adjusted)).filter((value) => value != null);
  return values.length === keys.size ? values.reduce((total, value) => total + value, 0) / values.length : null;
}

function trendLabel(score) {
  if (score == null) return { label: "n/a", tone: "neutral" };
  if (score > 0.25) return { label: "Up", tone: "positive" };
  if (score < -0.25) return { label: "Down", tone: "negative" };
  return { label: "Flat", tone: "neutral" };
}

function renderDecision() {
  const container = document.getElementById("decision-panel");
  const overlay = state.riskOverlay;
  if (!overlay?.state_id) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  const action = OVERLAY_ACTION[overlay.state_id] || { tone: "neutral", action: overlay.message };
  const inputs = Object.fromEntries((overlay.inputs || []).map((input) => [input.id, input]));
  const market = numberOrNull(inputs.market_trend?.value);
  const unemployment = numberOrNull(inputs.unemployment_trend?.value);
  const production = numberOrNull(inputs.industrial_production?.value);
  const buffer = numberOrNull(overlay.rule?.unemployment_buffer_pp) ?? 0;
  const distance = [
    market == null ? null : `S&P ${signed(market, 1)}% ${market >= 0 ? "above" : "below"} its 10-month average`,
    unemployment == null ? null : `unemployment ${signed(unemployment, 2)} pp vs its 12-month average (warns above ${signed(buffer, 1)})`,
    production == null ? null : `industrial production ${signed(production, 1)}% YoY (warns at or below 0)`,
  ].filter(Boolean).join(" · ");

  const momentum = new Map((state.directionalMomentum?.assets || []).map((asset) => [asset.id, asset]));
  const trends = TREND_ASSETS.map(([id, label]) => {
    const score = longTrendScore(momentum.get(id));
    const view = trendLabel(score);
    return `<span class="trend-chip ${view.tone}" title="Average of 6-, 10- and 12-month volatility-adjusted momentum: ${score == null ? "n/a" : signed(score, 2)}"><b>${escapeHtml(label)}</b>${escapeHtml(view.label)}</span>`;
  }).join("");

  const options = (state.premiumSelling?.assets || []).filter((asset) => numberOrNull(asset.options?.forward_iv_edge) != null);
  const spx = options.find((asset) => asset.id === "ES");
  const rich = options.filter((asset) => Number(asset.options.forward_iv_edge) > 2).length;
  const volLine = spx
    ? `S&P implied vol ${premiumNumber(spx.options.ivx)} vs ${premiumNumber(spx.options.forward_rv_forecast)} expected realized (${premiumSigned(spx.options.forward_iv_edge)} pts), IV rank ${ivRankText(spx.options.iv_rank)}.`
    : "S&P option data unavailable.";

  const forecasts = state.releases?.forecasts || [];
  const validated = forecasts.filter((row) => row.validated);
  const previews = forecasts.filter((row) => !row.validated && !(row.target_id === "payroll_change" && row.variant_id !== "core"));
  const validatedMarkup = validated.length
    ? validated.map((row) => `<p><strong>${escapeHtml(row.target_label)} (${escapeHtml(row.reference_month)}): ${releaseValue(row, row.forecast)}</strong> vs ${releaseValue(row, row.baseline_forecast)} naive baseline; last print ${releaseValue(row, row.prior)}.${row.model_mae != null ? ` Typical miss ${row.model_mae.toFixed(2)} vs ${row.baseline_mae.toFixed(2)} for the baseline.` : ""}</p>`).join("")
    : `<p>No release forecast currently passes validation.</p>`;
  const previewMarkup = previews.length ? `<small>Unvalidated previews: ${previews.map((row) => `${escapeHtml(row.target_label)} ${releaseValue(row, row.forecast)}`).join(" · ")}</small>` : "";
  const upcoming = nextReleasesMarkup();

  container.innerHTML = `<header><div><span class="eyebrow">WHAT TO DO NOW · VALIDATED SIGNALS ONLY</span><h2 class="decision-headline ${action.tone}">${escapeHtml(action.action)}</h2></div>
      <p>Everything below the line is context. The quad, monitor percentiles, and option strategy leans are not trading signals.</p></header>
    <div class="decision-grid">
      <article class="decision-card ${action.tone}"><span>Equity risk · ${escapeHtml(overlay.state_label)}</span>
        <p>${escapeHtml(distance)}.</p>
        <small>Risk-off needs the S&amp;P below its 10-month average <em>and</em> unemployment more than ${escapeHtml(buffer.toFixed(1))} pp above its 12-month average or industrial production contracting. Validated on future S&amp;P volatility and drawdowns across 17 separate stress periods.</small></article>
      <article class="decision-card"><span>Trend · 6–12 months</span>
        <div class="trend-chips">${trends}</div>
        <small>Trend following earned +3–4%/yr since 1990 (strongest at 12 months, weaker since 2010). Short-term range timing and quad filters added nothing.</small></article>
      <article class="decision-card"><span>Option pricing</span>
        <p>${escapeHtml(volLine)} ${rich} of ${options.length} optionable assets price implied vol more than 2 points above expected realized.</p>
        <small>Observed prices. Rich implied vol favors premium sellers on average; the per-asset strategy leans are untested.</small></article>
      <article class="decision-card"><span>Next data</span>
        ${validatedMarkup}${previewMarkup}${upcoming}</article>
    </div>`;
}

function releaseValue(row, value) {
  const parsed = numberOrNull(value);
  if (parsed == null) return "n/a";
  return row.unit === "percent" ? `${parsed.toFixed(2)}%` : `${Math.round(parsed)}k`;
}

function ivRankText(value) {
  const parsed = numberOrNull(value);
  if (parsed == null) return "n/a";
  return parsed > 100 ? "100+ (above its 1-year range)" : parsed.toFixed(0);
}

function nextReleasesMarkup() {
  const wanted = ["payroll_growth", "core_cpi_inflation", "core_pce_inflation", "real_gdp_growth", "industrial_production"];
  const rows = state.rows.filter((row) => wanted.includes(row.indicator_id) && row.expected_next_release)
    .map((row) => ({ label: row.label, date: new Date(row.expected_next_release), estimate: row.expected_next_release_is_estimate }))
    .sort((a, b) => a.date - b.date);
  if (!rows.length) return "";
  return `<small class="release-dates">Next releases: ${rows.map((row) => `${escapeHtml(row.label)} ${escapeHtml(formatDate(row.date))}${row.estimate ? "*" : ""}`).join(" · ")}${rows.some((row) => row.estimate) ? " (*estimated)" : ""}</small>`;
}

function renderFilters() {
  const nav = document.getElementById("component-filters");
  nav.innerHTML = Object.entries(COMPONENTS).map(([id, meta]) => {
    const count = id === "all" ? state.rows.length : state.rows.filter((row) => row.component === id).length;
    return `<button class="filter-button ${state.component === id ? "active" : ""}" type="button" data-component="${id}" style="--filter-color:${meta.color}"><i></i><b>${meta.label}</b><em>${count}</em></button>`;
  }).join("");
  nav.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    state.component = button.dataset.component;
    renderFilters();
    renderDataViews();
  }));
}

function filteredRows() {
  return state.rows.filter((row) => {
    const componentMatch = state.component === "all" || row.component === state.component;
    const textMatch = !state.search || `${row.label} ${row.series_ids}`.toLowerCase().includes(state.search);
    return componentMatch && textMatch && (!state.attentionOnly || isAttention(row));
  });
}

function renderDataViews() {
  renderRows();
  renderQuads();
  renderDirectionalMomentum();
  renderPremiumSelling();
}

function renderPremiumSelling() {
  const container = document.getElementById("premium-view");
  const payload = state.premiumSelling;
  const assets = Array.isArray(payload?.assets) ? payload.assets : [];
  if (!assets.length) {
    container.innerHTML = `<div class="empty-state">Premium-selling data are unavailable.</div>`;
    return;
  }
  const groups = ["futures", "sectors"].map((group) => {
    const rows = assets.filter((asset) => asset.group === group);
    if (!rows.length) return "";
    const optionCovered = rows.filter((asset) => asset.evidence?.grade === "B").length;
    const pathOnly = rows.length - optionCovered;
    return `<section class="premium-group"><header><span>${escapeHtml(rows[0].group_label)}</span><em>${optionCovered} option-covered${pathOnly ? ` · ${pathOnly} path-only` : ""}</em></header>
      <div class="premium-table-wrap"><div class="premium-table-head"><span>Asset / strategy lean</span><span>Analog path (untested)</span><span>Volatility edge</span><span>16Δ put side</span><span>16Δ call side</span><span>Evidence</span></div>
      ${rows.map(premiumAssetMarkup).join("")}</div></section>`;
  }).join("");
  const uncovered = (payload.uncovered_futures || []).map((item) => item.id).join(", ");
  const method = payload.methodology || {};
  container.innerHTML = `<div class="premium-intro"><div><span class="eyebrow">ONE-MONTH OPTIONS</span><h3>Option pricing first; analog leans are untested</h3></div><p>The volatility edge, IV rank and strike distances are observed prices. The path, touch rates and strategy lean come from the 100 most similar momentum readings; across our assets, short-horizon momentum has not predicted the next month, and no lean has been tested on option P&amp;L.</p></div>
    <div class="premium-key"><span><i class="positive"></i>Supportive</span><span><i class="negative"></i>Adverse</span><span><i class="neutral"></i>No clean edge</span><em>Touch = underlying crossed today's strike distance during a historical 21-session path.</em></div>
    <div class="premium-stack">${groups}</div>
    <div class="premium-coverage"><strong>Not covered</strong><span>${escapeHtml(uncovered || "None")}</span><small>No matching option surface in the Tanuki lake; directional momentum remains available on the Momentum tab.</small></div>
    <div class="premium-method"><strong>Method and limits</strong><span>${escapeHtml(method.path || "")}</span><span>${escapeHtml(method.touch || "")}</span><span>${escapeHtml(method.volatility || "")}</span><span>${escapeHtml(method.evidence || "")}</span><em>${escapeHtml(method.warning || "")}</em></div>`;
}

function premiumAssetMarkup(asset) {
  const path = asset.path || {};
  const options = asset.options || {};
  const touch = asset.touch || {};
  const components = asset.components || {};
  const strategy = asset.strategy || { tone: "neutral", label: "Unavailable", reason: "" };
  const direction = asset.direction || { tone: "neutral", label: "Unavailable" };
  const vol = options.vol_state || { tone: "neutral", label: "Unavailable" };
  return `<article class="premium-asset-row">
    <div class="premium-main-row">
      <span class="premium-asset"><b>${escapeHtml(asset.id)}</b><strong>${escapeHtml(asset.label)}</strong><small>${escapeHtml(asset.underlying_symbol)} proxy · ${escapeHtml(formatDate(asset.observation_date))}</small><em class="premium-lean untested" title="Untested on option P&amp;L">Untested lean: ${escapeHtml(strategy.label)}</em><small>${escapeHtml(strategy.reason)}</small></span>
      <span class="premium-cell"><b class="tone-${escapeHtml(direction.tone)}">${escapeHtml(direction.label)}</b><strong>${premiumPct(path.median_return_pct)} median return</strong><small>${premiumPct(path.positive_rate_pct, false)} positive · MFE ${premiumPct(path.median_upside_mfe_pct)} · MAE ${premiumPct(path.median_downside_mae_pct)}</small></span>
      <span class="premium-cell"><b class="tone-${escapeHtml(vol.tone)}">${escapeHtml(vol.label)}</b><strong>${premiumNumber(options.ivx)} IV − ${premiumNumber(options.forward_rv_forecast)} RV</strong><small>${premiumSigned(options.forward_iv_edge)} vol pts · IV rank ${escapeHtml(ivRankText(options.iv_rank))} · trailing spread ${premiumSigned(options.trailing_iv_hv_spread)}</small></span>
      ${premiumTouchMarkup("Put", options.put_16d_distance_pct, touch.put_conditional_pct, touch.put_unconditional_pct, touch.put_terminal_breach_pct)}
      ${premiumTouchMarkup("Call", options.call_16d_distance_pct, touch.call_conditional_pct, touch.call_unconditional_pct, touch.call_terminal_breach_pct)}
      <span class="premium-evidence"><b>${escapeHtml(asset.evidence?.grade || "—")}</b><strong>${escapeHtml(asset.evidence?.label || "Unavailable")}</strong><small>${escapeHtml(asset.evidence?.note || "")}</small></span>
    </div>
    <div class="premium-components"><span>Momentum components</span>${["one_month", "three_months", "six_months", "ten_months", "twelve_months"].map((key, index) => `<b>${["1m", "3m", "6m", "10m", "12m"][index]} <em>${premiumSigned(components[key])}</em></b>`).join("")}<span>${path.analog_count || 0} nearest analogs · ${path.sample_start || "—"}–${path.sample_end || "—"}</span><span>Expiry ${options.expiration_date ? formatDate(options.expiration_date) : "n/a"} · ${options.dte ?? "—"} DTE · skew pct ${premiumNumber(options.skew_percentile)}</span></div>
  </article>`;
}

function premiumTouchMarkup(label, distance, conditional, unconditional, terminal) {
  const change = numberOrNull(conditional) != null && numberOrNull(unconditional) != null ? Number(conditional) - Number(unconditional) : null;
  const tone = change == null ? "neutral" : change <= -3 ? "positive" : change >= 3 ? "negative" : "neutral";
  return `<span class="premium-cell"><b class="tone-${tone}">${escapeHtml(label)} touch ${premiumPct(conditional, false)}</b><strong>${premiumPct(distance, false)} OTM distance</strong><small>baseline ${premiumPct(unconditional, false)} · Δ ${premiumSigned(change)} pp · terminal breach ${premiumPct(terminal, false)}</small></span>`;
}

function premiumNumber(value) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : parsed.toFixed(1); }
function premiumSigned(value) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : signed(parsed, 1); }
function premiumPct(value, signedValue = true) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : `${signedValue ? signed(parsed, 1) : parsed.toFixed(1)}%`; }

function renderDirectionalMomentum() {
  const container = document.getElementById("momentum-view");
  const payload = state.directionalMomentum;
  const assets = Array.isArray(payload?.assets) ? payload.assets : [];
  if (!assets.length) {
    container.innerHTML = `<div class="empty-state">Directional momentum data are unavailable.</div>`;
    return;
  }
  const futures = assets.filter((asset) => asset.asset_class === "future");
  const sectors = assets.filter((asset) => asset.asset_class === "sector_etf");
  const groupOrder = ["equity_indexes", "treasuries", "dollar", "metals", "energy", "agriculture"];
  const futureGroups = groupOrder.map((group) => {
    const rows = futures.filter((asset) => asset.group === group).sort((a, b) => a.order - b.order);
    if (!rows.length) return "";
    return momentumGroupMarkup(rows[0].group_label, rows.map((asset) => momentumAssetRowMarkup(asset, "absolute")).join(""), rows.length);
  }).join("");
  const sectorRows = sectors.sort((a, b) => a.order - b.order).map((asset) => [
    momentumAssetRowMarkup(asset, "absolute"),
    momentumAssetRowMarkup(asset, "relative"),
  ].join("")).join("");
  const methodology = payload.methodology || {};
  container.innerHTML = `<div class="momentum-intro">
    <div><span class="eyebrow">DIRECTIONAL BIAS</span><h3>Multi-horizon trend</h3></div>
    <p>Positive always means the price is rising. The composite weights 6–12 months most (${escapeHtml((payload.horizons || []).map((horizon) => `${horizon.label} ${Math.round(Number(horizon.weight) * 100)}%`).join(" · "))}): across our assets, following 12-month trend earned the most and 3-month trend the least. Mixed readings are not forced into a direction.</p>
  </div>
  <div class="momentum-key"><span><i class="positive"></i>Positive</span><span><i class="negative"></i>Negative</span><span><i class="neutral"></i>Near zero</span><em>Scores are volatility-adjusted. Sector rows also show the raw % move; hover a futures cell for its normalized point move.</em></div>
  <div class="momentum-stack">${futureGroups}${momentumGroupMarkup("Select Sector SPDR ETFs", sectorRows, sectors.length, "Absolute and relative to SPY")}</div>
  <div class="momentum-method"><strong>Method</strong><span>Futures: ${escapeHtml(methodology.futures_raw || "")}</span><span>ETFs: ${escapeHtml(methodology.equity_raw || "")}</span><span>Risk adjustment: ${escapeHtml(methodology.adjusted || "")}</span><span>Relative: ${escapeHtml(methodology.relative || "")}</span></div>`;
}

function momentumGroupMarkup(label, rows, count, note = "") {
  return `<section class="momentum-group"><header><div><span>${escapeHtml(label)}</span>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div><em>${count} asset${count === 1 ? "" : "s"}</em></header>
    <div class="momentum-table-wrap"><div class="momentum-table-head"><span>Asset / lens</span><span>Bias</span><span>Alignment</span><span>1 month</span><span>3 months</span><span>6 months</span><span>10 months</span><span>12 months</span></div>${rows}</div>
  </section>`;
}

function momentumAssetRowMarkup(asset, lens) {
  const relative = lens === "relative";
  const bias = relative ? asset.relative_bias : asset.absolute_bias;
  const lensLabel = relative ? "vs SPY" : asset.asset_class === "future" ? "Futures price" : "Absolute";
  const components = (asset.components || []).map((component) => momentumComponentMarkup(component, asset.asset_class, relative)).join("");
  return `<div class="momentum-asset-row ${relative ? "relative" : ""}">
    <span class="momentum-asset"><b>${escapeHtml(asset.id)}</b><strong>${escapeHtml(asset.label)}</strong><small>${escapeHtml(lensLabel)} · ${escapeHtml(formatDate(asset.observation_date))}<i class="fresh-${escapeHtml(asset.freshness)}"></i></small></span>
    <span class="bias-badge ${escapeHtml(bias?.tone || "neutral")}"><b>${escapeHtml(bias?.label || "Unavailable")}</b><small>${numberOrNull(bias?.score) == null ? "—" : signed(Number(bias.score), 2)} composite</small></span>
    <span class="momentum-alignment"><b>${escapeHtml(bias?.alignment || "—")}</b><small>${escapeHtml(bias?.confidence || "—")} confidence</small></span>
    ${components}
  </div>`;
}

function momentumComponentMarkup(component, assetClass, relative) {
  const adjusted = numberOrNull(relative ? component.relative_adjusted : component.adjusted);
  const raw = numberOrNull(relative ? component.relative_raw : component.raw);
  const tone = adjusted == null ? "neutral" : adjusted > 0.25 ? "positive" : adjusted < -0.25 ? "negative" : "neutral";
  // Futures are back-adjusted, so their raw move is a scaled point change, not a percentage; show it only on hover.
  const rawDisplay = raw == null || assetClass === "future" ? "" : `${signed(raw * 100, 1)}%`;
  const title = raw == null ? "" : assetClass === "future" ? ` title="Normalized point move ${signed(raw * 100, 1)}"` : "";
  return `<span class="horizon-cell ${tone}"${title}><b>${adjusted == null ? "—" : signed(adjusted, 2)}</b>${rawDisplay ? `<small>${escapeHtml(rawDisplay)}</small>` : ""}</span>`;
}

function renderRows() {
  const filtered = filteredRows();
  const groupOrder = ["activity_labor", "inflation", "policy_financial_conditions", "housing"];
  const container = document.getElementById("series-groups");
  container.innerHTML = groupOrder.map((component) => {
    const rows = filtered.filter((row) => row.component === component).sort((a, b) => a.display_order - b.display_order);
    if (!rows.length) return "";
    const meta = COMPONENTS[component];
    return `<section class="series-group" style="--group-color:${meta.color}">
      <div class="group-heading"><i></i><span>${meta.label}</span><span>${rows.length} SERIES</span></div>
      ${rows.map((row) => rowMarkup(row, meta.color)).join("")}
    </section>`;
  }).join("");
  document.getElementById("empty-state").hidden = filtered.length > 0;
  container.querySelectorAll(".series-row").forEach((button) => {
    button.addEventListener("click", () => openDrawer(button.dataset.indicator));
  });
}

function rowMarkup(row, color) {
  const level = numberOrNull(row.level_percentile_10y);
  const velocity = numberOrNull(row.velocity_percentile_10y);
  const direction = row.persistence_direction || "flat";
  const directionGlyph = direction === "rising" ? "↑" : direction === "falling" ? "↓" : "→";
  const context = row.is_context ? `<span class="context-tag">CONTEXT</span>` : "";
  const flag = isAttention(row) ? `<i class="flag-dot" title="This row needs attention"></i>` : "";
  const momentum = momentumState(row);
  return `<button class="series-row" type="button" data-indicator="${escapeHtml(row.indicator_id)}" style="--row-color:${color}">
    <span class="series-id"><b>${escapeHtml(row.label)} ${context} ${flag}</b><small>${headlineReading(row)} · ${escapeHtml(row.series_ids)}</small></span>
    ${metricMarkup(level)}
    <span class="metric-cell"><i class="meter"><u class="meter-fill" style="--value:${velocity ?? 0}%"></u></i><b class="metric-number velocity-number ${momentum.tone}">${directionGlyph}${velocity == null ? "—" : Math.round(velocity)}</b></span>
    ${sparklineMarkup(row.indicator_id, color)}
    <span class="momentum ${momentum.tone} ${momentum.isTurn ? "turn" : ""}" title="${escapeHtml(momentum.toneLabel)}"><b>${escapeHtml(momentum.label)}</b>${momentum.note ? `<small>${escapeHtml(momentum.note)}</small>` : ""}</span>
    <span class="freshness ${escapeHtml(row.freshness_status)}"><i></i>${row.days_since_observation_release}d</span>
  </button>`;
}

function metricMarkup(value) {
  return `<span class="metric-cell"><i class="meter"><u class="meter-fill" style="--value:${value ?? 0}%"></u></i><b class="metric-number">${value == null ? "—" : Math.round(value)}</b></span>`;
}

function sparklineMarkup(indicatorId, color) {
  const values = (state.history.get(indicatorId) || []).map((item) => numberOrNull(item.level_percentile_10y)).filter((value) => value != null).slice(-18);
  if (values.length < 2) return `<svg class="sparkline" aria-label="Insufficient sparkline history"></svg>`;
  const points = values.map((value, index) => `${(index / (values.length - 1) * 104 + 2).toFixed(1)},${(29 - value / 100 * 26).toFixed(1)}`).join(" ");
  const last = points.split(" ").at(-1).split(",");
  return `<svg class="sparkline" viewBox="0 0 108 31" role="img" aria-label="Recent level percentile path" style="--row-color:${color}"><line class="grid" x1="0" y1="16" x2="108" y2="16"></line><polyline class="path" points="${points}"></polyline><circle class="end" cx="${last[0]}" cy="${last[1]}" r="2.3"></circle></svg>`;
}

function historySectionMarkup(row) {
  const history = historyForRange(row.indicator_id, state.historyRange);
  if (history.length < 2) return `<section class="detail-section"><h3>Release history</h3><div class="warning-box">Not enough saved release history to chart this row.</div></section>`;
  const ranges = [["1y", "1Y"], ["3y", "3Y"], ["5y", "5Y"], ["all", "All"]];
  return `<section class="detail-section history-section"><div class="history-heading"><h3>Release history</h3><div class="range-switch">${ranges.map(([value, label]) => `<button type="button" data-history-range="${value}" class="${state.historyRange === value ? "active" : ""}">${label}</button>`).join("")}</div></div>
    <p class="history-note">Point-in-time states as each release arrived. Orange marks identify releases containing revisions.</p>
    ${historyChartMarkup("Current reading", history, [{ key: "level_value", label: "Reading", css: "primary" }], null, true)}
    ${historyChartMarkup("Level percentile", history, [{ key: "level_percentile_10y", label: "10y", css: "primary" }, { key: "level_percentile_expanding", label: "Full", css: "secondary" }], [0, 100], false)}
    ${historyChartMarkup("Velocity percentile", history, [{ key: "velocity_percentile_10y", label: "10y", css: "primary" }, { key: "velocity_percentile_expanding", label: "Full", css: "secondary" }], [0, 100], false)}
  </section>`;
}

function historyForRange(indicatorId, range) {
  const history = [...(state.history.get(indicatorId) || [])].sort((a, b) => new Date(a.realtime_start) - new Date(b.realtime_start));
  if (!history.length || range === "all") return history;
  const years = Number.parseInt(range, 10);
  const latest = new Date(history.at(-1).realtime_start);
  const cutoff = new Date(latest);
  cutoff.setUTCFullYear(cutoff.getUTCFullYear() - years);
  return history.filter((item) => new Date(item.realtime_start) >= cutoff);
}

function historyChartMarkup(title, history, seriesDefinitions, fixedRange, showRevisions) {
  const width = 410;
  const height = 116;
  const bounds = { left: 34, right: 8, top: 12, bottom: 22 };
  const start = new Date(history[0].realtime_start).getTime();
  const end = new Date(history.at(-1).realtime_start).getTime();
  const allValues = seriesDefinitions.flatMap((definition) => history.map((item) => numberOrNull(item[definition.key])).filter((value) => value != null));
  if (allValues.length < 2) return `<div class="history-chart unavailable"><strong>${escapeHtml(title)}</strong><span>Insufficient chart history</span></div>`;
  let minimum = fixedRange ? fixedRange[0] : Math.min(...allValues);
  let maximum = fixedRange ? fixedRange[1] : Math.max(...allValues);
  if (minimum === maximum) { minimum -= 1; maximum += 1; }
  if (!fixedRange) { const padding = (maximum - minimum) * 0.12; minimum -= padding; maximum += padding; }
  const x = (item) => bounds.left + ((new Date(item.realtime_start).getTime() - start) / Math.max(1, end - start)) * (width - bounds.left - bounds.right);
  const y = (value) => bounds.top + (maximum - value) / (maximum - minimum) * (height - bounds.top - bounds.bottom);
  const paths = seriesDefinitions.map((definition) => {
    const points = history.map((item) => ({ item, value: numberOrNull(item[definition.key]) })).filter((point) => point.value != null);
    return `<path class="history-line ${definition.css}" d="${points.map((point, index) => `${index ? "L" : "M"}${x(point.item).toFixed(1)},${y(point.value).toFixed(1)}`).join(" ")}"></path>`;
  }).join("");
  const revisions = showRevisions ? history.filter((item) => item.is_revision === true && numberOrNull(item.level_value) != null).map((item) => `<circle class="revision-mark" cx="${x(item).toFixed(1)}" cy="${y(numberOrNull(item.level_value)).toFixed(1)}" r="2.8"><title>Revision release: ${formatDate(item.realtime_start)}</title></circle>`).join("") : "";
  const midpoint = fixedRange ? `<line class="history-midline" x1="${bounds.left}" y1="${y(50)}" x2="${width - bounds.right}" y2="${y(50)}"></line>` : "";
  const legend = seriesDefinitions.map((definition) => `<span class="${definition.css}"><i></i>${definition.label}</span>`).join("");
  return `<div class="history-chart"><div class="history-chart-head"><strong>${escapeHtml(title)}</strong><span>${legend}</span></div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)} release history">
    <line class="history-axis" x1="${bounds.left}" y1="${height - bounds.bottom}" x2="${width - bounds.right}" y2="${height - bounds.bottom}"></line>${midpoint}${paths}${revisions}
    <text class="history-axis-label" x="${bounds.left}" y="${height - 4}">${formatShortDate(history[0].realtime_start)}</text><text class="history-axis-label" x="${width - bounds.right}" y="${height - 4}" text-anchor="end">${formatShortDate(history.at(-1).realtime_start)}</text>
    <text class="history-axis-label" x="${bounds.left - 5}" y="${bounds.top + 3}" text-anchor="end">${formatAxisValue(maximum)}</text><text class="history-axis-label" x="${bounds.left - 5}" y="${height - bounds.bottom}" text-anchor="end">${formatAxisValue(minimum)}</text>
  </svg></div>`;
}

function formatAxisValue(value) { return Math.abs(value) >= 100 ? Math.round(value).toLocaleString("en-US") : value.toFixed(1); }
function formatShortDate(value) { const date = new Date(value); return new Intl.DateTimeFormat("en-US", { month: "short", year: "2-digit", timeZone: "UTC" }).format(date); }

function openDrawer(indicatorId) {
  const row = state.rows.find((item) => item.indicator_id === indicatorId);
  if (!row) return;
  const meta = COMPONENTS[row.component];
  const reading = READING_LANGUAGE[row.indicator_id] || { label: "Current reading", note: "" };
  const detailMetrics = parseDetailMetrics(row.detail_metrics);
  const marketRelevance = parseObject(row.market_relevance);
  const historySection = historySectionMarkup(row);
  const contextSection = detailMetrics.length ? `<section class="detail-section"><h3>Additional context</h3><div class="context-grid">
    ${detailMetrics.map(contextMetricMarkup).join("")}
  </div></section>` : "";
  const relevanceSection = marketRelevanceMarkup(marketRelevance);
  const warning = row.history_status !== "available" ? `<div class="warning-box">${escapeHtml(row.history_warning || "Normalized history is unavailable.")}</div>` : row.baseline_sensitive ? `<div class="warning-box">Trailing and expanding percentiles differ by at least 25 points. Treat this row as baseline-sensitive.</div>` : "";
  const momentum = momentumState(row);
  document.getElementById("detail-content").innerHTML = `<div style="--detail-color:${meta.color}">
    <div class="eyebrow">${meta.label.toUpperCase()}</div>
    <h2 class="drawer-title" id="detail-title">${escapeHtml(row.label)}</h2>
    <p class="drawer-subtitle">${escapeHtml(row.series_ids)} · ${escapeHtml(row.frequency)} · ${escapeHtml(row.polarity)}</p>
    ${warning}
    <section class="detail-section"><h3>Current state</h3><div class="detail-grid">
      ${detailStat(reading.label, formatRaw(row, "level"), reading.note)}
      ${detailStat("Raw velocity", formatRaw(row, "velocity"))}
      ${detailStat("Level percentile", formatPercentile(row.level_percentile_10y))}
      ${detailStat("Velocity percentile", formatPercentile(row.velocity_percentile_10y))}
      ${detailStat("Momentum", momentum.note ? `${momentum.label} · ${momentum.note}` : momentum.label)}
      ${detailStat("Freshness", `${row.freshness_status} · ${row.days_since_observation_release}d`)}
    </div></section>
    ${contextSection}
    ${relevanceSection}
    ${historySection}
    <section class="detail-section"><h3>Comparison baselines</h3>
      ${compareRow("Level 10y", row.level_percentile_10y)}${compareRow("Level full", row.level_percentile_expanding)}
      ${compareRow("Velocity 10y", row.velocity_percentile_10y)}${compareRow("Velocity full", row.velocity_percentile_expanding)}
    </section>
    <section class="detail-section"><h3>Source and timing</h3><dl class="detail-list">
      ${detailLine("Observation", formatDate(row.observation_date))}
      ${detailLine("Latest release", formatDate(row.release_date))}
      ${detailLine("Next release est.", formatDate(row.expected_next_release))}
      ${detailLine("Revision status", row.revision_status)}
      ${detailLine("Source series", row.series_ids)}
      ${detailLine("History coverage", `${row.level_history_observations} obs · ${Number(row.level_history_years).toFixed(1)}y`)}
    </dl></section>
    <section class="detail-section"><h3>Recipe</h3><div class="warning-box">${escapeHtml(RECIPES[row.recipe] || row.recipe)}</div></section>
  </div>`;
  document.querySelectorAll("[data-history-range]").forEach((button) => button.addEventListener("click", () => {
    state.historyRange = button.dataset.historyRange;
    openDrawer(indicatorId);
  }));
  const drawer = document.getElementById("detail-drawer");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  document.querySelector(".drawer-close").focus();
}

function closeDrawer() {
  const drawer = document.getElementById("detail-drawer");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function isAttention(row) {
  return row.history_status !== "available" || row.freshness_status !== "fresh" || row.baseline_sensitive === true || row.persistence_label === "new turn";
}

function momentumState(row) {
  const direction = row.persistence_direction;
  if (row.indicator_id === "treasury_curve" && row.curve_move_label) {
    const count = Number(row.persistence_count);
    const isTurn = count === 1;
    const note = isTurn ? "turned this release" : Number.isFinite(count) && count > 1 ? `${count} releases` : "";
    return { label: row.curve_move_label, note, isTurn, tone: "neutral", toneLabel: "Curve classification based on changes in the 10-year and 3-month yield legs" };
  }
  if (!direction) return { label: "Unavailable", note: "", isTurn: false, tone: "neutral", toneLabel: "Directional interpretation unavailable" };
  if (direction === "flat") return { label: "Little changed", note: "", isTurn: false, tone: "neutral", toneLabel: "Neutral momentum direction" };
  const language = MOMENTUM_LANGUAGE[row.indicator_id] || { rising: "Rising", falling: "Falling" };
  const count = Number(row.persistence_count);
  const isTurn = count === 1;
  const note = isTurn ? "turned this release" : Number.isFinite(count) && count > 1 ? `${count} releases` : "";
  const tone = momentumTone(row.indicator_id, direction);
  const toneLabel = tone === "positive" ? "Favorable momentum direction" : tone === "negative" ? "Unfavorable momentum direction" : "Neutral momentum direction";
  return { label: language[direction] || direction, note, isTurn, tone, toneLabel };
}

function momentumTone(indicatorId, direction) {
  const adverseWhenRising = new Set([
    "unemployment_rate",
    "core_pce_inflation",
    "core_cpi_inflation",
    "median_cpi",
    "five_year_inflation_expectations",
    "real_policy_rate",
    "high_yield_spread",
    "adjusted_financial_conditions",
    "mortgage_rate",
  ]);
  const risingIsPositive = !adverseWhenRising.has(indicatorId);
  return (direction === "rising") === risingIsPositive ? "positive" : "negative";
}

function parseDetailMetrics(value) {
  if (Array.isArray(value)) return value;
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function parseObject(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) return value;
  if (!value) return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch (_) {
    return {};
  }
}

function marketRelevanceMarkup(relevance) {
  const relationships = Array.isArray(relevance.relationships) ? relevance.relationships : [];
  if (!relationships.length) {
    return `<section class="detail-section"><h3>Market relevance</h3><div class="relevance-empty">No relationship is currently displayed for this indicator. None survived the standalone screen, or the available history was insufficient.</div></section>`;
  }
  const moreCount = Math.max(0, Number(relevance.relationship_count || relationships.length) - relationships.length);
  return `<section class="detail-section"><h3>Market relevance</h3>
    <div class="relevance-summary"><strong>${Number(relevance.robust_test_variants || 0)} robust test variants</strong><span>${Number(relevance.risk_test_variants || 0)} risk · ${Number(relevance.directional_test_variants || 0)} directional</span></div>
    <div class="relevance-grid">${relationships.map((item) => `<article>
      <span>${escapeHtml(item.asset)}</span>
      <strong>${escapeHtml(item.outcome)}</strong>
      <small>${escapeHtml((item.horizons || []).join(" / "))} · ${escapeHtml((item.features || []).join(", "))} · ${Number(item.test_variants || 0)} variant${Number(item.test_variants || 0) === 1 ? "" : "s"}</small>
    </article>`).join("")}</div>
    ${moreCount ? `<div class="relevance-more">+${moreCount} additional asset/outcome relationship${moreCount === 1 ? "" : "s"} in the research output</div>` : ""}
    <p class="relevance-note">${escapeHtml(relevance.note || "")}${relevance.sample_end ? ` Sample through ${escapeHtml(formatDate(relevance.sample_end))}.` : ""}</p>
  </section>`;
}

function contextMetricMarkup(metric) {
  const date = metric.observation_date ? formatDate(metric.observation_date) : "n/a";
  const signal = metric.signal_status ? `<em class="signal-badge ${escapeHtml(metric.signal_status)}">${escapeHtml(metric.signal_status)}</em>` : "";
  const note = metric.signal_note ? `<small class="signal-note">${escapeHtml(metric.signal_note)}</small>` : "";
  return `<div class="context-stat"><span>${escapeHtml(metric.label)}</span>${signal}<strong>${escapeHtml(formatContextMetric(metric))}</strong><small>${escapeHtml(metric.source)} · ${escapeHtml(date)}</small>${note}</div>`;
}

function formatContextMetric(metric) {
  const value = numberOrNull(metric.value);
  if (value == null) return "n/a";
  const decimals = Number.isFinite(Number(metric.decimals)) ? Number(metric.decimals) : 2;
  const formatted = value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  if (metric.unit?.startsWith("%")) return `${formatted}${metric.unit}`;
  return metric.unit ? `${formatted} ${metric.unit}` : formatted;
}

function formatRaw(row, kind) {
  const value = numberOrNull(row[`${kind}_value`]);
  if (value == null) return "n/a";
  const basisPoints = kind === "velocity" && ["real_policy_rate", "treasury_curve", "high_yield_spread", "mortgage_rate"].includes(row.indicator_id);
  if (basisPoints) return `${signed(value, 1)} bp`;
  if (row.indicator_id === "adjusted_financial_conditions") return signed(value, 2);
  if (row.indicator_id === "housing_starts_intensity") return kind === "level" ? `${value.toFixed(2)} / 1k` : `${signed(value, 2)}% YoY`;
  if (kind === "velocity") return `${signed(value, 2)} pp`;
  return `${value.toFixed(2)}%`;
}

function headlineReading(row) {
  const metrics = parseDetailMetrics(row.detail_metrics);
  if (["core_pce_inflation", "core_cpi_inflation"].includes(row.indicator_id)) {
    const mom = metrics.find((metric) => metric.unit === "% MoM");
    const yoy = metrics.find((metric) => metric.unit === "% YoY");
    if (mom && yoy) return `${Number(mom.value).toFixed(2)}% MoM · ${Number(yoy.value).toFixed(2)}% YoY`;
  }
  if (row.indicator_id === "median_cpi") return `${formatRaw(row, "level")} annualized MoM`;
  const note = READING_LANGUAGE[row.indicator_id]?.note;
  let text = `${formatRaw(row, "level")}${note ? ` ${note}` : ""}`;
  if (row.indicator_id === "real_gdp_growth") {
    const yoy = metrics.find((metric) => metric.unit === "% YoY");
    if (yoy) text += ` · ${Number(yoy.value).toFixed(2)}% YoY`;
  }
  // Only averaged series get a "latest" value, and only when it is the same measure (a level in %, not a monthly change).
  const averaged = /average/i.test(note || "");
  const latest = averaged ? metrics.find((metric) => /^latest/i.test(metric.label || "") && metric.unit === "%") : null;
  if (latest && numberOrNull(latest.value) != null) text += ` · latest ${Number(latest.value).toFixed(2)}%`;
  if (row.indicator_id === "payroll_growth") {
    const jobs = metrics.find((metric) => metric.unit === "thousand jobs" && /latest/i.test(metric.label || ""));
    if (jobs && numberOrNull(jobs.value) != null) text += ` · last month ${signed(Number(jobs.value), 0)}k jobs`;
  }
  if (row.history_status !== "available") text += ` · no percentile: ${row.level_history_observations ?? 0} obs of history`;
  return text;
}

function signed(value, decimals) { return `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(decimals)}`; }
function formatPercentile(value) { const parsed = numberOrNull(value); return parsed == null ? "n/a" : `${ordinal(Math.round(parsed))} pct`; }
function ordinal(value) { const remainder100 = value % 100; const remainder10 = value % 10; const suffix = remainder100 >= 11 && remainder100 <= 13 ? "th" : remainder10 === 1 ? "st" : remainder10 === 2 ? "nd" : remainder10 === 3 ? "rd" : "th"; return `${value}${suffix}`; }
function formatDate(value) { if (!value) return "n/a"; const date = value instanceof Date ? value : new Date(value); return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(date); }
function numberOrNull(value) { return value == null || Number.isNaN(Number(value)) ? null : Number(value); }
function setText(id, text) { document.getElementById(id).textContent = text; }
function detailStat(label, value, note = "") { return `<div class="detail-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ""}</div>`; }
function compareRow(label, value) { const parsed = numberOrNull(value); return `<div class="compare-row"><span>${escapeHtml(label)}</span><i class="compare-track"><i style="width:${parsed ?? 0}%"></i></i><b>${parsed == null ? "—" : Math.round(parsed)}</b></div>`; }
function detailLine(label, value) { return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value ?? "n/a"))}</dd></div>`; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }
