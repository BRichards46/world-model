#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
PYTHON="$VENV_PYTHON"
FRED_VINTAGE_DIR="/Users/brichards/Documents/data/fred/vintage"
SITE_DIR="$PROJECT_DIR/sites/macro-dashboard/dist"
OUTPUT_DIR="$PROJECT_DIR/outputs"
SERVER_PID_FILE="$OUTPUT_DIR/macro_dashboard_server.pid"
SERVER_LOG="$OUTPUT_DIR/macro_dashboard_server.log"
HOST="127.0.0.1"
PORT="8765"
START_SERVER=1

dashboard_args=()
for arg in "$@"; do
  case "$arg" in
    --no-serve)
      START_SERVER=0
      ;;
    *)
      dashboard_args+=("$arg")
      ;;
  esac
done

if [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" -c "import pandas, pyarrow, yaml" 2>/dev/null; then
  source "$PROJECT_DIR/venv/bin/activate"
else
  PYTHON="$(command -v python || command -v python3 || true)"
  if [ -z "$PYTHON" ] || ! "$PYTHON" -c "import pandas, pyarrow, yaml" 2>/dev/null; then
    echo "ERROR: No Python environment with the project dependencies is available." >&2
    echo "Install this project into $PROJECT_DIR/venv or provide pandas, pyarrow, and PyYAML to Python." >&2
    exit 1
  fi
  echo "WARNING: The project venv is incomplete; using $PYTHON" >&2
fi

if [ ! -d "$FRED_VINTAGE_DIR" ]; then
  echo "ERROR: FRED vintage directory is missing: $FRED_VINTAGE_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Checking that the Norgate update has finished writing..."
"$PYTHON" -m world_model.cli.wait_for_norgate

echo "Building incremental macro dashboard data..."
if [ "${#dashboard_args[@]}" -gt 0 ]; then
  "$PYTHON" -m world_model.cli.macro_dashboard "${dashboard_args[@]}"
else
  "$PYTHON" -m world_model.cli.macro_dashboard
fi

echo "Building growth/inflation quads with base-effect look-ahead..."
"$PYTHON" -m world_model.cli.growth_inflation_quad

echo "Refreshing asset performance by quad..."
"$PYTHON" -m world_model.cli.quad_asset_performance

echo "Building directional momentum data..."
"$PYTHON" -m world_model.cli.directional_momentum

echo "Building premium-selling suitability data..."
"$PYTHON" -m world_model.cli.premium_selling

echo "Refreshing the production-and-inflation risk study..."
"$PYTHON" -m world_model.cli.macro_regime_validation
echo "Describing risk for the current overlay state..."
"$PYTHON" -m world_model.cli.equity_risk_context
echo "Refreshing payroll and inflation release forecasts..."
"$PYTHON" -m world_model.cli.macro_release_forecast
echo "Building the bottom-up core CPI nowcast..."
"$PYTHON" -m world_model.cli.core_cpi
echo "Building the bottom-up energy and headline CPI nowcast..."
"$PYTHON" -m world_model.cli.energy_cpi
echo "Scoring release forecasts against Kalshi and the Cleveland Fed nowcast..."
"$PYTHON" -m world_model.cli.release_benchmarks || echo "WARNING: release benchmarks failed (network?); continuing." >&2
echo "Writing the outlook summary..."
"$PYTHON" -m world_model.cli.macro_outlook

# Research-only jobs (not shown on the dashboard) run on demand:
#   macro_regime_validation, macro_regime_performance, macro_regime_forecast,
#   quad_forecast_research, trend_range_study

if [ "$START_SERVER" -eq 0 ]; then
  echo "Dashboard data refreshed; local server skipped."
  exit 0
fi

# A prior run may still be serving correctly even if its PID file is stale
# (for example after an interrupted restart). Check the actual local endpoint
# before trying to bind the port again.
if server_page="$(curl --silent --fail --max-time 1 "http://$HOST:$PORT/" 2>/dev/null)" \
  && [[ "$server_page" == *"World Model · Macro Monitor"* ]]; then
  echo "Macro monitor is already running at http://$HOST:$PORT/"
  exit 0
fi

if [ -f "$SERVER_PID_FILE" ]; then
  server_pid="$(tr -d '[:space:]' < "$SERVER_PID_FILE")"
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    echo "Macro monitor is already running at http://$HOST:$PORT/"
    exit 0
  fi
fi

echo "Starting macro monitor at http://$HOST:$PORT/"
nohup "$PYTHON" -m http.server "$PORT" \
  --bind "$HOST" \
  --directory "$SITE_DIR" \
  > "$SERVER_LOG" 2>&1 &
server_pid=$!

sleep 0.25
if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "ERROR: Macro monitor server did not start. See: $SERVER_LOG" >&2
  tail -n 20 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "$server_pid" > "$SERVER_PID_FILE"

echo "Macro monitor PID: $server_pid"
echo "Server log: $SERVER_LOG"
