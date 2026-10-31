#!/usr/bin/env bash
# One command to reproduce everything: install dependencies, build the event split, train the models,
# generate predictions, the submission file and dashboard data, then launch the dashboard.
#
#   ./run_all.sh                  full run, ends by serving the dashboard on http://localhost:3000
#   ./run_all.sh --no-dashboard   stop after generating all outputs
#   ./run_all.sh --skip-install   reuse the existing .venv
#   ./run_all.sh --research       also rerun the research experiments (adds about 25 minutes)
#
# Needs Python 3.11+, about 1.5 GB of free disk (Python packages plus the dashboard's JavaScript build),
# and data/train_data.csv, data/test_data.csv (optionally data/challenge-1-event-labels.xlsx).
# Set VENV=/path/to/env to use a different virtual environment. Reruns reuse saved holdout results.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
DASHBOARD=1; INSTALL=1; RESEARCH=0
for arg in "$@"; do
  case "$arg" in
    --no-dashboard) DASHBOARD=0 ;;
    --skip-install) INSTALL=0 ;;
    --research) RESEARCH=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

step "Checking prerequisites"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"
command -v "$PYTHON" >/dev/null || fail "python3 not found. Install Python 3.11 or newer."
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || fail "Python 3.11+ required (found $("$PYTHON" --version 2>&1))."
for f in data/train_data.csv data/test_data.csv; do
  [ -f "$f" ] || fail "Missing $f. Put the supplied challenge files in data/."
done
[ -f data/challenge-1-event-labels.xlsx ] || echo "Note: data/challenge-1-event-labels.xlsx not found; labels are recomputed from train_data.csv without the cross-check."

if [ "$INSTALL" -eq 1 ]; then
  step "Installing dependencies into $VENV"
  [ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -r requirements-model.txt -r dashboard/requirements.txt \
    || fail "Dependency install failed (check free disk space and network)."
else
  [ -x "$VENV/bin/python" ] || fail "--skip-install given but $VENV does not exist."
fi
PY="$VENV/bin/python"
export MPLBACKEND=Agg MPLCONFIGDIR="$ROOT/artifacts/matplotlib-cache"
mkdir -p artifacts outputs

step "Running tests"
"$PY" -m unittest discover -s tests

step "Building the frozen event split (EDA notebook)"
"$PY" scripts/run_eda.py

if [ "$RESEARCH" -eq 1 ]; then
  step "Research experiments (row-level v2, ensembles)"
  "$PY" -m src.experiment_v2
  "$PY" -m src.final_tier_ensembles
fi

step "Training the final-tier model (cross-validation, selection, one-time holdout)"
"$PY" scripts/train_final_tier.py --finalize

step "Calibrating final-tier probabilities"
"$PY" scripts/train_tier_probability.py --holdout

step "Training the per-message Pc model and scoring test messages"
"$PY" scripts/train_row_model.py
"$PY" scripts/predict.py

step "Writing the submission file"
"$PY" scripts/make_submission.py

step "Exporting dashboard data"
"$PY" scripts/build_dashboard_data.py
"$PY" scripts/export_dashboard_outputs.py

cat <<SUMMARY

Outputs
  submission/submission.csv           one row per test event: predicted final tier + probability
  outputs/test_event_predictions.csv  event predictions with all probabilities and persistence tier
  outputs/test_predictions.csv        per-message estimated Pc and tiers
  outputs/event_predictions.csv       dashboard data (events), with outputs/cdm_history.csv
  outputs/dashboard/                  risk-rule tables, action queues and insight text
  reports/                            model_development.md, final_tier_report.md, tier_probability_report.md
SUMMARY

if [ "$DASHBOARD" -eq 1 ]; then
  step "Starting the dashboard on http://localhost:3000 (Ctrl+C to stop)"
  echo "In the top bar, choose 'Real model outputs (outputs/)' and press reload."
  cd dashboard
  [ -x "$VENV/bin/reflex" ] || fail "Reflex is not installed in $VENV (pip install -r dashboard/requirements.txt)."
  exec "$VENV/bin/reflex" run --env prod --single-port
fi
