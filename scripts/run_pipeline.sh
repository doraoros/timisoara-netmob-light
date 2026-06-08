#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -e .
else
  .venv/bin/pip install -e . -q
fi

PY=.venv/bin/python

echo "[1/5] Building dataset..."
$PY -m src.utils.build_radio_gps_dataset

echo "[2/5] Validating..."
$PY -m src.utils.validate_dataset

echo "[3/5] Training models..."
$PY -m src.models.train_handoff_models

echo "[4/5] Thesis figures..."
$PY -m src.reports.generate_thesis_figures

echo "[5/5] Dissertation bundle..."
$PY -m src.reports.generate_dissertation_bundle

echo "Done. See dissertation/CAPITOL_REZULTATE.md"
