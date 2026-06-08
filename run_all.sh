#!/usr/bin/env bash
# Timisoara NetMob Light — one-click end-to-end reproducible pipeline.
# Usage:  bash run_all.sh
#
# Steps: install -> build dataset -> validate -> train baselines ->
# feature-engineering experiment -> robustness -> interpretability ->
# route map -> save deployable model -> thesis figures -> tests.
# Deep-learning / SHAP steps are optional (run only if torch / shap installed).
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

PY="${PYTHON:-python}"

echo "==> [0/12] Install package (editable)"
$PY -m pip install -e . -q

echo "==> [1/12] Build unified radio+GPS dataset"
$PY -m src.utils.build_radio_gps_dataset

echo "==> [2/12] Validate dataset"
$PY -m src.utils.validate_dataset

echo "==> [3/12] Train baseline + classic models (incl. Dummy + DecisionTree)"
$PY -m src.models.train_handoff_models

echo "==> [4/12] Feature-engineering experiment (Stage 1 vs Stage 2)"
$PY scripts/experiment_feature_engineering.py

echo "==> [5/12] Robustness, generalization & edge benchmark"
$PY scripts/experiment_robustness.py

echo "==> [6/12] Classic interpretability (LR coefficients + PDP)"
$PY scripts/interpretability_classic.py

echo "==> [7/12] Route maps (static PNG/PDF + interactive HTML)"
$PY scripts/generate_route_map.py
$PY scripts/generate_interactive_route_map.py

echo "==> [8/12] Train & save deployable model artifact"
$PY scripts/train_and_save_model.py

echo "==> [9/12] (optional) Deep sequential models (needs torch)"
$PY scripts/experiment_sequential.py || echo "   skipped (torch not installed)"

echo "==> [10/12] (optional) SHAP explainability (needs shap)"
$PY scripts/experiment_explainability.py || echo "   skipped (shap not installed)"

echo "==> [11/12] Thesis figures + dissertation bundle"
$PY -m src.reports.generate_thesis_figures
$PY -m src.reports.generate_dissertation_bundle

echo "==> [12/12] Run tests"
$PY -m pytest tests/ -q || echo "   pytest not installed (pip install -e .[test])"

echo ""
echo "Done. See reports/figures/ and dissertation/ for outputs."
