# Timisoara NetMob Light — one-click end-to-end reproducible pipeline (Windows).
# Usage:  ./run_all.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot
$py = "python"

function Step($n, $msg, $block) {
    Write-Host "`n==> [$n] $msg" -ForegroundColor Cyan
    & $block
}

function StepOptional($n, $msg, $block) {
    Write-Host "`n==> [$n] $msg" -ForegroundColor Cyan
    try { & $block } catch { Write-Host "   skipped ($($_.Exception.Message))" -ForegroundColor Yellow }
}

Step  "0/12"  "Install package (editable)"                  { & $py -m pip install -e . -q }
Step  "1/12"  "Build unified radio+GPS dataset"             { & $py -m src.utils.build_radio_gps_dataset }
Step  "2/12"  "Validate dataset"                            { & $py -m src.utils.validate_dataset }
Step  "3/12"  "Train baseline + classic models"             { & $py -m src.models.train_handoff_models }
Step  "4/12"  "Feature-engineering experiment"              { & $py scripts/experiment_feature_engineering.py }
Step  "5/12"  "Robustness & edge benchmark"                 { & $py scripts/experiment_robustness.py }
Step  "6/12"  "Classic interpretability (LR + PDP)"         { & $py scripts/interpretability_classic.py }
Step  "7/12"  "Route maps (static + interactive)"         { & $py scripts/generate_route_map.py; & $py scripts/generate_interactive_route_map.py }
Step  "8/12"  "Train & save deployable model"              { & $py scripts/train_and_save_model.py }
StepOptional "9/12"  "Deep sequential models (needs torch)" { & $py scripts/experiment_sequential.py }
StepOptional "10/12" "SHAP explainability (needs shap)"     { & $py scripts/experiment_explainability.py }
Step  "11/12" "Thesis figures + dissertation bundle"        { & $py -m src.reports.generate_thesis_figures; & $py -m src.reports.generate_dissertation_bundle }
StepOptional "12/12" "Run tests"                            { & $py -m pytest tests/ -q }

Write-Host "`nDone. See reports/figures/ and dissertation/." -ForegroundColor Green
