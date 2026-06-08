# Timișoara NetMob Light — pipeline complet
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating venv..."
    python -m venv .venv
    .\.venv\Scripts\pip install -e .
} else {
    .\.venv\Scripts\pip install -e . -q
}

$py = ".\.venv\Scripts\python.exe"

Write-Host "`n[1/5] Building dataset..." -ForegroundColor Cyan
& $py -m src.utils.build_radio_gps_dataset

Write-Host "`n[2/5] Validating..." -ForegroundColor Cyan
& $py -m src.utils.validate_dataset

Write-Host "`n[3/5] Training models..." -ForegroundColor Cyan
& $py -m src.models.train_handoff_models

Write-Host "`n[4/5] Thesis figures..." -ForegroundColor Cyan
& $py -m src.reports.generate_thesis_figures

Write-Host "`n[5/5] Dissertation bundle..." -ForegroundColor Cyan
& $py -m src.reports.generate_dissertation_bundle

Write-Host "`nDone. Open dissertation/CAPITOL_REZULTATE.md and reports/figures/thesis/" -ForegroundColor Green
