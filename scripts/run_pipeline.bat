@echo off
REM Timisoara NetMob Light — pipeline (fara ExecutionPolicy PowerShell)
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Creating venv...
    python -m venv .venv
    call .venv\Scripts\pip install -e .
) else (
    call .venv\Scripts\pip install -e . -q
)

set PY=.venv\Scripts\python.exe

echo.
echo [1/5] Building dataset...
%PY% -m src.utils.build_radio_gps_dataset
if errorlevel 1 exit /b 1

echo.
echo [2/5] Validating...
%PY% -m src.utils.validate_dataset
if errorlevel 1 exit /b 1

echo.
echo [3/5] Training models...
%PY% -m src.models.train_handoff_models
if errorlevel 1 exit /b 1

echo.
echo [4/5] Thesis figures...
%PY% -m src.reports.generate_thesis_figures
if errorlevel 1 exit /b 1

echo.
echo [5/5] Dissertation bundle...
%PY% -m src.reports.generate_dissertation_bundle
if errorlevel 1 exit /b 1

echo.
echo Done. Open dissertation\CAPITOL_REZULTATE.md and reports\figures\thesis\
endlocal
