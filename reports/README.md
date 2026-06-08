# Rapoarte generate

Rulează `.\run_all.ps1` sau `bash run_all.sh`.

## Figuri teză (`figures/thesis/`)

| Fișier | Descriere |
|--------|-----------|
| `route_map.png` | Hărți rute statice (PNG/PDF, o linie per sesiune) |
| `../interactive/interactive_route_map.html` | Hartă interactivă Folium |
| `target_distribution.png` | Distribuție etichete |
| `mean_speed_by_route.png` | Viteză medie pe rută |
| `model_comparison_pr_auc.png` | Comparație modele Stage 1 |
| `stage_comparison.png` | Stage 1 vs 2 |
| `feature_importance_stage2.png` | Importanță features Stage 2 |
| `robustness_*.png` | Robustețe (LORO, jitter, sampling) |
| `sequential_comparison.png` | GRU / Transformer vs XGBoost |
| `logreg_coefficients.png`, `partial_dependence.png` | Interpretabilitate Stage 2 |
| `shap_*.png` | SHAP (după `make explainability`) |

## JSON (`text/`)

| Fișier | Conținut |
|--------|----------|
| `dataset_summary.json` | Statistici dataset |
| `handoff_model_metrics.json` | Stage 1 — holdout + CV ± std |
| `feature_engineering_experiment.json` | Stage 1 vs 2 |
| `robustness_experiment.json` | LORO, jitter, edge |
| `sequential_experiment.json` | GRU / Transformer |
| `limitations.md` | Limitări |

Capitole: `../dissertation/CAPITOL_*.md`
