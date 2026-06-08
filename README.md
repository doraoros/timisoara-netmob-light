# Timișoara NetMob Light

Proiect de disertație (UPT, 2026): construiesc un dataset radio + mobilitate din măsurători NetMonster pe rute din Timișoara, etichetez schimbările de celulă servitoare și antrenez modele ML care estimează dacă urmează o schimbare în ~15 secunde.

**Date:** export NetMonster (`notes/`), trasee GPS la 1 Hz din `gps_simulated/` (aliniate pe rute reale, nu log GNSS live), Speedtest punctual. Ținta `handoff_next` = schimbare CID/PCI în următorul interval de 15 s — nu eveniment 3GPP de handover.

**Set procesat:** 31 320 rânduri · 16 sesiuni · 9 rute · prevalență pozitivi ~27% (`reports/text/dataset_summary.json`).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Instalare

```powershell
git clone https://github.com/doraoros/timisoara-netmob-light.git
cd timisoara-netmob-light
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[test]
```

Conda: `conda env create -f environment.yml` · parametri în [`config.yaml`](config.yaml).

## Rulare

```powershell
.\run_all.ps1          # Windows
bash run_all.sh        # Linux / Git Bash
```

Pași principali:

| Pas | Comandă | Output |
|-----|---------|--------|
| Dataset | `python -m src.utils.build_radio_gps_dataset` | `data/processed/dataset_radio_gps.csv` |
| Validare | `python -m src.utils.validate_dataset` | `reports/text/dataset_summary.json` |
| Hărți rute | `python scripts/generate_route_map.py` · `python scripts/generate_interactive_route_map.py` | PNG/PDF în `reports/figures/thesis/` · HTML în `reports/figures/interactive/` |
| Baseline ML (Stage 1) | `python -m src.models.train_handoff_models` | `reports/text/handoff_model_metrics.json` |
| Feature engineering (Stage 2) | `python scripts/experiment_feature_engineering.py` | `reports/text/feature_engineering_experiment.json` |
| Robustețe | `python scripts/experiment_robustness.py` | `reports/text/robustness_experiment.json` |
| Figuri + capitol rezultate | `python -m src.reports.generate_dissertation_bundle` | `dissertation/CAPITOL_REZULTATE.md` |

Scripturile din `scripts/` necesită `PYTHONPATH` spre rădăcina proiectului.

## Hărți rute

Două variante: statică (disertație) și interactivă (demo/prezentare).

**Statică** — PNG/PDF offline, fundal dark, o linie per sesiune:

```bash
python scripts/generate_route_map.py
```

Output: `reports/figures/thesis/route_map.png` (+ PDF)

**Interactivă** — HTML cu trasee animate, fullscreen, minimap:

```bash
python scripts/generate_interactive_route_map.py
```

Output: `reports/figures/interactive/interactive_route_map.html`

sau `make routemap` pentru ambele.

## Rezultate

Am două etape de modelare; metricile sunt în fișiere JSON regenerate de pipeline.

**Stage 1** — coloane brute, fără inginerie secvențială (`train_handoff_models.py`). Cel mai bun model: regresie logistică, ROC-AUC holdout **0.643**.

**Stage 2** — trăsături secvențiale + curățare coloane (`experiment_feature_engineering.py`). Cel mai bun model: **XGBoost**, ROC-AUC holdout **0.792**, PR-AUC **0.450**, CV ROC-AUC **0.808 ± 0.064**.

| Model | ROC-AUC S1 | ROC-AUC S2 | PR-AUC S2 |
|-------|----------:|----------:|----------:|
| Logistic Regression | 0.629 | 0.776 | 0.382 |
| Random Forest | 0.546 | 0.726 | 0.334 |
| Gradient Boosting | 0.554 | 0.759 | 0.393 |
| **XGBoost** | 0.551 | **0.792** | **0.450** |

Alte experimente (opțional, `pip install -e .[deep]`): modele secvențiale GRU/Transformer, analiză SHAP, leave-one-route-out (ROC-AUC mediu ~0.87), model salvat ~570 KB / inferență ~13 ms.

Capitol detaliat: [`dissertation/CAPITOL_REZULTATE.md`](dissertation/CAPITOL_REZULTATE.md). Limitări: [`reports/text/limitations.md`](reports/text/limitations.md).

## Structură

```
├── data/           raw, interim, processed (+ schema.json)
├── src/            parsing, features, models, reports, utils
├── scripts/        experimente și CLI (predict, evaluate, route map)
├── tests/          pytest
├── reports/        figuri și JSON generate
├── dissertation/   capitole Markdown (disertație)
├── models/         artefact .joblib (regenerabil, nu e în git)
├── config.yaml
└── run_all.ps1 / run_all.sh
```

## Documente utile

| Fișier | Conținut |
|--------|----------|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Surse, etichetare, validare |
| [`dissertation/CAPITOL_FEATURE_ENGINEERING.md`](dissertation/CAPITOL_FEATURE_ENGINEERING.md) | Stage 1 → Stage 2 |
| [`dissertation/CAPITOL_ROBUSTETE.md`](dissertation/CAPITOL_ROBUSTETE.md) | Robustețe, edge |
| [`reports/README.md`](reports/README.md) | Index figuri |

## Limitări (rezumat)

- Fără RSRP/RSRQ/SINR; radio = snapshot NetMonster expandat pe segmente egale.
- GPS sintetic, nu măsurat live pe teren.
- Speedtest rar — doar analiză exploratorie, nu antrenare.
- Nu compar operatorii la nivel național și nu detectez triggeri 3GPP.

## Autor

Teodora Oros — Universitatea Politehnica Timișoara, 2026.

Licență MIT. Datele brute rămân la autor.
