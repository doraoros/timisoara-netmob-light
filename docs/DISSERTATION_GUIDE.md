# Ghid disertație — ce copiezi unde

## Înainte de susținere

```cmd
scripts\run_pipeline.bat
```

(PowerShell: `powershell -ExecutionPolicy Bypass -File .\scripts\run_pipeline.ps1`)

Verifică că există toate fișierele din [reports/README.md](../reports/README.md).

---

## Capitol 1–2 (Introducere + Metodologie)

| Secțiune | Sursă |
|----------|--------|
| Descriere campanie Timișoara | `data/interim/sessions.csv`, `docs/METHODOLOGY.md` |
| Diagramă pipeline | `reports/figures/thesis/pipeline_overview.png` |
| Limitări | Copiază/adaptează `reports/text/limitations.md` **integral** |

---

## Capitol 3 (Rezultate experimentale)

### 3.2.4 Poziționare față de literatură

| Figură | Fișier |
|--------|--------|
| Figura 3.1 | `reports/figures/thesis/methodological_positioning.png` |

Legendă: *Figura 3.1. Poziționarea metodologică a studiului față de cercetările existente.*

### 3.5 Pipeline date

| Figură | Fișier |
|--------|--------|
| Pipeline | `reports/figures/thesis/pipeline_overview.png` |

### 3.1 Statistici dataset (capitol Rezultate)

Deschide `reports/text/dataset_summary.json` sau capitolul generat:

**`dissertation/CAPITOL_REZULTATE.md`**

Tabel sesiuni: `reports/text/tables/operator_handoff.csv`, `route_mobility.csv`

### 3.2 Figuri obligatorii (PNG 300 DPI)

| Figură | Fișier |
|--------|--------|
| Distribuție ținte | `reports/figures/thesis/target_distribution.png` |
| Handoff pe operator | `reports/figures/thesis/handoff_rate_by_operator.png` |
| Viteză pe rută | `reports/figures/thesis/mean_speed_by_route.png` |
| Throughput operator | `reports/figures/thesis/throughput_by_operator_matched.png` |
| Throughput vs HO | `reports/figures/thesis/throughput_vs_handoff_next.png` |
| Comparație modele | `reports/figures/thesis/model_comparison_pr_auc.png` |
| Curbe ROC | `reports/figures/ml/roc_curves_holdout.png` |
| Importanță features | `reports/figures/thesis/feature_importance_top15.png` |
| Sesiuni heatmap | `reports/figures/thesis/session_handoff_heatmap.png` |

### 3.3 Tabele ML

| Tabel | Fișier |
|-------|--------|
| Metrici complete | `reports/text/tables/model_metrics_holdout.csv` |
| Metrici CV | `reports/text/tables/model_metrics_cv.csv` |

**Model de prezentat:** Regresie logistică (interpretabilă, cel mai bun ROC-AUC holdout).

### 3.4 Hărți interactive (anexă)

Rulează `notebooks/03_visualize_maps.ipynb` → HTML în `reports/figures/maps/`

---

## Capitol 4 (Discuții)

Puncte cheie (deja formulate în `CAPITOL_REZULTATE.md`):

1. Semnal predictiv moderat dar peste baseline → util pentru planificare urbană/QoS.
2. Viteza și contextul temporal domină (vezi `feature_importance_rf.csv`).
3. Diferențe între operatori la rată handoff și throughput (unde există speedtest).
4. Limitări GPS sintetic + radio expandat — nu extrapola la nivel național.

---

## Anexe

- `data/processed/dataset_radio_gps.csv` (schema în `data/README.md`)
- `reports/text/handoff_model_metrics.json`
- Capturi nPerf din `data/processed/nperf_*.png`

---

## Checklist jurizare

- [ ] `limitations.md` citit în prezentare
- [ ] Toate figurile au legendă și unități
- [ ] Baseline menționat explicit
- [ ] Nu se pretinde replicare NYU-METS 1:1
- [ ] Repo accesibil (USB / GitHub privat) dacă comisia cere reproducibilitate
