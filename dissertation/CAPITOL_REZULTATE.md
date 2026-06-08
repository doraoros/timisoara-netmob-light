# Capitol 3 — Rezultate experimentale

*Generat automat: 2026-06-04 00:34 · Pipeline Timișoara NetMob Light*

## 3.0 Rezumat

Obiectiv: pipeline reproductibil pentru estimarea schimbării iminente de celulă (CID/PCI) din NetMonster + mobilitate. Nu urmăresc triggeri 3GPP, benchmark național de operatori sau campanii tip NYU-METS.

Model raportat ca rezultat principal: **XGBoost Stage 2**. Stage 1 (LR, features brute) rămâne doar baseline.

| Întrebare | Răspuns scurt | Unde |
|-----------|---------------|------|
| Cât de multe date? | 31,320 rânduri, 16 sesiuni | `dataset_summary.json` |
| Există semnal pe features brute? | Da, LR ROC-AUC 0.643 | Stage 1 |
| Ajută ingineria secvențială? | Da, XGBoost 0.792 vs 0.551 | Stage 2 |
| Ține pe sesiuni noi? | CV 0.808 ± 0.064 | GroupKFold |
| Generalizează pe rute noi? | Parțial, LORO mediu 0.869 | `robustness_experiment.json` |
| Model mic / rapid? | 570 KB, 13.3 ms | edge benchmark |
| Benchmark operatori? | Nu | Speedtest rar |
| GPS live? | Nu | `limitations.md` |

## 3.1 Prezentare generală a datasetului

Campania a produs **31,320** eșantioane pe **16** sesiuni și **9** rute urbane (2026-05-13 08:05:00 — 2026-05-29 18:51:59).

**Operatori:**

- **Digi:** 17,220
- **Orange:** 8,340
- **Vodafone:** 5,760

**Mod transport:**

- tram: 10,920
- car: 10,800
- walking: 5,700
- bus: 3,900

Viteza medie: **22.9 km/h**.

**Figuri:** `route_map.png`, `mean_speed_by_route.png`, `target_distribution.png`.

## 3.2 Etichetare

- **handoff** (schimbare CID/PCI): 1.8%.
- **handoff_next** (în 15 s): 26.8%.
- Speedtest aliniat: 47.1% (14,760 rânduri cu download).

> Vezi `limitations.md` — GPS sintetic, expandare radio pe segmente egale.

## 3.3 Stage 1 — baseline pe features brute

Pipeline: `train_handoff_models.py` · metrici: `handoff_model_metrics.json`.
Fiecare secundă tratată izolat; modelele de ansamblu rămân aproape de aleator.

### Model de referință Stage 1: Regresie logistică

| Metrică | Holdout | CV mean ± std |
|---------|--------:|--------------:|
| ROC-AUC | 0.643 | 0.617 ± 0.051 |
| PR-AUC | 0.343 | 0.347 ± 0.088 |
| F1 | 0.420 | 0.339 ± 0.190 |

**Concluzie Stage 1:** există semnal (LR > majority), dar reprezentarea brută nu exploatează natura secvențială a fenomenului.

**Figuri:** `model_comparison_pr_auc.png`, `roc_curves_holdout.png`, `confusion_matrix_logistic_regression.png`.

## 3.4 Stage 2 — contribuția principală (inginerie + XGBoost)

Pipeline: `experiment_feature_engineering.py` · metrici: `feature_engineering_experiment.json`.
Trăsături cauzale (`dwell_time`, istoric HO, timp ciclic, target encoding CID/PCI); eliminare artefacte (`radio_cell_index`, oră/minut brut).

### Comparație Stage 1 → Stage 2 (holdout, sesiuni nevăzute)

| Model | ROC-AUC S1 | ROC-AUC S2 | PR-AUC S2 | F1 S2 |
|-------|----------:|----------:|---------:|------:|
| LogReg | 0.629 | 0.776 | 0.382 | 0.592 |
| RF | 0.546 | 0.726 | 0.334 | 0.529 |
| GradBoost | 0.554 | 0.759 | 0.393 | 0.523 |
| XGBoost | 0.551 | 0.792 | 0.450 | 0.537 |

### Model principal: XGBoost Stage 2

- Holdout: ROC-AUC **0.792**, PR-AUC **0.450**, F1 0.537
- GroupKFold: ROC-AUC 0.808 ± 0.064, PR-AUC 0.568 ± 0.149
- Artefact deploy: `models/handoff_xgb_stage2.joblib`

**Figuri:** `stage_comparison.png`, `feature_importance_stage2.png`.

**Notă:** `dwell_time` explică o parte din câștigul Stage 2 (expandare pe segmente) — detaliu în secțiunea 3.7 și în `limitations.md`.

## 3.5 Robustețe și generalizare

- **Per sesiune** (16 sesiuni): ROC-AUC median 0.953 (IQR 0.923–0.975)
- **Leave-one-route-out:** ROC-AUC mediu 0.869, PR-AUC mediu 0.651
- **Zgomot GPS:** ROC-AUC ≈ 0.825 (insensibil la jitter ±20 m)
- **Rată eșantionare:** ROC-AUC 0.77 (2 s) → 0.49 (5 s) — degradare așteptată
- **Edge:** 570 KB, 13.3 ms/inferență

Detalii: `CAPITOL_ROBUSTETE.md` · figuri `robustness_*.png`.

## 3.6 Modele secvențiale profunde (comparație diagnostică)

Antrenate pe semnale brute per-secundă (fără rolling hand-crafted):

| Model | ROC-AUC | PR-AUC |
|-------|--------:|-------:|
| GRU | 0.743 | 0.412 |
| Transformer-lite | 0.761 | 0.513 |
| XGBoost Stage 2 | 0.792 | 0.450 |

**Concluzie:** fenomenul este secvențial — Transformer depășește XGBoost pe PR-AUC. Detalii: `CAPITOL_MODELE_SECVENTIALE.md`.

## 3.7 Interpretabilitate (SHAP + LR + PDP)

- SHAP: `dwell_time_s` are cea mai mare contribuție.
- Grafic dependență + coeficienți LR: `explainability.md`, `partial_dependence.png`.

Detalii: `reports/text/explainability.md` · `CAPITOL_FEATURE_ENGINEERING.md` §9.bis.

## 3.8 Analiză throughput (EDA, fără antrenare)

Speedtest punctual (~47% rânduri aliniate): distribuție pe operator, relație exploratorie cu `handoff_next`. **Nu** susține benchmark de operatori.

**Figuri:** `throughput_by_operator_matched.png`, `throughput_vs_handoff_next.png`.

## 3.9 Ce nu rezultă din date

- triggeri handover 3GPP;
- benchmark național operatori;
- reproducere campanie NYU-METS;
- validare pe GPS live de teren.

Detalii: `reports/text/limitations.md`.

## 3.10 Concluzii

Am construit un set local (16 sesiuni) și un pipeline reproductibil. Pe holdout, XGBoost Stage 2 atinge ROC-AUC 0.792 (CV 0.808 ± 0.064). Stage 1 arată că există semnal și fără inginerie secvențială, dar ansamblurile rămân slabe până la Stage 2. Interpretabilitatea (SHAP) arată că o parte din performanță vine din structura expandării radio — limită asumată, nu ascunsă.
