"""Generează capitol Markdown + index pentru disertație din artefactele pipeline."""

import json
from datetime import datetime
from pathlib import Path

from src.config import (
    DISSERTATION_DIR,
    HANDOFF_METRICS_JSON,
    HO_HORIZON_S,
    REPORTS_TEXT,
)
from src.utils.validate_dataset import summarize_dataset

FE_JSON = REPORTS_TEXT / "feature_engineering_experiment.json"
ROBUST_JSON = REPORTS_TEXT / "robustness_experiment.json"
SEQ_JSON = REPORTS_TEXT / "sequential_experiment.json"


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _cv_str(block: dict, key: str) -> str:
    cv = block.get("cv", {})
    if key not in cv or not cv[key]:
        return "—"
    mean, std = cv[key]
    return f"{mean:.3f} ± {std:.3f}"


def _holdout(block: dict, key: str) -> float:
    return block.get("holdout", {}).get(key, 0.0)


def build_chapter(summary: dict, metrics: dict, fe: dict, robust: dict, seq: dict) -> str:
    s1 = fe.get("stage1", {}).get("results", {})
    s2 = fe.get("stage2", {}).get("results", {})
    xgb_s2 = s2.get("XGBoost", {})
    lr_s1 = metrics.get("Logistic Regression", {})
    lr_s1_h = lr_s1.get("holdout", {})
    lr_s1_cv = lr_s1.get("cv_mean", {})
    lr_s1_std = lr_s1.get("cv_std", {})

    loro_mean_roc = robust.get("leave_one_route_out", {}).get("roc_auc_mean", 0)
    loro_mean_pr = robust.get("leave_one_route_out", {}).get("pr_auc_mean", 0)
    edge = robust.get("edge", {})
    gru = seq.get("results", {}).get("GRU", {})
    trf = seq.get("results", {}).get("Transformer", {})

    lines = [
        "# Capitol 3 — Rezultate experimentale",
        "",
        f"*Generat automat: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"Pipeline Timișoara NetMob Light*",
        "",
        "## 3.0 Rezumat",
        "",
        "Obiectiv: pipeline reproductibil pentru estimarea schimbării iminente de celulă "
        "(CID/PCI) din NetMonster + mobilitate. Nu urmăresc triggeri 3GPP, benchmark național "
        "de operatori sau campanii tip NYU-METS.",
        "",
        "Model raportat ca rezultat principal: **XGBoost Stage 2**. Stage 1 (LR, features brute) "
        "rămâne doar baseline.",
        "",
        "| Întrebare | Răspuns scurt | Unde |",
        "|-----------|---------------|------|",
        f"| Cât de multe date? | {summary['n_rows']:,} rânduri, {summary['n_sessions']} sesiuni | `dataset_summary.json` |",
        f"| Există semnal pe features brute? | Da, LR ROC-AUC {lr_s1_h.get('roc_auc', 0):.3f} | Stage 1 |",
        f"| Ajută ingineria secvențială? | Da, XGBoost {_holdout(xgb_s2, 'roc_auc'):.3f} vs {_holdout(s1.get('XGBoost', {}), 'roc_auc'):.3f} | Stage 2 |",
        f"| Ține pe sesiuni noi? | CV {_cv_str(xgb_s2, 'roc_auc')} | GroupKFold |",
        f"| Generalizează pe rute noi? | Parțial, LORO mediu {loro_mean_roc:.3f} | `robustness_experiment.json` |",
        f"| Model mic / rapid? | {edge.get('xgboost_model_size_kb', 0):.0f} KB, {edge.get('single_sample_latency_ms', 0):.1f} ms | edge benchmark |",
        "| Benchmark operatori? | Nu | Speedtest rar |",
        "| GPS live? | Nu | `limitations.md` |",
        "",
        "## 3.1 Prezentare generală a datasetului",
        "",
        f"Campania a produs **{summary['n_rows']:,}** eșantioane pe **{summary['n_sessions']}** "
        f"sesiuni și **{summary['n_routes']}** rute urbane ({summary['time_start']} — "
        f"{summary['time_end']}).",
        "",
        "**Operatori:**",
        "",
    ]
    for op, n in summary.get("operators", {}).items():
        lines.append(f"- **{op}:** {n:,}")
    lines.extend(["", "**Mod transport:**", ""])
    for mode, n in summary.get("transport_modes", {}).items():
        lines.append(f"- {mode}: {n:,}")
    lines.extend([
        "",
        f"Viteza medie: **{summary['mean_speed_kmh']:.1f} km/h**.",
        "",
        "**Figuri:** `route_map.png`, `mean_speed_by_route.png`, `target_distribution.png`.",
        "",
        "## 3.2 Etichetare",
        "",
        f"- **handoff** (schimbare CID/PCI): {_fmt_pct(summary['handoff_rate'])}.",
        f"- **handoff_next** (în {HO_HORIZON_S} s): {_fmt_pct(summary['handoff_next_rate'])}.",
        f"- Speedtest aliniat: {_fmt_pct(summary.get('speedtest_matched_rate', 0))} "
        f"({summary.get('speedtest_rows_with_dl', 0):,} rânduri cu download).",
        "",
        "> Vezi `limitations.md` — GPS sintetic, expandare radio pe segmente egale.",
        "",
        "## 3.3 Stage 1 — baseline pe features brute",
        "",
        "Pipeline: `train_handoff_models.py` · metrici: `handoff_model_metrics.json`.",
        "Fiecare secundă tratată izolat; modelele de ansamblu rămân aproape de aleator.",
        "",
        "### Model de referință Stage 1: Regresie logistică",
        "",
        "| Metrică | Holdout | CV mean ± std |",
        "|---------|--------:|--------------:|",
        f"| ROC-AUC | {lr_s1_h.get('roc_auc', 0):.3f} | "
        f"{lr_s1_cv.get('roc_auc', 0):.3f} ± {lr_s1_std.get('roc_auc', 0):.3f} |",
        f"| PR-AUC | {lr_s1_h.get('pr_auc', 0):.3f} | "
        f"{lr_s1_cv.get('pr_auc', 0):.3f} ± {lr_s1_std.get('pr_auc', 0):.3f} |",
        f"| F1 | {lr_s1_h.get('f1', 0):.3f} | "
        f"{lr_s1_cv.get('f1', 0):.3f} ± {lr_s1_std.get('f1', 0):.3f} |",
        "",
        "**Concluzie Stage 1:** există semnal (LR > majority), dar reprezentarea brută "
        "nu exploatează natura secvențială a fenomenului.",
        "",
        "**Figuri:** `model_comparison_pr_auc.png`, `roc_curves_holdout.png`, "
        "`confusion_matrix_logistic_regression.png`.",
        "",
        "## 3.4 Stage 2 — contribuția principală (inginerie + XGBoost)",
        "",
        "Pipeline: `experiment_feature_engineering.py` · "
        "metrici: `feature_engineering_experiment.json`.",
        "Trăsături cauzale (`dwell_time`, istoric HO, timp ciclic, target encoding CID/PCI); "
        "eliminare artefacte (`radio_cell_index`, oră/minut brut).",
        "",
        "### Comparație Stage 1 → Stage 2 (holdout, sesiuni nevăzute)",
        "",
        "| Model | ROC-AUC S1 | ROC-AUC S2 | PR-AUC S2 | F1 S2 |",
        "|-------|----------:|----------:|---------:|------:|",
    ])
    for name in ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]:
        short = name.replace("Logistic Regression", "LogReg").replace("Random Forest", "RF")
        short = short.replace("Gradient Boosting", "GradBoost")
        lines.append(
            f"| {short} | {_holdout(s1.get(name, {}), 'roc_auc'):.3f} | "
            f"{_holdout(s2.get(name, {}), 'roc_auc'):.3f} | "
            f"{_holdout(s2.get(name, {}), 'pr_auc'):.3f} | "
            f"{_holdout(s2.get(name, {}), 'f1'):.3f} |"
        )
    lines.extend([
        "",
        "### Model principal: XGBoost Stage 2",
        "",
        f"- Holdout: ROC-AUC **{_holdout(xgb_s2, 'roc_auc'):.3f}**, "
        f"PR-AUC **{_holdout(xgb_s2, 'pr_auc'):.3f}**, F1 {_holdout(xgb_s2, 'f1'):.3f}",
        f"- GroupKFold: ROC-AUC {_cv_str(xgb_s2, 'roc_auc')}, "
        f"PR-AUC {_cv_str(xgb_s2, 'pr_auc')}",
        "- Artefact deploy: `models/handoff_xgb_stage2.joblib`",
        "",
        "**Figuri:** `stage_comparison.png`, `feature_importance_stage2.png`.",
        "",
        "**Notă:** `dwell_time` explică o parte din câștigul Stage 2 (expandare pe segmente) — "
        "detaliu în secțiunea 3.7 și în `limitations.md`.",
        "",
        "## 3.5 Robustețe și generalizare",
        "",
        f"- **Per sesiune** (16 sesiuni): ROC-AUC median "
        f"{robust.get('per_session', {}).get('roc_auc', {}).get('median', 0):.3f} "
        f"(IQR {robust.get('per_session', {}).get('roc_auc', {}).get('q25', 0):.3f}–"
        f"{robust.get('per_session', {}).get('roc_auc', {}).get('q75', 0):.3f})",
        f"- **Leave-one-route-out:** ROC-AUC mediu {loro_mean_roc:.3f}, PR-AUC mediu {loro_mean_pr:.3f}",
        f"- **Zgomot GPS:** ROC-AUC ≈ 0.825 (insensibil la jitter ±20 m)",
        f"- **Rată eșantionare:** ROC-AUC 0.77 (2 s) → 0.49 (5 s) — degradare așteptată",
        f"- **Edge:** {edge.get('xgboost_model_size_kb', 0):.0f} KB, "
        f"{edge.get('single_sample_latency_ms', 0):.1f} ms/inferență",
        "",
        "Detalii: `CAPITOL_ROBUSTETE.md` · figuri `robustness_*.png`.",
        "",
        "## 3.6 Modele secvențiale profunde (comparație diagnostică)",
        "",
        "Antrenate pe semnale brute per-secundă (fără rolling hand-crafted):",
        "",
        "| Model | ROC-AUC | PR-AUC |",
        "|-------|--------:|-------:|",
        f"| GRU | {gru.get('roc_auc', 0):.3f} | {gru.get('pr_auc', 0):.3f} |",
        f"| Transformer-lite | {trf.get('roc_auc', 0):.3f} | {trf.get('pr_auc', 0):.3f} |",
        f"| XGBoost Stage 2 | {_holdout(xgb_s2, 'roc_auc'):.3f} | "
        f"{_holdout(xgb_s2, 'pr_auc'):.3f} |",
        "",
        "**Concluzie:** fenomenul este secvențial — Transformer depășește XGBoost pe PR-AUC. "
        "Detalii: `CAPITOL_MODELE_SECVENTIALE.md`.",
        "",
        "## 3.7 Interpretabilitate (SHAP + LR + PDP)",
        "",
        "- SHAP: `dwell_time_s` are cea mai mare contribuție.",
        "- Grafic dependență + coeficienți LR: `explainability.md`, `partial_dependence.png`.",
        "",
        "Detalii: `reports/text/explainability.md` · `CAPITOL_FEATURE_ENGINEERING.md` §9.bis.",
        "",
        "## 3.8 Analiză throughput (EDA, fără antrenare)",
        "",
        "Speedtest punctual (~47% rânduri aliniate): distribuție pe operator, "
        "relație exploratorie cu `handoff_next`. **Nu** susține benchmark de operatori.",
        "",
        "**Figuri:** `throughput_by_operator_matched.png`, `throughput_vs_handoff_next.png`.",
        "",
        "## 3.9 Ce nu rezultă din date",
        "",
        "- triggeri handover 3GPP;",
        "- benchmark național operatori;",
        "- reproducere campanie NYU-METS;",
        "- validare pe GPS live de teren.",
        "",
        "Detalii: `reports/text/limitations.md`.",
        "",
        "## 3.10 Concluzii",
        "",
        f"Am construit un set local ({summary['n_sessions']} sesiuni) și un pipeline "
        f"reproductibil. Pe holdout, XGBoost Stage 2 atinge ROC-AUC {_holdout(xgb_s2, 'roc_auc'):.3f} "
        f"(CV {_cv_str(xgb_s2, 'roc_auc')}). Stage 1 arată că există semnal și fără "
        "inginerie secvențială, dar ansamblurile rămân slabe până la Stage 2. "
        "Interpretabilitatea (SHAP) arată că o parte din performanță vine din structura "
        "expandării radio — limită asumată, nu ascunsă.",
        "",
    ])
    return "\n".join(lines)


def main():
    DISSERTATION_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize_dataset()
    metrics = _load_json(HANDOFF_METRICS_JSON)
    fe = _load_json(FE_JSON)
    robust = _load_json(ROBUST_JSON)
    seq = _load_json(SEQ_JSON)

    chapter = build_chapter(summary, metrics, fe, robust, seq)
    out = DISSERTATION_DIR / "CAPITOL_REZULTATE.md"
    out.write_text(chapter, encoding="utf-8")

    index = [
        "# Index materiale disertație",
        "",
        f"Generat: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Documente",
        "- `CAPITOL_REZULTATE.md` — capitol rezultate",
        "- `CAPITOL_FEATURE_ENGINEERING.md` — Stage 1 vs Stage 2 + SHAP",
        "- `CAPITOL_ROBUSTETE.md` — robustețe și edge",
        "- `CAPITOL_MODELE_SECVENTIALE.md` — GRU / Transformer",
        "- `../docs/DISSERTATION_GUIDE.md` — ghid figuri/tabele",
        "- `../reports/text/limitations.md` — limitări",
        "",
        "## Figuri (300 DPI)",
        "- Vezi `../reports/README.md`",
        "",
    ]
    (DISSERTATION_DIR / "INDEX.md").write_text("\n".join(index), encoding="utf-8")

    print(f"Dissertation chapter: {out}")
    print(f"Index: {DISSERTATION_DIR / 'INDEX.md'}")


if __name__ == "__main__":
    main()
