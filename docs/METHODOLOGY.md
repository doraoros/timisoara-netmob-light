# Metodologie

## 1. Surse de date

| Sursă | Folder | Observații |
|-------|--------|------------|
| NetMonster | `data/raw/notes/` | Snapshot-uri JSON (CID, PCI, operator, tehnologie) — puține citiri per sesiune |
| GPS | `data/raw/gps_simulated/` | Trasee 1 Hz aliniate pe rute reale; nu am folosit log GNSS live |
| Speedtest | `data/raw/speedtest/` | Măsurători punctuale, folosite doar în EDA |
| Sesiuni | `data/interim/sessions.csv` | 16 sesiuni pe 9 rute |

Schema coloanelor: [`../data/schema.json`](../data/schema.json).

## 2. Construirea datasetului

**Expandare radio.** NetMonster dă puține celule per fișier; GPS are câte un punct pe secundă. Funcția `expand_radio_to_gps` repetă fiecare celulă pe segmente egale ale traseului (`segment_size = max(30, n_gps / n_radio)`). E o ipoteză de lucru, nu măsurătoare PHY la 1 Hz. Din cauza asta, `dwell_time` poate reflecta și structura expandării — am verificat efectul cu SHAP și l-am notat în [`../reports/text/limitations.md`](../reports/text/limitations.md).

**Speedtest.** Alipire cu `merge_asof` + filtru de distanță. Nu intră în antrenare.

**Output:** `data/processed/dataset_radio_gps.csv` (~31k rânduri).

## 3. Etichete

- `handoff` = 1 când se schimbă CID sau PCI față de rândul anterior.
- `handoff_next` = 1 dacă apare un `handoff` în următoarele 15 s (`HO_HORIZON_S`).

Implementare: [`../src/features/handoff_labels.py`](../src/features/handoff_labels.py).

## 4. Trăsături și modele

**Stage 1** — viteză, oră/minut, rută, operator, `radio_cell_index`, etc. (`train_handoff_models.py`).

**Stage 2** — elimin coloane zgomotoase; adaug timp ciclic, Δviteză, istoric handover, `dwell_time`, target encoding pe CID/PCI. Trăsăturile secvențiale sunt calculate doar din trecut, per sesiune (`sequential_features.py`).

Modele comparate: Dummy, DecisionTree (depth 3), Logistic Regression, Random Forest, HistGradientBoosting, XGBoost.

## 5. Validare

- Holdout: 25% sesiuni (`GroupShuffleSplit`), restul antrenare.
- CV: `GroupKFold` pe `session_id` — nu amestec rânduri din aceeași cursă între train și test.
- Clasă dezechilibrată (~27% pozitivi): `class_weight='balanced'` / `scale_pos_weight`; raportez și PR-AUC.
- Metrici în JSON: holdout, `cv_mean`, `cv_std`.

Experimente suplimentare: leave-one-route-out, jitter GPS, downsampling (`experiment_robustness.py`), evaluare pe o sesiune (`evaluate_on_new_session.py`).

## 6. Limitări metodologice

- Ținta e derivată din CID/PCI, nu din semnalizare 3GPP.
- Lipsesc mărimi PHY (RSRP, RSRQ, SINR).
- Un singur oraș, set mic de sesiuni, GPS simulat.
- Speedtest prea rar pentru concluzii despre throughput continuu.

Detalii: [`../reports/text/limitations.md`](../reports/text/limitations.md).

## 7. Raportare față de NYU-METS

Campaniile NYU-METS folosesc throughput continuu (iPerf) și granularitate diferită. Proiectul meu e o variantă locală, cu date de terminal (NetMonster) și mobilitate modelată — nu o reproducere a acelor campanii.
