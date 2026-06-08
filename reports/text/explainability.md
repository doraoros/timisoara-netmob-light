# Interpretabilitatea modelului (analiză SHAP)

> Rezultate generate de `scripts/experiment_explainability.py` pe modelul XGBoost (Stage 2), evaluat pe 8.520 de rânduri din sesiunile holdout. Figuri: `reports/figures/thesis/`: `shap_beeswarm`, `shap_importance_bar`, `shap_dependence_dwell_time`, `shap_dependence_handoff_history`.

## De ce SHAP

Pentru a depăși limitele importanței bazate pe impuritate (specifică fiecărui model și sensibilă la cardinalitate), am folosit valorile **SHAP** (SHapley Additive exPlanations), care atribuie fiecărei caracteristici contribuția sa aditivă la predicția individuală, pe scala log-odds. Aceasta oferă atât o ierarhie globală robustă, cât și explicații locale și interacțiuni.

## Importanța globală (medie |SHAP|)

Ierarhia globală confirmă și nuanțează concluziile din analiza de importanță RF:

1. **Timpul de staționare pe celulă** (`dwell_time`) — dominant, cu un impact mediu de aproape un ordin de mărime peste restul.
2. Aria de localizare (`area`) și **contextul temporal ciclic** (oră), care surprind tipare de trafic/mobilitate.
3. Modul de transport (mașină) și **viteza medie mobilă (15 s)** — dinamica deplasării.
4. **Identitatea celulei codificată pe țintă** (`cid`), care confirmă rolul celulelor „magistrale" de handover.

## Interacțiunea cheie: timp de staționare × identitatea celulei

Graficul de dependență `shap_dependence_dwell_time` arată un efect **monoton** clar: la timp de staționare mic, contribuția SHAP este puternic negativă (modelul împinge predicția spre „fără handover iminent"), iar pe măsură ce timpul de staționare crește, contribuția devine puternic pozitivă (handover iminent), cu un prag de inversiune în jurul a 40–50 de secunde. Codarea prin culoare a ratei istorice de handover a celulei (CID target-encoded) evidențiază **interacțiunea**: pentru același timp de staționare, celulele cu istoric de handover diferit primesc contribuții diferite.

## Observație critică (onestitate metodologică)

Structura „în trepte" vizibilă în graficul de dependență, împreună cu dominanța masivă a `dwell_time`, confirmă vizual ipoteza din capitolul de feature engineering: **expandarea radio pe segmente egale induce o regularitate cvasi-periodică a tranzițiilor**, pe care modelul o exploatează prin timpul de staționare. Prin urmare, o parte importantă a puterii predictive provine din această regularitate construită, nu exclusiv dintr-un context radio independent. Analiza SHAP face acest mecanism transparent și verificabil, ceea ce întărește, nu slăbește, rigoarea lucrării: limita este identificată cantitativ, nu ascunsă. Cu loguri radio dense reale (RSRP/RSRQ), timpul de staționare ar fi guvernat de declanșatori fizici, nu de segmentare.
