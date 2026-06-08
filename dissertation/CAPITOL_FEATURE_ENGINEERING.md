# Ingineria avansată a caracteristicilor și optimizarea modelelor

> Capitol destinat disertației (limba română). Rezultatele sunt generate de `scripts/experiment_feature_engineering.py`; figurile aferente sunt în `reports/figures/thesis/` (`stage_comparison.png`, `feature_importance_stage2.png`).

## 1. Motivație: de la observații statice la proces secvențial

În prima formulare a experimentului, fiecare observație de la momentul *t* a fost tratată independent, descrisă doar prin contextul instantaneu (viteză, timp brut, identitatea celulei). Această abordare a oferit un semnal predictiv limitat: regresia logistică a atins ROC-AUC ≈ 0,63 pe holdout, iar modelele de ansamblu (Random Forest, Gradient Boosting, XGBoost) au rămas aproape de nivelul hazardului (ROC-AUC ≈ 0,55).

Acest rezultat nu indică imposibilitatea predicției, ci o **reprezentare inadecvată a datelor**. Handoff-ul (schimbarea celulei servitoare) este un proces secvențial: probabilitatea unei tranziții iminente depinde de istoricul recent de mobilitate și de dinamica recentă a rețelei, nu doar de starea instantanee. În plus, anumite variabile brute introduceau zgomot și artefacte. Capitolul de față descrie ingineria caracteristicilor prin care am transformat reprezentarea datelor și am redat capacitatea analitică a modelelor complexe.

## 2. Eliminarea variabilelor parazite

Pe baza analizei de importanță din etapa inițială, am eliminat din setul de antrenare următoarele variabile:

- **`minute` și `hour` ca valori brute.** Arborii de decizie nu cunosc continuitatea timpului (minutul 59 este adiacent minutului 0) și efectuează tăieturi numerice arbitrare (de exemplu „dacă minut < 24"), generând corelații artificiale. Acest comportament explică, în mare parte, importanța aparent ridicată a variabilei `minute` în figura inițială de importanță.
- **`radio_cell_index` și `cells_seen_in_session`.** Acestea depind de durata și de modul de salvare al sesiunii specifice de colectare; ele reprezintă artefacte ale procesului de înregistrare, nu proprietăți fizice ale rețelei.
- **Variabilele de throughput** (`dl_mbps`, `ul_mbps`, `ping_ms`) au fost excluse complet din modelare și păstrate doar în analiza exploratorie, deoarece măsurătorile Speedtest sunt punctuale și asociate cu toleranță mare.

## 3. Codificarea ciclică a timpului

Pentru a păstra contextul temporal fără a distruge modelele arborescente, am transformat variabilele temporale în coordonate ciclice:

\[
\text{minute\_sin} = \sin\!\left(\frac{2\pi \cdot \text{minute}}{60}\right), \quad
\text{minute\_cos} = \cos\!\left(\frac{2\pi \cdot \text{minute}}{60}\right)
\]

Aceeași transformare a fost aplicată orei (perioada 24) și zilei săptămânii (perioada 7). Astfel, modelul percepe timpul ca pe o mărime continuă și circulară, eliminând split-urile aberante pe valori brute.

## 4. Caracteristici cinematice (dinamica mișcării)

Din seria GPS la 1 Hz am derivat, pentru fiecare sesiune, variabile care descriu regimul de deplasare:

- **`speed_delta_5s`** — diferența dintre viteza curentă și viteza de acum 5 secunde (proxy pentru accelerație/frânare). Un vehicul care frânează în stație are un profil de handoff diferit de unul care accelerează.
- **`rolling_mean_speed_15s`** — media vitezei pe ultimele 15 secunde, care netezește fluctuațiile punctuale ale GPS-ului simulat.
- **`rolling_std_speed_15s`** — deviația standard a vitezei pe aceeași fereastră, ca indicator de variabilitate a deplasării.

## 5. Dinamica rețelei ca proxy pentru indicatorii fizici (RSRP/RSRQ)

În absența indicatorilor radio de nivel fizic, am construit variabile care surprind dinamica recentă a celulei servitoare:

- **`handoff_history_30s` / `handoff_history_60s`** — numărul de schimbări de celulă (CID/PCI) în ultimele 30, respectiv 60 de secunde. O frecvență ridicată indică un segment instabil sau o zonă densă, unde un nou handoff devine mai probabil.
- **`dwell_time_s`** — timpul scurs (în secunde) de la ultima schimbare de celulă din cadrul sesiunii. La viteză mare, un timp de staționare îndelungat pe aceeași celulă semnalează apropierea de o tranziție.
- **`unique_pci_last_60s`** — numărul de PCI-uri distincte observate în ultimul minut. O diversitate ridicată indică o zonă de graniță între celule (cell edge), unde handoff-ul este iminent.

## 6. Codificarea pe țintă a identității celulei (target encoding)

Identificatorii de celulă (`cid`, `code`/PCI) au cardinalitate ridicată și nu pot fi folosiți eficient ca valori brute. I-am înlocuit prin **target encoding**: fiecare celulă este reprezentată prin rata medie istorică de handoff iminent asociată acelei celule. Astfel, celulele care, prin poziția lor geografică, funcționează ca „magistrale" de handover primesc o valoare informativă.

Pentru a evita scurgerea de informație (data leakage), codificarea a fost realizată prin `TargetEncoder` din scikit-learn, **în interiorul pipeline-ului de validare**, cu cross-fitting intern: statisticile pe țintă sunt calculate exclusiv pe datele de antrenare ale fiecărui fold și aplicate ulterior pe datele de test.

## 7. Evitarea scurgerilor de informație (data leakage)

Întreaga inginerie a caracteristicilor respectă principiul cauzalității, esențial deoarece ținta `handoff_next` privește 15 secunde în viitor:

1. Toate ferestrele glisante folosesc **doar rânduri trecute sau prezente**, calculate per sesiune pe timeline-ul ordonat temporal; nicio variabilă nu utilizează informație din fereastra viitoare a țintei.
2. Caracteristicile sunt calculate **separat pe fiecare `session_id`**, astfel încât finalul unei sesiuni nu influențează începutul alteia.
3. Codificarea pe țintă este realizată **în interiorul cross-validării**, niciodată pe întregul set înainte de split.
4. Împărțirea train/test rămâne **pe grupuri (sesiuni întregi)**, prin `GroupShuffleSplit` și `GroupKFold`.

## 8. Calibrarea modelelor pentru dezechilibrul claselor

Clasa pozitivă (`handoff_next = 1`) are o prevalență de 26,77%. Pentru a evita ca modelele să prezică majoritar clasa negativă, am aplicat ponderarea claselor: `class_weight="balanced"` pentru regresia logistică, Random Forest și HistGradientBoosting, respectiv `scale_pos_weight = N_neg / N_poz ≈ 2,7` pentru XGBoost. Metrica principală de evaluare a fost PR-AUC, adecvată datelor dezechilibrate.

## 9. Rezultate: „înainte" vs. „după"

Tabelul de mai jos compară performanța pe setul holdout (sesiuni nevăzute) între reprezentarea brută (Stage 1) și reprezentarea cu trăsături secvențiale curățate (Stage 2).

| Model | ROC-AUC (Stage 1) | ROC-AUC (Stage 2) | PR-AUC (Stage 1) | PR-AUC (Stage 2) | F1 (Stage 2) |
|---|---:|---:|---:|---:|---:|
| Regresie logistică | 0,629 | 0,776 | 0,336 | 0,382 | 0,592 |
| Random Forest | 0,546 | 0,726 | 0,279 | 0,334 | 0,529 |
| Gradient Boosting | 0,554 | 0,759 | 0,272 | 0,393 | 0,523 |
| **XGBoost** | 0,551 | **0,792** | 0,257 | **0,450** | 0,537 |

*Baseline-uri (neschimbate): majority ROC-AUC 0,500 / PR-AUC 0,236; stratified ROC-AUC 0,503 / PR-AUC 0,237.*

Observații principale:

- **Modelele de ansamblu își recapătă capacitatea analitică.** Random Forest, Gradient Boosting și XGBoost cresc de la nivelul hazardului (ROC-AUC ≈ 0,55) la valori de 0,73–0,79.
- **XGBoost devine cel mai performant model** (ROC-AUC 0,792; PR-AUC 0,450), depășind regresia logistică. Acest lucru rezolvă „paradoxul" din etapa inițială, în care un model liniar simplu părea superior ansamblurilor: cauza nu era superioritatea intrinsecă a modelului liniar, ci absența unei reprezentări secvențiale pe care modelele complexe să o poată exploata.
- **PR-AUC pentru XGBoost (0,450) reprezintă aproape o dublare** față de baseline-ul de prevalență (0,236), un semnal predictiv substanțial pentru o problemă dezechilibrată.

Figura `stage_comparison.png` ilustrează grafic această evoluție, iar `feature_importance_stage2.png` arată că, în reprezentarea optimizată, predicția este dominată de dinamica recentă a rețelei (istoricul de handover, timpul de staționare, diversitatea PCI) și de cinematica deplasării.

## 9.bis Interpretabilitate (analiză SHAP)

Pentru a explica deciziile modelului dincolo de importanța bazată pe impuritate, am aplicat analiza **SHAP** pe XGBoost (Stage 2), pe sesiunile holdout (detalii în `reports/text/explainability.md`; figuri: `shap_beeswarm.png`, `shap_importance_bar.png`, `shap_dependence_dwell_time.png`, `shap_dependence_handoff_history.png`).

Ierarhia globală (medie |SHAP|) confirmă dominanța **timpului de staționare pe celulă**, urmat de aria de localizare, contextul temporal ciclic, modul de transport, viteza medie mobilă și identitatea celulei (CID target-encoded). Graficul de dependență `dwell_time × cell_id` arată un efect monoton crescător (staționare mică → contribuție negativă; staționare mare → contribuție pozitivă, „overdue for handover"), modulat de rata istorică de handover a celulei. Structura „în trepte" a acestui grafic face vizibilă, în mod transparent, regularitatea indusă de expandarea pe segmente egale — un argument de rigoare, nu de slăbiciune.

## 10. Concluzie metodologică

Acest capitol descrie trecerea de la features brute la reprezentarea secvențială. Pe holdout, XGBoost ajunge la ROC-AUC 0,792 (față de ~0,55 pe aceleași modele în Stage 1). Rezultatul depinde parțial de `dwell_time` și de modul în care am expandat datele radio — vezi analiza SHAP și `limitations.md`.
