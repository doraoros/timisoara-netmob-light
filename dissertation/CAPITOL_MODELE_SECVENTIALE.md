# Modele secvențiale profunde (GRU, Transformer) vs. ingineria caracteristicilor

> Capitol destinat disertației (limba română). Rezultate generate de `scripts/experiment_sequential.py`; figura aferentă este `reports/figures/thesis/sequential_comparison.png`.

## 1. Întrebarea de cercetare

Capitolul de inginerie a caracteristicilor a arătat că modelul tabular optimizat (XGBoost, Stage 2) atinge performanțe bune **doar după** ce i se furnizează caracteristici secvențiale construite manual (timp de staționare, istoric de handover, viteză netezită). Apare o întrebare firească: **un model secvențial profund poate învăța singur această structură temporală, direct din semnalele brute de la fiecare secundă, fără trăsături inginerești?**

Pentru a răspunde, am antrenat două arhitecturi secvențiale și le-am comparat cu XGBoost-ul ingineresc pe **exact același set holdout de sesiuni nevăzute**.

## 2. Construcția datelor secvențiale (fără scurgeri de informație)

Pentru fiecare moment *t* am construit o fereastră cu ultimele **30 de secunde** de context, `[t-29, t]`, strict în interiorul aceleiași sesiuni (cu umplere cu zero la începutul sesiunii), și am prezis `handoff_next(t)` (handover în următoarele 15 s).

Vectorul de trăsături per secundă conține **doar semnale brute, cauzale**: viteza, codificarea ciclică a timpului (sin/cos pentru oră, minut, zi), indicatorii derivați (`is_peak_hour`, `is_high_speed`), parametrii radio statici (frecvență, arie) și **indicatorul observabil de schimbare a celulei la momentul curent** (`handoff`). În mod deliberat, **nu** am inclus trăsăturile inginerești de tip fereastră glisantă — scopul este ca rețeaua să le „reinventeze" singură. Identitatea operatorului/tehnologiei/modului de transport a fost adăugată ca variabile statice one-hot.

Standardizarea a folosit exclusiv statisticile din setul de antrenare, iar dezechilibrul de clase a fost tratat prin `pos_weight = N_neg / N_poz ≈ 2.6` în funcția de cost `BCEWithLogitsLoss`.

## 3. Arhitecturile

- **GRU** (Gated Recurrent Unit): un strat recurent (dimensiune ascunsă 64), urmat de un clasificator liniar pe ultima stare ascunsă. ~16.800 de parametri.
- **Transformer-lite**: proiecție liniară + codificare pozițională + 2 straturi de encoder Transformer (4 capete de atenție), agregare prin medie și clasificator liniar. ~68.400 de parametri.

Ambele au fost antrenate 30 de epoci pe CPU (Adam, rată de învățare 1e-3), fără GPU.

## 4. Rezultate

| Model | Intrare | ROC-AUC | PR-AUC | Parametri |
|---|---|---:|---:|---:|
| XGBoost (Stage 2, ingineresc) | trăsături secvențiale construite | **0.792** | 0.450 | — |
| Transformer-lite | semnale brute per secundă | 0.761 | **0.513** | ~68k |
| GRU | semnale brute per secundă | 0.743 | 0.412 | ~17k |

*Prevalența clasei pozitive în holdout: 0.236.*

Interpretare:

1. **Modelele secvențiale învață dinamica direct din date brute.** Fără nicio trăsătură inginerească de tip fereastră glisantă, GRU și Transformer ating ROC-AUC de 0.74–0.76, apropiate de modelul tabular optimizat (0.792). Aceasta confirmă că structura temporală pe care XGBoost o primește „de-a gata" poate fi extrasă automat de o rețea recurentă/atențională.
2. **Pe PR-AUC, Transformer-ul depășește XGBoost** (0.513 vs. 0.450). Întrucât PR-AUC este metrica primară pentru un set dezechilibrat, acesta este un rezultat semnificativ: mecanismul de atenție surprinde mai bine tiparele clasei pozitive (handover iminent) decât arborele de decizie pe trăsături agregate.
3. **Compromis performanță–complexitate.** XGBoost rămâne cel mai bun pe ROC-AUC și este extrem de ușor (0.57 MB, vezi capitolul de robustețe), deci o alegere excelentă pentru implementarea pe terminal. Transformer-ul oferă cel mai bun PR-AUC cu un cost computațional mai mare. Cele două abordări sunt **complementare**, nu concurente.

## 5. Concluzie

Comparația validează ipoteza centrală a lucrării: predicția handover-ului iminent este un **fenomen secvențial**. Atât abordarea tabulară cu inginerie a caracteristicilor, cât și modelele secvențiale profunde captează acest semnal, prima prin trăsături explicite, celelalte prin învățarea automată a dependențelor temporale. Faptul că un Transformer antrenat pe semnale brute depășește XGBoost pe PR-AUC întărește argumentul că **istoricul recent de mobilitate este un proxy valid pentru indicatorii radio fizici absenți (RSRP/RSRQ/SINR)**, și deschide o direcție clară de continuare doctorală: arhitecturi secvențiale pe loguri radio dense, reale.
