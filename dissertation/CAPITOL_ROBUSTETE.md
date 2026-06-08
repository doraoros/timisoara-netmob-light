# Robustețe, generalizare și fezabilitate de implementare

> Capitol destinat disertației (limba română). Rezultate generate de `scripts/experiment_robustness.py`; figurile sunt în `reports/figures/thesis/` (`robustness_per_session.png`, `robustness_leave_one_route_out.png`, `robustness_gps_jitter.png`, `robustness_sampling_rate.png`). Toate experimentele folosesc modelul optimizat (XGBoost, Stage 2) descris în capitolul de inginerie a caracteristicilor.

O performanță medie bună pe un set holdout nu este suficientă pentru o lucrare matură. Acest capitol evaluează **distribuția** performanței (nu doar media), **generalizarea** la rute nevăzute, **robustețea** la perturbații realiste (zgomot GPS, rată de eșantionare redusă) și **fezabilitatea implementării** pe terminal (latență, dimensiune model).

## 1. Distribuția performanței per sesiune

În loc să raportăm o singură valoare medie, am calculat metricile **per sesiune** folosind predicțiile out-of-fold ale validării încrucișate pe grupuri (GroupKFold). Au fost punctate doar sesiunile cu ambele clase prezente.

| Metrică | Mediană | IQR [Q1, Q3] | Min | Max |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.95 | [0.92, 0.97] | 0.80 | 1.00 |
| PR-AUC | 0.86 | [0.64, 0.92] | 0.28 | 0.99 |

Observăm o performanță **per sesiune foarte ridicată**, dar cu o coadă de sesiuni dificile (PR-AUC coboară până la 0.28). Această valoare ridicată trebuie interpretată critic: în interiorul unei singure sesiuni, structura cvasi-periodică a tranzițiilor (indusă de expandarea radio pe segmente egale) face problema aproape trivială. Prin urmare, **metrica per sesiune supraestimează performanța reală**, iar evaluarea corectă rămâne cea pe sesiuni/rute nevăzute (secțiunile următoare). Raportarea acestei distribuții, inclusiv a outlierilor, este o dovadă de transparență metodologică.

## 2. Generalizare la rute nevăzute (Leave-One-Route-Out)

Cel mai sever test de generalizare: antrenăm pe 8 rute și testăm pe **a 9-a rută, complet nevăzută**, repetând pentru fiecare rută.

- **ROC-AUC mediu pe rute nevăzute: 0.869**
- **PR-AUC mediu pe rute nevăzute: 0.651**

Toate rutele depășesc nivelul aleator la ROC-AUC. Rezultatele variază de la rute „ușoare" (ex. `aem_complex`: ROC-AUC ≈ 0.98, PR-AUC ≈ 0.97) la rute dificile (`victoriei_michelangelo_complex`: ROC-AUC 0.74, PR-AUC 0.22; `iulius_airport`: ROC-AUC 0.71, PR-AUC 0.43). Faptul că modelul **generalizează la geometrii de traseu pe care nu le-a văzut** indică învățarea unor tipare de mobilitate transferabile, nu doar memorarea rutelor de antrenare.

## 3. Robustețe la zgomotul GPS

Am injectat zgomot gaussian pe viteză (σ ∈ {0, 1, 2, 5, 10} km/h) în setul de test și am recalculat caracteristicile cinematice derivate, păstrând modelul antrenat pe date curate.

Rezultatul este remarcabil: performanța rămâne **practic neschimbată** (ROC-AUC ≈ 0.82–0.825) chiar și la σ = 10 km/h. Explicația este coerentă cu analiza de importanță: predicția este dominată de **dinamica rețelei** (timp de staționare, istoric de handover, diversitate PCI), nu de viteza instantanee. Astfel, **erorile tipice ale GPS-ului nu compromit predicția**, ceea ce este esențial pentru o aplicație reală care folosește GPS de consum.

## 4. Robustețe la rata de eșantionare (compromis baterie–performanță)

Am simulat o eșantionare mai rară (perioadă de 1, 2, 3 și 5 secunde) prin subeșantionarea timeline-ului de 1 Hz, cu reantrenare la fiecare nivel.

| Perioadă eșantionare | ROC-AUC | PR-AUC |
|---|---:|---:|
| 1 s (1 Hz, complet) | 0.769 | 0.418 |
| 2 s | 0.603 | 0.269 |
| 3 s | 0.598 | 0.258 |
| 5 s | 0.491 | 0.222 |

Concluzie practică clară: **modelul necesită eșantionarea la 1 Hz**. Sub această rată, ferestrele glisante își pierd rezoluția temporală, iar performanța se degradează rapid spre nivelul aleator. Aceasta este o constrângere de proiectare relevantă pentru implementarea pe terminal (compromis între consumul de baterie și capacitatea de predicție).

## 5. Fezabilitatea implementării pe terminal (edge)

Am măsurat costul de inferență al modelului optimizat (XGBoost, 300 de arbori, Stage 2):

| Indicator | Valoare |
|---|---:|
| Dimensiune model (serializat) | ≈ 0.57 MB |
| Latență inferență, lot (per eșantion) | ≈ 0.0075 ms |
| Debit, inferență în lot | ≈ 134.000 eșantioane/s |
| Latență per eșantion individual (pipeline Python) | ≈ 13 ms |

Modelul însuși este extrem de rapid (sub 0.01 ms/eșantion în lot). Latența de ≈ 13 ms pentru un eșantion individual reflectă supraîncărcarea pipeline-ului Python (transformările scikit-learn pe un singur rând) și ar fi eliminabilă printr-o cale de inferență nativă/compilată. În orice caz, cu un orizont de predicție de 15 secunde și o frecvență de inferență de 1 Hz, **bugetul de timp este amplu respectat**, iar dimensiunea de 0.57 MB confirmă fezabilitatea rulării direct pe un terminal mobil (arhitectură lightweight).

## 6. Sinteză

Experimentele de mai sus susțin patru afirmații de robustețe și relevanță practică:

1. Performanța per sesiune este ridicată, dar trebuie interpretată critic (periodicitate construită); evaluarea onestă se face pe rute/sesiuni nevăzute.
2. Modelul **generalizează la rute nevăzute** (ROC-AUC 0.87 mediu, leave-one-route-out).
3. Modelul este **robust la zgomotul GPS** (insensibil la σ până la 10 km/h), deoarece se bazează pe dinamica rețelei, nu pe viteza brută.
4. Modelul **necesită eșantionare la 1 Hz** și este **suficient de ușor pentru implementare pe terminal** (0.57 MB, latență neglijabilă față de orizontul de 15 s).

Aceste rezultate transformă studiul dintr-o demonstrație de feasibility într-o evaluare cu relevanță operațională, asumând explicit atât punctele forte, cât și limitele induse de natura datelor.
