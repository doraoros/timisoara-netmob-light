# Limitări metodologice

Constrângerile de mai jos definesc ce pot și ce nu pot concluziona din datele folosite.

## Caracterul datelor utilizate

**Traiectoriile GPS** provin din fișiere simulate la rezoluție de 1 Hz (`gps_simulated/`), aliniate pe rute reale din Timișoara. Nu am dispus de înregistrări GNSS continue, capturate direct din aplicație pe durata deplasării pe teren. Această alegere mi-a permis un timeline uniform și reproductibil, însă nu reflectă erorile și discontinuitățile tipice unei măsurători live.

**Măsurătorile radio** au fost exportate din NetMonster și salvate ca snapshot-uri JSON (`notes/`). Per sesiune am obținut relativ puține înregistrări de celulă. Pentru a le alinia seriei GPS am aplicat o procedură de expandare pe segmente egale ale traseului, astfel încât aceeași celulă rămâne activă pe intervale consecutive. Această ipoteză aproximează comportamentul de campare pe celulă, dar **nu reprezintă un log la nivel PHY la 1 Hz** și nu include mărimi precum RSRP, RSRQ sau SINR în exportul folosit.

**Măsurătorile de throughput** provin din aplicația Speedtest și sunt punctuale (aproximativ 218 de teste agregate). Le-am atașat timeline-ului principal prin sincronizare temporală (toleranță de până la 3 ore) și filtru de distanță (≤15 km). Rezultatul este util pentru analize exploratorii, nu pentru o corelație strictă throughput–handoff la rezoluție de secundă.

## Concluzii pe care le consider susținute de date

Pe baza constrângerilor de mai sus, consider că pot formula în mod responsabil următoarele tipuri de rezultate:

- asocierea dintre **contextul de mobilitate** (rută, viteză, oră, mod de transport, operator asociat celulei) și **probabilitatea unui handoff iminent** în următoarele 15 secunde;
- comparații **descriptive** între operatori și rute, atât la nivel de rată a handoff-ului iminent, cât și al throughput-ului, acolo unde măsurătorile Speedtest s-au putut alinia temporal și spațial;
- un **model predictiv exploratoriu**, evaluat prin validare pe sesiuni noi (`GroupShuffleSplit`, `GroupKFold`) și raportat față de baseline-uri simple (majority, stratified).

## Concluzii pe care nu le consider justificate

Conștient de limitele metodologice, **nu** pretind că studiul:

- reproduce campaniile extinse de tip NYU-METS (throughput continuu, ex. iPerf la 1 s);
- identifică trigger-e de handover conform specificațiilor 3GPP pe baza RSRP/RSRQ, deoarece aceste mărimi nu sunt disponibile în setul de date procesat;
- oferă un benchmark național de performanță a rețelelor mobile, fără a ține cont de ipoteza de expandare radio și de dezechilibrul clasei `handoff_next` (~27% pozitivi).

## Artefact identificat prin interpretabilitate (SHAP)

Analiza SHAP pe XGBoost (Stage 2) arată că **`dwell_time`** contribuie cel mai mult la predicție. Graficul de dependență are trepte — pattern așteptat dacă expandarea pe segmente egale impune o regularitate în schimbările de celulă. O parte din performanța Stage 2 vine deci din structura datelor, nu doar din context radio independent. Menționez asta ca limită, nu ca surpriză ascunsă.

## Poziționare

Lucrarea e un studiu de mobilitate urbană în Timișoara, cu resurse limitate dar pipeline reproductibil. Contribuția principală: organizarea datelor locale și testarea predictibilității schimbării iminente de celulă în LTE. Nu e evaluare exhaustivă a operatorilor la scară națională.
