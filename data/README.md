# Structura datelor

Documentație pentru folderele `raw/`, `interim/` și `processed/`, plus schema datasetului principal.

---

## `raw/`

| Folder | Conținut |
|--------|----------|
| `notes/` | Export NetMonster (JSON în `.txt`), câte un fișier per sesiune |
| `gps_simulated/` | Trasee GPS la 1 Hz **sintetice/aliniate pe rute reale** (nu log GNSS live) |
| `speedtest/` | Rezultate Speedtest (CSV), măsurători **punctuale** |
| `netmonster/` | Opțional: export app `log.json` |

---

## `interim/`

| Fișier | Rol |
|--------|-----|
| `sessions.csv` | Mapare `session_id` → `json_file` + `gps_file` |

**16 sesiuni** pe **9 rute** urbane din Timișoara.

---

## `processed/`

| Fișier | Rol |
|--------|-----|
| `dataset_radio_gps.csv` | **Dataset principal** (~31k rânduri, 1 Hz) |
| `speedtest_all.csv` | Speedtest agregat |
| `nperf_*.png` | Hărți referință operatori (anexă) |

---

## Schema dataset principal

Descriere completă, mașină-citibilă (tip, unitate, descriere per coloană + trăsături inginerite Stage 2):

→ [`schema.json`](schema.json)

### Coloane principale (rezumat)

| Coloană | Descriere |
|---------|-----------|
| `timestamp`, `latitude`, `longitude`, `speed_kmh` | Mobilitate (GPS sintetic/aliniat) |
| `route_id`, `transport_mode`, `is_peak_hour` | Context cursă |
| `operator`, `technology`, `cid`, `code`, `frequency` | Radio (NetMonster snapshot) |
| `handoff`, `handoff_next` | Etichete (schimbare celulă / schimbare iminentă în 15 s) |
| `dl_mbps`, `ul_mbps`, `ping_ms`, `speedtest_matched` | Throughput (**EDA only**, sparse) |
| `session_id` | Cheie de grupare pentru validare ML |
| `radio_cell_index` | Artefact expandare radio — **exclus din Stage 2** |

---

## Regenerare

```bash
python -m src.utils.build_radio_gps_dataset
python -m src.utils.validate_dataset
```

Vezi și [`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) (expandare radio, etichete) și [`../reports/text/limitations.md`](../reports/text/limitations.md).
