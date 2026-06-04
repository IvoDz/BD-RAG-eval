# RAG izvērtēšanas analīze

Pētījums RAG sistēmu izvērtēšanai un analīzei bakalaura darba ietvaros.

Projektā ir apkopoti pētījumā izmantotie koda fragmenti un dati, kas lietoti RAG prototipa izvērtēšanai un analīzei.

## Struktūra

### `data/`

Satur testa datu kopu un priekšapstrādes starprezultātus.

**`data/raw/`** — sākotnēji izgūti dati bez priekšapstrādes, iekļaujot avotus:

- `combined_rag_base.csv` - Sākotnējo datu apkopojums
- `laws-lv-raw.csv` - Latvijas likumi
- `news-lv-raw.csv` - Latvijas ziņas
- `wiki-en-raw.csv` - Wikipedia raksti

**`data/processed/`** — strukturētas un priekšapstrādātas datu kopas:

- **`data/processed/base/`** - Priekšapastrādātas datu kopas pa kategorijām
- **`data/processed/final/`** - Gala datu kopas pirms manuālām korekcijām
- **`data/results/`** - Datu kopas pa kategorijām, apvienojumā ar RAG izvaddatiem

### `eval/`

Izvērtēšanas rezultāti.

- `FULL_global_evaluation_matrix.csv` - Metriku un ietvara rādītāju apkopojums testa kopā
- `HUMAN_grading_subset.csv` - Manuālās izvērtēšanas rezultāti apakškopā

### `scripts/`

Datu priekšapstrādes skripti tulkošanai, trokšņa sintēzei un datu izguvei.

## Datu formāts
Gala datu kopas ir pieejamas **JSONL** formātā.
Atsevišķi sākotnējie dati tiek saglabāti **CSV** formātā.
