# Cloud Composer DAG – smoother Bronze loads (v2)

**Use:** `dags/airbnb_sydney_elt_dag_v2.py`  
**DAG id:** `sydney_airbnb_bronze_elt_v2`  
**Owner:** Akshat Dubey  

Pause or ignore the legacy DAG `sydney_airbnb_bronze_elt` (`airbnb_sydney_elt_dag.py`).

## Why v2 is faster / easier

| v1 (legacy) | v2 (preferred) |
|-------------|----------------|
| Row-by-row `INSERT` | Postgres `COPY FROM STDIN` |
| Hard-coded bucket / always all months | Airflow Variables – change month without re-upload |
| Always truncates | `airbnb_truncate` on/off |

`schedule_interval=None`, `max_active_runs=1`, `catchup=False`. **dbt is not triggered** – run dbt Cloud manually after each successful month load.

## Airflow Variables

Set under **Admin → Variables** (or `airflow variables set …`):

| Variable | Example | Meaning |
|----------|---------|---------|
| `airbnb_gcs_bucket` | `us-central1-bde-xxxx-bucket` | Composer data bucket (**no** `gs://` prefix). Fallback: edit `BUCKET_FALLBACK` in the DAG once. |
| `airbnb_load_month` | `05_2020` or `all` | One month file stem, or full chronological chain. |
| `airbnb_truncate` | `true` / `false` | `true` on **first** run only (wipes Bronze). Later months: `false`. |
| `airbnb_load_static` | `false` (optional) | Force re-load Census/LGA when truncate is already `false`. |

Connection id remains **`postgres_bde`**.

## GCS upload paths (unchanged)

```
gs://<airbnb_gcs_bucket>/
  data/Census LGA/2016Census_G01_NSW_LGA.csv
  data/Census LGA/2016Census_G02_NSW_LGA.csv
  data/NSW_LGA/NSW_LGA_CODE.csv
  data/NSW_LGA/NSW_LGA_SUBURB.csv
  data/listings/05_2020.csv
  data/listings/06_2020.csv
  … through …
  data/listings/04_2021.csv
```

Run `part_1.sql` in DBeaver before the first DAG trigger so Bronze tables exist.

## First run (Part 1)

1. Upload Census + LGA + at least `05_2020.csv` (or all 12 if using `all`).
2. Variables: `airbnb_gcs_bucket=<bucket>`, `airbnb_truncate=true`, `airbnb_load_month=05_2020` (or `all`).
3. Trigger `sydney_airbnb_bronze_elt_v2`.
4. Run **dbt snapshot** then **dbt run** in dbt Cloud.

## Later months (Part 3) – no DAG re-upload

1. Ensure `data/listings/{next_month}.csv` is in the bucket.
2. Set `airbnb_truncate=false`.
3. Set `airbnb_load_month` to the next stem (e.g. `06_2020`). Other month tasks no-op quickly.
4. Trigger the **same** DAG again → only that month `COPY`s into `bronze.listings_raw`.
5. Manually run dbt again; repeat chronologically through `04_2021`.

Optional: set `airbnb_load_month=all` with `airbnb_truncate=true` once to reload everything in order (static in parallel after truncate, months chained).

## Checklist if a trigger feels “stuck”

- Confirm Variable `airbnb_gcs_bucket` matches the bucket that actually has `data/…`.
- Confirm connection `postgres_bde` points at Cloud SQL and Bronze DDL was applied.
- Prefer v2 DAG; pause legacy v1 so you are not waiting on INSERT loops.
- Single-month mode keeps each run short; avoid `all` until static files are confirmed.
