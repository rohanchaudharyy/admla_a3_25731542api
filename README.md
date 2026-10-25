# Assignment 3 – Sydney Airbnb + Census ELT (Akshat Dubey)

Big Data Engineering: Airflow → Bronze Postgres, dbt Cloud Medallion (Bronze / Silver / Gold / Datamart), Part 4 ad-hoc SQL.

## Deliverables

| File | Description |
|------|-------------|
| `part_1.sql` | Bronze DDL for listings, Census G01/G02, LGA mappings |
| `dags/airbnb_sydney_elt_dag_v2.py` | **Preferred** Part 1+3 DAG – Postgres COPY + Variables (`schedule_interval=None`) |
| `dags/airbnb_sydney_elt_dag.py` | Legacy v1 (row INSERT) – pause on Composer |
| `docs/dag-run-smoother.md` | Composer Variables + first-run / later-month guide |
| `part_4.sql` | Ad-hoc queries for Part 4 a–e (Sydney, 12 months, *100 rates) |
| `dbt_airbnb/` | dbt project: `models/`, `snapshots/`, `dbt_project.yml` |
| `data/` | Census + LGA reference files; listings placeholders |
| `docs/Assignment_3_Handover_Report.md` | Handover report (≤3000 words) |
| `Assignment_3_Akshat_Dubey.zip` | Submission bundle |

## Quick start (GCP + dbt Cloud)

1. Create Cloud SQL Postgres + Cloud Composer; set Airflow connection `postgres_bde`.
2. Run `part_1.sql` in DBeaver.
3. Upload `data/` (including Airbnb month CSVs) to the Composer GCS bucket.
4. Deploy **`airbnb_sydney_elt_dag_v2.py`**. Set Variables `airbnb_gcs_bucket`, `airbnb_truncate`, `airbnb_load_month` (see `docs/dag-run-smoother.md`). Trigger `sydney_airbnb_bronze_elt_v2`; run **dbt Cloud manually** after each month (`dbt snapshot` then `dbt run`).
5. Run `part_4.sql` against Gold for the business questions.

## Metric definitions (authoritative)

- Active listing: `has_availability = 't'` / `true`
- Active listing rate = active / total × **100**
- Superhost rate = distinct superhosts / distinct hosts × **100**
- MoM % change = (new − old) / old × **100**
- Stays (active) = **30 − availability_30**
- Estimated revenue (active) = stays × price

## Schemas

- `bronze` – raw Airflow loads  
- `silver` – cleaned entities + SCD2 snapshots  
- `gold` – ≥14 dimensions, 2 Census refs, star fact  
- `datamart` – `dm_listing_neighbourhood`, `dm_property_type`, `dm_host_neighbourhood`
