# Assignment 3 – Sydney Airbnb + Census ELT (Rohan Chaudhary)

Big Data Engineering: Airflow → Bronze Postgres, dbt Cloud Medallion (Bronze / Silver / Gold / Datamart), Part 4 ad-hoc SQL.

## Deliverables

| File | Description |
|------|-------------|
| `part_1.sql` | Bronze DDL for listings, Census G01/G02, LGA mappings |
| `dags/airbnb_sydney_elt_dag.py` | Combined Part 1+3 DAG (`schedule_interval=None`) |
| `part_4.sql` | Ad-hoc queries for Part 4 a–e (Sydney, 12 months, *100 rates) |
| `dbt_airbnb/` | dbt project: `models/`, `snapshots/`, `dbt_project.yml` |
| `data/` | Census + LGA reference files; listings placeholders |
| `docs/Assignment_3_Handover_Report.md` | Handover report (≤3000 words) |
| `Assignment_3_Rohan_Chaudhary.zip` | Submission bundle |

## Quick start (GCP + dbt Cloud)

1. Create Cloud SQL Postgres + Cloud Composer; set Airflow connection `postgres_bde`.
2. Run `part_1.sql` in DBeaver.
3. Upload `data/` (including 12 Airbnb month CSVs) to the Composer GCS bucket; set `BUCKET_NAME` in the DAG.
4. Trigger DAG once (truncates + loads static + all months). For Part 3 pedagogy, clear months after the first and re-enable tasks month-by-month, running **dbt Cloud manually** after each month (`dbt snapshot` then `dbt run`).
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
