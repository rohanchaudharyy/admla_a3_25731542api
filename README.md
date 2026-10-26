# Assignment 3 – Sydney Airbnb + Census ELT (Akshat Dubey)

Big Data Engineering: Composer Bronze load → dbt Medallion → Part 4 ad-hoc SQL.

## Deliverables

| File | Description |
|------|-------------|
| `part_1.sql` | Bronze DDL matching the working DAG (`bronze.raw_*`) |
| `dags/airbnb_census_elt_dag.py` | **Submission DAG** – `airbnb_census_elt_bronze_load` |
| `part_4.sql` | Part 4 a–e (Sydney, 12 months, ×100 rates) |
| `dbt_airbnb/` | models, snapshots, `dbt_project.yml` |
| `data/` | Census + LGA reference extracts; listings schema note |
| `docs/dag-run-smoother.md` | How to trigger months + dbt order |
| `docs/Assignment_3_Handover_Report.md` | Handover report |
| `Assignment_3_Akshat_Dubey.zip` | Submission bundle |

Deprecated stubs (not for Composer): `dags/airbnb_sydney_elt_dag.py`, `dags/airbnb_sydney_elt_dag_v2.py`.

## Quick start

1. Cloud SQL Postgres + Cloud Composer; Airflow connection id **`postgres`**.
2. Upload CSVs under the Composer bucket `data/` tree (see `docs/dag-run-smoother.md`).
3. Deploy **`dags/airbnb_census_elt_dag.py`** only.
4. Trigger with `{}` (loads `05_2020`), then `{"month":"06_2020"}` … through `04_2021`.
5. After each month: dbt silver entities → `dbt snapshot` → `dbt run`.
6. Verify: `SELECT COUNT(*) FROM bronze.raw_listings;`
7. Run `part_4.sql` for business questions.

## Metric definitions

- Active listing: `has_availability = 't'`
- Rates / MoM change: × **100**
- Stays (active) = **30 − availability_30**
- Estimated revenue (active) = stays × price

## Schemas

- `bronze` – raw TEXT tables from the DAG  
- `silver` – cleaned types + SCD2 snapshots  
- `gold` – ≥14 dimensions, Census G01/G02 refs, star fact  
- `datamart` – three assignment views  
