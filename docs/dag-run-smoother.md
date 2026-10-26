# Composer run guide – `airbnb_census_elt_bronze_load`

**Submission DAG:** `dags/airbnb_census_elt_dag.py`  
**DAG id:** `airbnb_census_elt_bronze_load`  
**Author:** Akshat Dubey  

Do **not** use the deprecated `airbnb_sydney_elt_dag*.py` stubs.

## Connection & paths

| Item | Value |
|------|--------|
| Airflow connection | `postgres` |
| Data root on Composer | `/home/airflow/gcs/data/` (bucket `data/` folder) |
| Schedule | `None` / manual only |
| dbt | **Not** triggered by Airflow – run in dbt Cloud after each month |

### Expected files

```
/home/airflow/gcs/data/listings/05_2020.csv … 04_2021.csv
/home/airflow/gcs/data/Census LGA/2016Census_G01_NSW_LGA.csv
/home/airflow/gcs/data/Census LGA/2016Census_G02_NSW_LGA.csv
/home/airflow/gcs/data/NSW_LGA/NSW_LGA_CODE.csv
/home/airflow/gcs/data/NSW_LGA/NSW_LGA_SUBURB.csv
```

### Bronze tables (created by the DAG)

- `bronze.raw_listings` (+ `source_file`, `loaded_at`) — all listing cols TEXT  
- `bronze.raw_census_g01` — G01 subset (totals + age bands)  
- `bronze.raw_census_g02`  
- `bronze.raw_nsw_lga_code`  
- `bronze.raw_nsw_lga_suburb`  

`part_1.sql` mirrors the same DDL for marking; the DAG’s `create_bronze_tables` task is enough in practice.

## Trigger months (Part 1 then Part 3)

| Trigger conf | File loaded |
|--------------|-------------|
| `{}` or omit | `05_2020.csv` (default) |
| `{"month": "06_2020"}` | `06_2020.csv` |
| … | … |
| `{"month": "04_2021"}` | `04_2021.csv` |

Each run: validate → create tables → **reload** Census/LGA references → upsert that Airbnb month (`DELETE` by `source_file` then `COPY`).

Continue **chronologically**. After **each** successful month:

## dbt Cloud order (manual)

1. `dbt run --select listings_clean host_entity listing_entity property_entity lga_code_clean lga_suburb_clean census_g01_clean census_g02_clean`  
   (or `dbt run --select silver.*` / full project once models exist)
2. `dbt snapshot` (host / listing / property timestamp SCD2)
3. `dbt run` (Gold dims + fact + datamart views)

Optional first-time: `dbt run --full-refresh` then snapshots as above after Bronze has data.

## DBeaver verify

```sql
SELECT COUNT(*) FROM bronze.raw_listings;
SELECT source_file, COUNT(*) FROM bronze.raw_listings GROUP BY 1 ORDER BY 1;
SELECT COUNT(*) FROM bronze.raw_census_g01;
SELECT COUNT(*) FROM bronze.raw_census_g02;
SELECT COUNT(*) FROM bronze.raw_nsw_lga_code;
SELECT COUNT(*) FROM bronze.raw_nsw_lga_suburb;
```
