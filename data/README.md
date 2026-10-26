# Dataset layout (Composer `/home/airflow/gcs/data/`)

Matches `dags/airbnb_census_elt_dag.py`:

```
data/
  Census LGA/
    2016Census_G01_NSW_LGA.csv
    2016Census_G02_NSW_LGA.csv
  NSW_LGA/
    NSW_LGA_CODE.csv
    NSW_LGA_SUBURB.csv
  listings/
    05_2020.csv … 04_2021.csv
```

## Included in this repo
- Census G01/G02 NSW LGA extracts under `Census LGA/`
- `NSW_LGA_CODE.csv` / `NSW_LGA_SUBURB.csv` under `NSW_LGA/`
- Prefer course-provided mapping/Airbnb files when available

## Airbnb months
Upload curated `05_2020.csv` … `04_2021.csv` from Canvas. Trigger the DAG with `{"month":"MM_YYYY"}` (default `05_2020`).

Expected listing headers (case-insensitive):  
`listing_id, scrape_id, scraped_date, host_id, host_name, host_since, host_is_superhost, host_neighbourhood, listing_neighbourhood, property_type, room_type, accommodates, price, has_availability, availability_30, number_of_reviews, review_scores_*`
